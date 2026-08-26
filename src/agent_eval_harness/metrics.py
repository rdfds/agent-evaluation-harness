"""Failure-aware scoring, slice metrics, uncertainty, latency, and cost."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import random
from statistics import fmean, pstdev
from typing import Any, Callable, Iterable

from .models import AgentResponse, EvalCase
from .runtime import Agent


@dataclass(frozen=True)
class ScoredCase:
    case_id: str
    slice_name: str
    difficulty: str
    route_correct: bool
    answer_correct: bool
    evidence_supported: bool
    unsupported_answer: bool
    passed: bool
    failures: tuple[str, ...]
    response: AgentResponse


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _cluster_bootstrap_interval(
    records_by_case: list[list[ScoredCase]],
    value: Callable[[ScoredCase], bool],
    *,
    samples: int = 4_000,
    seed: int = 17,
) -> list[float]:
    """Bootstrap whole cases so repeated runs are not treated as independent data."""

    if not records_by_case:
        return [0.0, 0.0]
    randomizer = random.Random(seed)
    case_count = len(records_by_case)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [records_by_case[randomizer.randrange(case_count)] for _ in range(case_count)]
        values = [float(value(record)) for case_records in selected for record in case_records]
        estimates.append(fmean(values))
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [round(lower, 4), round(upper, 4)]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def score_case(case: EvalCase, response: AgentResponse) -> ScoredCase:
    route_correct = response.route == case.expected_route
    answer_correct = response.answer.strip() == case.expected_answer.strip()
    evidence_supported = set(case.required_evidence) <= set(response.evidence_ids)
    unsupported_answer = (
        case.expected_route == "abstain" and response.route != "abstain"
    ) or (
        response.route != "abstain"
        and bool(case.required_evidence)
        and not evidence_supported
        and response.answer not in {"NO_MATCH", "INSUFFICIENT_EVIDENCE"}
    )
    failures: list[str] = []
    if not route_correct:
        failures.append("route_mismatch")
    if not answer_correct:
        failures.append("answer_mismatch")
    if not evidence_supported:
        failures.append("missing_evidence")
    if unsupported_answer:
        failures.append("unsupported_answer")
    passed = not failures
    return ScoredCase(
        case_id=case.case_id,
        slice_name=case.slice_name,
        difficulty=case.difficulty,
        route_correct=route_correct,
        answer_correct=answer_correct,
        evidence_supported=evidence_supported,
        unsupported_answer=unsupported_answer,
        passed=passed,
        failures=tuple(failures),
        response=response,
    )


def _slice_summary(results: Iterable[ScoredCase]) -> dict[str, Any]:
    records = list(results)
    return {
        "runs": len(records),
        "pass_rate": _ratio(sum(record.passed for record in records), len(records)),
        "route_accuracy": _ratio(sum(record.route_correct for record in records), len(records)),
        "answer_accuracy": _ratio(sum(record.answer_correct for record in records), len(records)),
    }


def evaluate(agent: Agent, cases: Iterable[EvalCase], *, repeats: int = 10) -> dict[str, Any]:
    cases = list(cases)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    run_results: list[list[ScoredCase]] = []
    for _ in range(repeats):
        run_results.append([score_case(case, agent.answer(case.query)) for case in cases])
    results = [record for run in run_results for record in run]
    records_by_case = [
        [run_results[run_index][case_index] for run_index in range(repeats)]
        for case_index in range(len(cases))
    ]
    evidence_records = [record for record in results if record.response.route != "abstain" or record.failures]
    expected_evidence_records = [
        record
        for record, case in zip(results, cases * repeats)
        if bool(case.required_evidence)
    ]
    abstention_records = [
        record
        for record, case in zip(results, cases * repeats)
        if case.expected_route == "abstain"
    ]
    failure_counts = Counter(failure for record in results for failure in record.failures)
    by_slice_records: dict[str, list[ScoredCase]] = defaultdict(list)
    for record in results:
        by_slice_records[record.slice_name].append(record)
    run_pass_rates = [
        sum(record.passed for record in run) / len(run) for run in run_results
    ]
    route_successes = sum(record.route_correct for record in results)
    answer_successes = sum(record.answer_correct for record in results)
    pass_successes = sum(record.passed for record in results)
    latencies = [record.response.latency_ms for record in results]
    costs = [record.response.usage.estimated_cost_usd for record in results]
    examples = [
        {
            "case_id": record.case_id,
            "slice": record.slice_name,
            "failures": list(record.failures),
            "route": record.response.route,
            "answer": record.response.answer,
            "evidence_ids": list(record.response.evidence_ids),
            "trace": [asdict(event) for event in record.response.trace],
        }
        for record in results[: len(cases)]
        if record.failures
    ][:10]
    return {
        "agent": agent.name,
        "cases": len(cases),
        "repeats": repeats,
        "runs": len(results),
        "quality": {
            "pass_rate": _ratio(pass_successes, len(results)),
            "route_accuracy": _ratio(route_successes, len(results)),
            "answer_accuracy": _ratio(answer_successes, len(results)),
            "evidence_support_rate": _ratio(
                sum(record.evidence_supported for record in expected_evidence_records),
                len(expected_evidence_records),
            ),
            "abstention_accuracy": _ratio(
                sum(record.route_correct and record.answer_correct for record in abstention_records),
                len(abstention_records),
            ),
            "unsupported_answer_rate": _ratio(
                sum(record.unsupported_answer for record in results), len(results)
            ),
            "pass_rate_95ci": _cluster_bootstrap_interval(
                records_by_case, lambda record: record.passed
            ),
            "route_accuracy_95ci": _cluster_bootstrap_interval(
                records_by_case, lambda record: record.route_correct
            ),
            "answer_accuracy_95ci": _cluster_bootstrap_interval(
                records_by_case, lambda record: record.answer_correct
            ),
            "interval_method": "case_cluster_bootstrap",
            "independent_case_units": len(cases),
            "observations": len(results),
        },
        "reliability": {
            "run_pass_rate_stddev": round(pstdev(run_pass_rates), 6),
            "min_run_pass_rate": round(min(run_pass_rates), 4),
            "max_run_pass_rate": round(max(run_pass_rates), 4),
        },
        "operations": {
            "p50_latency_ms": round(_percentile(latencies, 0.50), 4),
            "p95_latency_ms": round(_percentile(latencies, 0.95), 4),
            "mean_estimated_cost_usd": round(fmean(costs), 8),
            "total_estimated_cost_usd": round(sum(costs), 8),
            "cost_profile_note": "Reference token accounting for regression comparison; no model API is called.",
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_slice": {
            key: _slice_summary(value) for key, value in sorted(by_slice_records.items())
        },
        "failure_examples": examples,
        "diagnostics": {
            "non_abstain_or_failed_runs": len(evidence_records),
            "expected_evidence_runs": len(expected_evidence_records),
        },
    }
