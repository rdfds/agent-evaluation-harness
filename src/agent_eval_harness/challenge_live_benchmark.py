"""Repeated live benchmark for atomic workflows and composed faults."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean
import time
from typing import Any, Callable, Iterable, Sequence

from .challenge_dataset import (
    DEFAULT_CHALLENGE_CASES_PATH,
    DEFAULT_CHALLENGE_SEED_PATH,
    load_challenge_cases,
)
from .challenge_env import ChallengeEnvironment, FAULT_PLANS
from .challenge_eval import challenge_score_to_dict, score_challenge_run
from .challenge_model_agent import (
    CHALLENGE_PROMPTS,
    ChallengeModelAgent,
    ChallengePromptVariant,
)
from .challenge_models import ChallengeCase, ChallengeFaultPlan, ChallengeRun
from .live_benchmark import MODEL_PRICING_PER_MILLION
from .live_client import ResponsesClient


DEFAULT_CHALLENGE_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol")
DEFAULT_CHALLENGE_PROMPTS: tuple[ChallengePromptVariant, ...] = (
    "minimal",
    "principled",
)
DEFAULT_FAULT_CHANGE_SET_IDS = ("atomic-ready-3", "long-horizon-5")
DEFAULT_REPETITIONS = 3
CHALLENGE_FAULT_SUITE: tuple[ChallengeFaultPlan, ...] = (
    "transient_chain",
    "evidence_drift",
    "concurrency_race",
    "ambiguous_commit",
    "degraded_auth",
)
SECRET_PATTERN = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}")


@dataclass(frozen=True)
class ChallengeEpisodeSpec:
    model: str
    prompt_variant: ChallengePromptVariant
    case_id: str
    fault_plan: ChallengeFaultPlan
    trial: int

    @property
    def episode_id(self) -> str:
        return "::".join(
            (
                self.model,
                self.prompt_variant,
                self.case_id,
                self.fault_plan,
                f"trial-{self.trial}",
            )
        )


def _dataset_sha256() -> str:
    digest = hashlib.sha256()
    for path in (DEFAULT_CHALLENGE_CASES_PATH, DEFAULT_CHALLENGE_SEED_PATH):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_challenge_specs(
    *,
    models: Sequence[str] = DEFAULT_CHALLENGE_MODELS,
    prompts: Sequence[ChallengePromptVariant] = DEFAULT_CHALLENGE_PROMPTS,
    repetitions: int = DEFAULT_REPETITIONS,
    cases: Sequence[ChallengeCase] | None = None,
    fault_change_set_ids: Sequence[str] = DEFAULT_FAULT_CHANGE_SET_IDS,
) -> list[ChallengeEpisodeSpec]:
    if repetitions < 2:
        raise ValueError("Challenge benchmark requires at least two repetitions")
    cases = list(cases or load_challenge_cases())
    known = {case.change_set_id for case in cases}
    missing = sorted(set(fault_change_set_ids) - known)
    if missing:
        raise ValueError(f"Unknown fault change sets: {', '.join(missing)}")
    if any(prompt not in CHALLENGE_PROMPTS for prompt in prompts):
        raise ValueError("Unsupported challenge prompt variant")
    specs: list[ChallengeEpisodeSpec] = []
    for model in models:
        for prompt in prompts:
            for trial in range(1, repetitions + 1):
                specs.extend(
                    ChallengeEpisodeSpec(model, prompt, case.case_id, "none", trial)
                    for case in cases
                )
                for case in cases:
                    if case.change_set_id not in fault_change_set_ids:
                        continue
                    specs.extend(
                        ChallengeEpisodeSpec(model, prompt, case.case_id, fault, trial)
                        for fault in CHALLENGE_FAULT_SUITE
                    )
    return specs


def run_challenge_live_episode(
    spec: ChallengeEpisodeSpec,
    *,
    client: ResponsesClient,
    cases_by_id: dict[str, ChallengeCase],
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    case = cases_by_id[spec.case_id]
    agent = ChallengeModelAgent(
        client,
        model=spec.model,
        prompt_variant=spec.prompt_variant,
        reasoning_effort=reasoning_effort,
    )
    started = time.perf_counter()
    with ChallengeEnvironment(fault_plan=spec.fault_plan) as environment:
        run = agent.run(case.task(), environment)
    elapsed_ms = round((time.perf_counter() - started) * 1_000, 2)
    score = score_challenge_run(run, case)
    return {
        "episode_id": spec.episode_id,
        "requested_model": spec.model,
        "prompt_variant": spec.prompt_variant,
        "reasoning_effort": reasoning_effort,
        "trial": spec.trial,
        "case_id": case.case_id,
        "change_set_id": case.change_set_id,
        "slice": case.slice_name,
        "fault_plan": spec.fault_plan,
        "score": challenge_score_to_dict(score),
        "run": run.to_dict(),
        "provider": agent.last_metadata,
        "episode_latency_ms": elapsed_ms,
    }


def load_challenge_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[str(record["episode_id"])] = record
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid challenge checkpoint on line {line_number}") from exc
    return records


def append_challenge_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def execute_challenge_specs(
    specs: Sequence[ChallengeEpisodeSpec],
    *,
    client: ResponsesClient,
    checkpoint_path: Path,
    max_workers: int = 6,
    reasoning_effort: str = "low",
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    cases_by_id = {case.case_id: case for case in load_challenge_cases()}
    requested_ids = {spec.episode_id for spec in specs}
    checkpoint = load_challenge_checkpoint(checkpoint_path)
    records = {
        episode_id: record
        for episode_id, record in checkpoint.items()
        if episode_id in requested_ids
    }
    missing = [spec for spec in specs if spec.episode_id not in records]
    completed = len(records)
    total = len(specs)
    if progress and records:
        progress(completed, total, {"resumed": True})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_challenge_live_episode,
                spec,
                client=client,
                cases_by_id=cases_by_id,
                reasoning_effort=reasoning_effort,
            ): spec
            for spec in missing
        }
        try:
            for future in as_completed(futures):
                record = future.result()
                append_challenge_checkpoint(checkpoint_path, record)
                records[record["episode_id"]] = record
                completed += 1
                if progress:
                    progress(completed, total, record)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return [records[spec.episode_id] for spec in specs]


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _wilson(successes: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    observed = successes / total
    denominator = 1 + z * z / total
    center = (observed + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * percentile)
    return round(float(ordered[index]), 2)


def _cell_reliability(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        cells[(record["case_id"], record["fault_plan"])].append(
            bool(record["score"]["passed"])
        )
    return {
        "cells": len(cells),
        "all_trials_pass_rate": _rate(all(values) for values in cells.values()),
        "any_trial_pass_rate": _rate(any(values) for values in cells.values()),
        "mixed_result_cell_rate": _rate(
            any(values) and not all(values) for values in cells.values()
        ),
        "failed_all_trials": [
            {"case_id": case_id, "fault_plan": fault_plan}
            for (case_id, fault_plan), values in sorted(cells.items())
            if not any(values)
        ],
        "unstable_cells": [
            {
                "case_id": case_id,
                "fault_plan": fault_plan,
                "passes": sum(values),
                "trials": len(values),
            }
            for (case_id, fault_plan), values in sorted(cells.items())
            if any(values) and not all(values)
        ],
    }


def _score_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool(record["score"]["passed"]) for record in records)
    return {
        "episodes": len(records),
        "passed": passed,
        "pass_rate": _rate(record["score"]["passed"] for record in records),
        "pass_rate_wilson_95": _wilson(passed, len(records)),
        "outcome_accuracy": _rate(
            record["score"]["outcome_correct"] for record in records
        ),
        "change_set_state_accuracy": _rate(
            record["score"]["change_set_state_correct"] for record in records
        ),
        "request_state_accuracy": _rate(
            record["score"]["request_states_correct"] for record in records
        ),
        "trajectory_compliance_rate": _rate(
            record["score"]["trajectory_compliant"] for record in records
        ),
        "atomicity_rate": _rate(
            record["score"]["atomicity_preserved"] for record in records
        ),
        "fault_coverage_rate": _rate(
            record["score"]["fault_coverage_complete"] for record in records
        ),
        "reliability": _cell_reliability(records),
    }


def _estimated_cost(model: str, records: Sequence[dict[str, Any]]) -> float | None:
    prices = MODEL_PRICING_PER_MILLION.get(model)
    if prices is None:
        return None
    usage = [record["provider"]["usage"] for record in records]
    input_tokens = sum(item["input_tokens"] for item in usage)
    cached_tokens = sum(item["cached_input_tokens"] for item in usage)
    output_tokens = sum(item["output_tokens"] for item in usage)
    cost = (
        (input_tokens - cached_tokens) * prices["input"]
        + cached_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000
    return round(cost, 6)


def summarize_challenge_config(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clean = [record for record in records if record["fault_plan"] == "none"]
    faults = [record for record in records if record["fault_plan"] != "none"]
    usage = [record["provider"]["usage"] for record in records]
    by_fault = {
        fault: _score_summary(
            [record for record in faults if record["fault_plan"] == fault]
        )
        for fault in CHALLENGE_FAULT_SUITE
    }
    failures = [
        {
            "case_id": record["case_id"],
            "fault_plan": record["fault_plan"],
            "trial": record["trial"],
            "failures": record["score"]["failures"],
        }
        for record in records
        if not record["score"]["passed"]
    ]
    failure_counts = Counter(
        failure
        for record in records
        for failure in record["score"]["failures"]
    )
    model = str(records[0]["requested_model"])
    return {
        "requested_model": model,
        "resolved_models": sorted(
            {
                resolved
                for record in records
                for resolved in record["provider"]["resolved_models"]
            }
        ),
        "prompt_variant": records[0]["prompt_variant"],
        "episodes": len(records),
        "overall": _score_summary(records),
        "clean": _score_summary(clean),
        "faults": {**_score_summary(faults), "by_fault": by_fault},
        "usage": {
            "response_calls": sum(record["provider"]["response_calls"] for record in records),
            "input_tokens": sum(item["input_tokens"] for item in usage),
            "cached_input_tokens": sum(item["cached_input_tokens"] for item in usage),
            "output_tokens": sum(item["output_tokens"] for item in usage),
            "reasoning_tokens": sum(item["reasoning_tokens"] for item in usage),
        },
        "latency_ms": {
            "episode_mean": round(fmean(record["episode_latency_ms"] for record in records), 2),
            "episode_p50": _percentile(
                (record["episode_latency_ms"] for record in records), 0.50
            ),
            "episode_p95": _percentile(
                (record["episode_latency_ms"] for record in records), 0.95
            ),
            "provider_total": round(
                sum(record["provider"]["provider_latency_ms"] for record in records), 2
            ),
        },
        "estimated_cost_usd": _estimated_cost(model, records),
        "failure_counts": dict(sorted(failure_counts.items())),
        "failures": failures,
    }


def build_challenge_artifact(
    records: Sequence[dict[str, Any]],
    *,
    models: Sequence[str],
    prompts: Sequence[ChallengePromptVariant],
    repetitions: int,
    fault_change_set_ids: Sequence[str] = DEFAULT_FAULT_CHANGE_SET_IDS,
) -> dict[str, Any]:
    configs = []
    for model in models:
        for prompt in prompts:
            subset = [
                record
                for record in records
                if record["requested_model"] == model
                and record["prompt_variant"] == prompt
            ]
            if not subset:
                raise ValueError(f"No challenge records for {model}/{prompt}")
            configs.append(summarize_challenge_config(subset))

    deltas = []
    if "minimal" in prompts and "principled" in prompts:
        indexed = {
            (config["requested_model"], config["prompt_variant"]): config
            for config in configs
        }
        for model in models:
            minimal = indexed[(model, "minimal")]
            principled = indexed[(model, "principled")]
            deltas.append(
                {
                    "model": model,
                    "clean_pass_rate": round(
                        principled["clean"]["pass_rate"]
                        - minimal["clean"]["pass_rate"],
                        4,
                    ),
                    "fault_pass_rate": round(
                        principled["faults"]["pass_rate"]
                        - minimal["faults"]["pass_rate"],
                        4,
                    ),
                    "overall_all_trials_pass_rate": round(
                        principled["overall"]["reliability"]["all_trials_pass_rate"]
                        - minimal["overall"]["reliability"]["all_trials_pass_rate"],
                        4,
                    ),
                    "episode_p95_ms": round(
                        principled["latency_ms"]["episode_p95"]
                        - minimal["latency_ms"]["episode_p95"],
                        2,
                    ),
                    "estimated_cost_usd": round(
                        (principled["estimated_cost_usd"] or 0)
                        - (minimal["estimated_cost_usd"] or 0),
                        6,
                    ),
                }
            )

    clean_cases = len(
        {record["case_id"] for record in records if record["fault_plan"] == "none"}
    )
    fault_cases = len(
        {record["case_id"] for record in records if record["fault_plan"] != "none"}
    )
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "models": list(models),
            "prompts": {
                prompt: {
                    "sha256": hashlib.sha256(
                        CHALLENGE_PROMPTS[prompt].encode("utf-8")
                    ).hexdigest()
                }
                for prompt in prompts
            },
            "reasoning_effort": records[0]["reasoning_effort"],
            "repetitions_per_cell": repetitions,
            "clean_cases_per_config": clean_cases,
            "fault_cases_per_config": fault_cases,
            "fault_plans": {
                fault: list(FAULT_PLANS[fault]) for fault in CHALLENGE_FAULT_SUITE
            },
            "fault_change_set_ids": list(fault_change_set_ids),
            "episodes_per_config": repetitions
            * (clean_cases + fault_cases * len(CHALLENGE_FAULT_SUITE)),
            "total_episodes": len(records),
            "dataset_sha256": _dataset_sha256(),
            "pricing_usd_per_million_tokens": {
                model: MODEL_PRICING_PER_MILLION[model]
                for model in models
                if model in MODEL_PRICING_PER_MILLION
            },
        },
        "configs": configs,
        "prompt_deltas_principled_minus_minimal": deltas,
        "totals": {
            "episodes": len(records),
            "passed": sum(bool(record["score"]["passed"]) for record in records),
            "response_calls": sum(config["usage"]["response_calls"] for config in configs),
            "input_tokens": sum(config["usage"]["input_tokens"] for config in configs),
            "output_tokens": sum(config["usage"]["output_tokens"] for config in configs),
            "estimated_cost_usd": round(
                sum(config["estimated_cost_usd"] or 0 for config in configs), 6
            ),
        },
        "episodes": sorted(records, key=lambda record: record["episode_id"]),
    }


def verify_challenge_artifact(artifact: dict[str, Any]) -> dict[str, int]:
    protocol = artifact.get("protocol") or {}
    models = list(protocol.get("models") or [])
    prompts = list((protocol.get("prompts") or {}).keys())
    repetitions = int(protocol.get("repetitions_per_cell") or 0)
    records = list(artifact.get("episodes") or [])
    fault_ids = protocol.get("fault_change_set_ids") or DEFAULT_FAULT_CHANGE_SET_IDS
    if artifact.get("schema_version") != 2 or not models or not prompts or not records:
        raise ValueError("Incomplete challenge artifact")
    if SECRET_PATTERN.search(json.dumps(artifact)):
        raise ValueError("Potential API credential found in challenge artifact")
    specs = build_challenge_specs(
        models=models,
        prompts=prompts,
        repetitions=repetitions,
        fault_change_set_ids=fault_ids,
    )
    expected_ids = {spec.episode_id for spec in specs}
    recorded_ids = [str(record.get("episode_id")) for record in records]
    if len(recorded_ids) != len(set(recorded_ids)):
        raise ValueError("Duplicate challenge episode IDs")
    if set(recorded_ids) != expected_ids:
        raise ValueError("Challenge episode matrix does not match its protocol")

    cases_by_id = {case.case_id: case for case in load_challenge_cases()}
    for record in records:
        case = cases_by_id.get(str(record.get("case_id")))
        if case is None:
            raise ValueError(f"Unknown challenge case: {record.get('case_id')}")
        run_payload = dict(record.get("run") or {})
        run_payload["trace"] = tuple(run_payload.get("trace") or [])
        run_payload["activated_faults"] = tuple(
            run_payload.get("activated_faults") or []
        )
        try:
            run = ChallengeRun(**run_payload)
        except TypeError as exc:
            raise ValueError(f"Invalid challenge run for {record['episode_id']}") from exc
        score_payload = challenge_score_to_dict(score_challenge_run(run, case))
        if score_payload != record.get("score"):
            raise ValueError(f"Challenge score drift for {record['episode_id']}")
        if record.get("fault_plan") != run.fault_plan:
            raise ValueError(f"Challenge fault identity drift for {record['episode_id']}")
        if run.tool_calls != sum(
            event.get("stage") == "tool_call" for event in run.trace
        ):
            raise ValueError(f"Challenge tool-call drift for {record['episode_id']}")
        provider = record.get("provider") or {}
        if provider.get("requested_model") != record.get("requested_model"):
            raise ValueError(f"Challenge model identity drift for {record['episode_id']}")
        if provider.get("prompt_variant") != record.get("prompt_variant"):
            raise ValueError(f"Challenge prompt identity drift for {record['episode_id']}")
        if provider.get("response_calls") != sum(
            event.get("stage") == "model_call" for event in run.trace
        ):
            raise ValueError(f"Challenge provider-call drift for {record['episode_id']}")

    rebuilt = build_challenge_artifact(
        records,
        models=models,
        prompts=prompts,
        repetitions=repetitions,
        fault_change_set_ids=fault_ids,
    )
    for key in (
        "protocol",
        "configs",
        "prompt_deltas_principled_minus_minimal",
        "totals",
    ):
        if rebuilt[key] != artifact.get(key):
            raise ValueError(f"Challenge artifact {key} summary drift")
    passed = sum(bool(record["score"]["passed"]) for record in records)
    return {"episodes": len(records), "passed": passed, "failed": len(records) - passed}


def _rate_ci(summary: dict[str, Any]) -> str:
    low, high = summary["pass_rate_wilson_95"]
    return f"{summary['pass_rate']:.1%} [{low:.1%}, {high:.1%}]"


def render_challenge_markdown(artifact: dict[str, Any]) -> str:
    protocol = artifact["protocol"]
    lines = [
        "# Repeated atomic-workflow challenge",
        "",
        (
            f"This recorded run contains **{protocol['total_episodes']} hosted-model episodes** "
            f"across {len(protocol['models'])} models, {len(protocol['prompts'])} prompt contracts, "
            f"and {protocol['repetitions_per_cell']} trials per task/fault cell. Each configuration "
            f"runs {protocol['clean_cases_per_config']} clean multi-request change sets and "
            f"{protocol['fault_cases_per_config'] * len(protocol['fault_plans'])} composed-fault cells."
        ),
        "",
        "| Model | Prompt | Clean pass (95% Wilson) | Clean all-trials | Fault pass (95% Wilson) | Fault all-trials | Trajectory | Atomicity | p95 episode | Est. cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in artifact["configs"]:
        lines.append(
            f"| `{config['requested_model']}` | {config['prompt_variant']} | "
            f"{_rate_ci(config['clean'])} | "
            f"{config['clean']['reliability']['all_trials_pass_rate']:.1%} | "
            f"{_rate_ci(config['faults'])} | "
            f"{config['faults']['reliability']['all_trials_pass_rate']:.1%} | "
            f"{config['overall']['trajectory_compliance_rate']:.1%} | "
            f"{config['overall']['atomicity_rate']:.1%} | "
            f"{config['latency_ms']['episode_p95'] / 1_000:.2f}s | "
            f"${config['estimated_cost_usd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Principle prompt ablation",
            "",
            "Deltas are principled minus minimal. Accuracy and all-trials deltas favor the general recovery contract when positive; latency and cost deltas measure overhead.",
            "",
            "| Model | Clean pass | Fault pass | Overall all-trials | p95 episode | Est. cost |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for delta in artifact["prompt_deltas_principled_minus_minimal"]:
        lines.append(
            f"| `{delta['model']}` | {delta['clean_pass_rate']:+.1%} | "
            f"{delta['fault_pass_rate']:+.1%} | "
            f"{delta['overall_all_trials_pass_rate']:+.1%} | "
            f"{delta['episode_p95_ms'] / 1_000:+.2f}s | "
            f"${delta['estimated_cost_usd']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Composed-fault reliability",
            "",
            "Each percentage is the episode pass rate across two long change sets and three trials per change set.",
            "",
            "| Model | Prompt | Transient chain | Evidence drift | Concurrency race | Ambiguous commit | Degraded auth |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for config in artifact["configs"]:
        by_fault = config["faults"]["by_fault"]
        lines.append(
            f"| `{config['requested_model']}` | {config['prompt_variant']} | "
            f"{by_fault['transient_chain']['pass_rate']:.1%} | "
            f"{by_fault['evidence_drift']['pass_rate']:.1%} | "
            f"{by_fault['concurrency_race']['pass_rate']:.1%} | "
            f"{by_fault['ambiguous_commit']['pass_rate']:.1%} | "
            f"{by_fault['degraded_auth']['pass_rate']:.1%} |"
        )
    lines.extend(["", "## Observed failure modes", ""])
    for config in artifact["configs"]:
        failures = config["failure_counts"]
        summary = (
            ", ".join(f"`{name}`: {count}" for name, count in failures.items())
            if failures
            else "none"
        )
        unstable = config["overall"]["reliability"]["mixed_result_cell_rate"]
        lines.append(
            f"- `{config['requested_model']}` / {config['prompt_variant']}: {summary}; "
            f"mixed-result cells {unstable:.1%}."
        )
    totals = artifact["totals"]
    lines.extend(
        [
            "",
            "## Run accounting",
            "",
            f"The run made **{totals['response_calls']:,} Responses API calls**, consumed "
            f"**{totals['input_tokens']:,} input tokens** and **{totals['output_tokens']:,} output tokens**, "
            f"and has an estimated cost of **${totals['estimated_cost_usd']:.4f}** at the recorded prices.",
            "",
            "The JSON artifact retains every trial, structured tool call and result, composed-fault activation, terminal request map, hidden-grader decision, token count, and observed latency. It excludes credentials and private reasoning content.",
            "",
            "## Interpretation limits",
            "",
            "Three trials reveal gross instability but do not provide a high-powered stochastic estimate. Wilson intervals treat episodes as Bernoulli observations even though trials share tasks and prompts, so cell-level all-trials consistency is the stricter reliability view. The benchmark covers one atomic change-control domain, and concurrent wall-clock execution makes latency descriptive rather than a controlled speed ranking.",
            "",
        ]
    )
    return "\n".join(lines)


def write_challenge_artifacts(
    artifact: dict[str, Any], json_path: Path, markdown_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_challenge_markdown(artifact), encoding="utf-8")
