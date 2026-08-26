"""Strict trace replay and counterfactual tool-boundary attribution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from .workflow_agents import WorkflowAgent
from .workflow_env import FAULT_TARGETS
from .workflow_eval import FAULT_SUITE, run_episode
from .workflow_models import FaultMode, WorkflowCase, WorkflowRun


@dataclass(frozen=True)
class ReplayCheck:
    matched: bool
    recorded_digest: str
    replayed_digest: str


@dataclass(frozen=True)
class CounterfactualAttribution:
    case_id: str
    fault: FaultMode
    fault_activated: bool
    strict_replay_matched: bool
    outcome_changed: bool
    attributed_tool: str | None
    expected_tool: str
    localized: bool
    observed_digest: str
    clean_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deterministic_run_payload(run: WorkflowRun) -> dict[str, Any]:
    """Return the complete deterministic episode record used for replay checks."""

    return run.to_dict()


def trace_digest(run: WorkflowRun) -> str:
    encoded = json.dumps(
        deterministic_run_payload(run), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def strict_replay(
    agent: WorkflowAgent,
    case: WorkflowCase,
    recorded: WorkflowRun,
) -> ReplayCheck:
    replayed = run_episode(agent, case, fault=recorded.fault)
    recorded_digest = trace_digest(recorded)
    replayed_digest = trace_digest(replayed)
    return ReplayCheck(
        matched=recorded_digest == replayed_digest,
        recorded_digest=recorded_digest,
        replayed_digest=replayed_digest,
    )


def _tool_results(run: WorkflowRun) -> list[dict[str, Any]]:
    return [event for event in run.trace if event.get("stage") == "tool_result"]


def fault_was_activated(run: WorkflowRun) -> bool:
    return any(event.get("fault") not in {None, "none"} for event in _tool_results(run))


def first_divergent_tool(observed: WorkflowRun, clean: WorkflowRun) -> str | None:
    """Locate the first tool whose recorded result diverges from a clean replay."""

    observed_results = _tool_results(observed)
    clean_results = _tool_results(clean)
    for observed_event, clean_event in zip(observed_results, clean_results):
        if observed_event["tool"] != clean_event["tool"]:
            return str(observed_event["tool"])
        observed_payload = {
            key: value for key, value in observed_event.items() if key != "fault"
        }
        clean_payload = {key: value for key, value in clean_event.items() if key != "fault"}
        if observed_payload != clean_payload:
            return str(observed_event["tool"])
    if len(observed_results) > len(clean_results):
        return str(observed_results[len(clean_results)]["tool"])
    if len(clean_results) > len(observed_results):
        return str(clean_results[len(observed_results)]["tool"])
    return None


def attribute_counterfactual(
    agent: WorkflowAgent,
    case: WorkflowCase,
    fault: FaultMode,
) -> CounterfactualAttribution:
    if fault == "none":
        raise ValueError("Counterfactual attribution requires an injected fault")
    observed = run_episode(agent, case, fault=fault)
    replay = strict_replay(agent, case, observed)
    clean = run_episode(agent, case)
    activated = fault_was_activated(observed)
    attributed_tool = first_divergent_tool(observed, clean) if activated else None
    expected_tool = FAULT_TARGETS[fault]
    return CounterfactualAttribution(
        case_id=case.case_id,
        fault=fault,
        fault_activated=activated,
        strict_replay_matched=replay.matched,
        outcome_changed=(
            observed.outcome != clean.outcome
            or observed.final_status != clean.final_status
            or observed.mutation_count != clean.mutation_count
        ),
        attributed_tool=attributed_tool,
        expected_tool=expected_tool,
        localized=activated and attributed_tool == expected_tool,
        observed_digest=trace_digest(observed),
        clean_digest=trace_digest(clean),
    )


def evaluate_attribution(
    agent: WorkflowAgent,
    cases: list[WorkflowCase],
) -> dict[str, Any]:
    applied_cases = [case for case in cases if case.expected_outcome == "applied"]
    records = [
        attribute_counterfactual(agent, case, fault)
        for fault in FAULT_SUITE
        for case in applied_cases
    ]
    activated = [record for record in records if record.fault_activated]
    return {
        "agent": agent.name,
        "probes": len(records),
        "activated_probes": len(activated),
        "strict_replay_match_rate": round(
            sum(record.strict_replay_matched for record in records) / len(records), 4
        ),
        "localization_accuracy": round(
            sum(record.localized for record in activated) / len(activated), 4
        )
        if activated
        else 0.0,
        "outcome_changed_probes": sum(record.outcome_changed for record in records),
        "by_fault": {
            fault: {
                "probes": len([record for record in records if record.fault == fault]),
                "activated": len(
                    [
                        record
                        for record in records
                        if record.fault == fault and record.fault_activated
                    ]
                ),
                "localized": len(
                    [
                        record
                        for record in records
                        if record.fault == fault and record.localized
                    ]
                ),
            }
            for fault in FAULT_SUITE
        },
        "records": [record.to_dict() for record in records],
    }
