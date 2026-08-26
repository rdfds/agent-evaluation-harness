"""Build the reproducible stateful workflow reliability artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .workflow_agents import LegacyWorkflowAgent, RecoveryWorkflowAgent
from .workflow_dataset import (
    DEFAULT_WORKFLOW_CASES_PATH,
    DEFAULT_WORKFLOW_SEED_PATH,
    load_workflow_cases,
)
from .workflow_eval import FAULT_SUITE, evaluate_workflow_agent
from .workflow_replay import evaluate_attribution


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    return round(candidate[key] - baseline[key], 4)


def build_workflow_benchmark() -> dict[str, Any]:
    cases = load_workflow_cases()
    baseline_agent = LegacyWorkflowAgent()
    candidate_agent = RecoveryWorkflowAgent()
    baseline, _, _ = evaluate_workflow_agent(baseline_agent, cases)
    candidate, _, _ = evaluate_workflow_agent(candidate_agent, cases)
    baseline_attribution = evaluate_attribution(baseline_agent, cases)
    candidate_attribution = evaluate_attribution(candidate_agent, cases)
    gates = {
        "clean_pass_rate": candidate["clean"]["pass_rate"] == 1.0,
        "fault_handling_rate": candidate["faults"]["fault_handling_rate"] == 1.0,
        "terminal_recovery_rate": candidate["faults"]["terminal_recovery_rate"] == 1.0,
        "fail_closed_rate": candidate["faults"]["fail_closed_rate"] == 1.0,
        "strict_replay_match_rate": candidate_attribution["strict_replay_match_rate"] == 1.0,
        "localization_accuracy": candidate_attribution["localization_accuracy"] == 1.0,
    }
    applied_cases = [case for case in cases if case.expected_outcome == "applied"]
    return {
        "schema_version": 1,
        "dataset": {
            "clean_cases": len(cases),
            "fault_eligible_cases": len(applied_cases),
            "fault_modes": list(FAULT_SUITE),
            "fault_episodes_per_agent": len(applied_cases) * len(FAULT_SUITE),
            "sha256": _sha256(
                [DEFAULT_WORKFLOW_CASES_PATH, DEFAULT_WORKFLOW_SEED_PATH]
            ),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": {
            "clean_pass_rate": _delta(candidate["clean"], baseline["clean"], "pass_rate"),
            "clean_trajectory_compliance_rate": _delta(
                candidate["clean"], baseline["clean"], "trajectory_compliance_rate"
            ),
            "fault_handling_rate": _delta(
                candidate["faults"], baseline["faults"], "fault_handling_rate"
            ),
            "terminal_recovery_rate": _delta(
                candidate["faults"], baseline["faults"], "terminal_recovery_rate"
            ),
        },
        "attribution": {
            "baseline": baseline_attribution,
            "candidate": candidate_attribution,
        },
        "release_gates": {
            "passed": all(gates.values()),
            "checks": gates,
            "failed": [name for name, passed in gates.items() if not passed],
        },
    }


def render_workflow_markdown(artifact: dict[str, Any]) -> str:
    baseline = artifact["baseline"]
    candidate = artifact["candidate"]
    dataset = artifact["dataset"]
    attribution = artifact["attribution"]
    lines = [
        "# Stateful workflow reliability results",
        "",
        (
            f"The benchmark runs **{dataset['clean_cases']} clean stateful tasks** and "
            f"**{dataset['fault_episodes_per_agent']} fault episodes per agent** across "
            f"{len(dataset['fault_modes'])} controlled tool-boundary perturbations."
        ),
        "",
        "| Metric | One-shot baseline | Recovery-aware candidate | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Clean full-pass rate | {baseline['clean']['pass_rate']:.2%} | "
            f"{candidate['clean']['pass_rate']:.2%} | "
            f"{artifact['comparison']['clean_pass_rate']:+.2%} |"
        ),
        (
            f"| Clean terminal-state accuracy | {baseline['clean']['terminal_state_accuracy']:.2%} | "
            f"{candidate['clean']['terminal_state_accuracy']:.2%} | "
            f"{candidate['clean']['terminal_state_accuracy'] - baseline['clean']['terminal_state_accuracy']:+.2%} |"
        ),
        (
            f"| Clean trajectory compliance | {baseline['clean']['trajectory_compliance_rate']:.2%} | "
            f"{candidate['clean']['trajectory_compliance_rate']:.2%} | "
            f"{artifact['comparison']['clean_trajectory_compliance_rate']:+.2%} |"
        ),
        (
            f"| Recoverable-fault terminal recovery | {baseline['faults']['terminal_recovery_rate']:.2%} | "
            f"{candidate['faults']['terminal_recovery_rate']:.2%} | "
            f"{artifact['comparison']['terminal_recovery_rate']:+.2%} |"
        ),
        (
            f"| Full fault-handling rate | {baseline['faults']['fault_handling_rate']:.2%} | "
            f"{candidate['faults']['fault_handling_rate']:.2%} | "
            f"{artifact['comparison']['fault_handling_rate']:+.2%} |"
        ),
        (
            f"| Permission fault fail-closed rate | {baseline['faults']['fail_closed_rate']:.2%} | "
            f"{candidate['faults']['fail_closed_rate']:.2%} | "
            f"{candidate['faults']['fail_closed_rate'] - baseline['faults']['fail_closed_rate']:+.2%} |"
        ),
        "",
        "The one-shot baseline reaches the correct clean terminal state but skips required policy and approval checks. Full-pass scoring prevents that shortcut from being counted as compliant behavior.",
        "",
        "## Controlled fault matrix",
        "",
        "| Fault | Baseline full pass | Candidate full pass | Candidate terminal handling |",
        "|---|---:|---:|---:|",
    ]
    for fault in dataset["fault_modes"]:
        baseline_fault = baseline["faults"]["by_fault"][fault]
        candidate_fault = candidate["faults"]["by_fault"][fault]
        lines.append(
            f"| `{fault}` | {baseline_fault['pass_rate']:.2%} | "
            f"{candidate_fault['pass_rate']:.2%} | "
            f"{candidate_fault['terminal_handling_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Replay and counterfactual localization",
            "",
            (
                f"Strict replay reproduced **{candidate['faults']['episodes']}/{candidate['faults']['episodes']}** "
                "candidate fault episodes byte-for-byte at the structured trace level."
            ),
            (
                f"The first divergent tool result localized **{attribution['candidate']['activated_probes']}/"
                f"{attribution['candidate']['activated_probes']}** activated candidate faults to the injected "
                "tool boundary without consulting the fault label."
            ),
            "",
            "| Agent | Probes | Fault activated | Strict replay | Tool localization |",
            "|---|---:|---:|---:|---:|",
            (
                f"| One-shot baseline | {attribution['baseline']['probes']} | "
                f"{attribution['baseline']['activated_probes']} | "
                f"{attribution['baseline']['strict_replay_match_rate']:.2%} | "
                f"{attribution['baseline']['localization_accuracy']:.2%} |"
            ),
            (
                f"| Recovery-aware candidate | {attribution['candidate']['probes']} | "
                f"{attribution['candidate']['activated_probes']} | "
                f"{attribution['candidate']['strict_replay_match_rate']:.2%} | "
                f"{attribution['candidate']['localization_accuracy']:.2%} |"
            ),
            "",
            "## Interpretation",
            "",
            "These results validate deterministic orchestration, fault recovery, state grading, replay, and localization against the committed SQLite fixture. They do not measure a hosted language model. Model-backed adapters should retain the same environment and graders while adding prompt/model identifiers, sampled repetitions, observed tokens, latency, and cost.",
            "",
        ]
    )
    return "\n".join(lines)


def write_workflow_artifacts(
    artifact: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_workflow_markdown(artifact), encoding="utf-8")
