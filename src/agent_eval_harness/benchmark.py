"""Build the comparison artifact used by documentation and CI."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset import DEFAULT_CASES_PATH, DEFAULT_KNOWLEDGE_PATH, load_cases, load_knowledge_base
from .faults import FaultInjectingAgent
from .gates import check_gates
from .metrics import evaluate
from .runtime import CandidateAgent, KnowledgeBase, LegacyAgent


FAULT_MODES = ["route_swap", "stale_answer", "drop_evidence", "disable_abstain"]


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _quality_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = [
        "pass_rate",
        "route_accuracy",
        "answer_accuracy",
        "evidence_support_rate",
        "abstention_accuracy",
        "unsupported_answer_rate",
    ]
    return {
        key: round(candidate["quality"][key] - baseline["quality"][key], 4)
        for key in keys
    }


def build_benchmark(*, repeats: int = 10) -> dict[str, Any]:
    cases = load_cases()
    knowledge = KnowledgeBase(load_knowledge_base())
    baseline_agent = LegacyAgent(knowledge)
    candidate_agent = CandidateAgent(knowledge)
    baseline = evaluate(baseline_agent, cases, repeats=repeats)
    candidate = evaluate(candidate_agent, cases, repeats=repeats)
    candidate_gates = check_gates(candidate)
    mutations: list[dict[str, Any]] = []
    for mode in FAULT_MODES:
        report = evaluate(FaultInjectingAgent(candidate_agent, mode), cases, repeats=1)
        gates = check_gates(report)
        mutations.append(
            {
                "fault": mode,
                "caught": not gates["passed"],
                "failed_gates": gates["failed"],
                "pass_rate": report["quality"]["pass_rate"],
            }
        )
    return {
        "schema_version": 2,
        "dataset": {
            "cases": len(cases),
            "independent_case_units": len(cases),
            "repeats": repeats,
            "candidate_observations": len(cases) * repeats,
            "slices": sorted({case.slice_name for case in cases}),
            "difficulty_levels": sorted({case.difficulty for case in cases}),
            "sha256": _sha256([DEFAULT_CASES_PATH, DEFAULT_KNOWLEDGE_PATH]),
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _quality_delta(candidate, baseline),
        "candidate_gates": candidate_gates,
        "mutation_testing": {
            "caught": sum(mutation["caught"] for mutation in mutations),
            "total": len(mutations),
            "mutations": mutations,
        },
    }


def deterministic_projection(artifact: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-timing fields before comparing a fresh run with CI goldens."""

    projected = deepcopy(artifact)
    for label in ["baseline", "candidate"]:
        operations = projected[label]["operations"]
        operations.pop("p50_latency_ms", None)
        operations.pop("p95_latency_ms", None)
    return projected


def render_markdown(artifact: dict[str, Any]) -> str:
    baseline = artifact["baseline"]
    candidate = artifact["candidate"]
    mutations = artifact["mutation_testing"]
    lines = [
        "# Public benchmark results",
        "",
        (
            f"The committed fixture contains **{artifact['dataset']['cases']} cases** and the default "
            f"run records **{artifact['dataset']['candidate_observations']} candidate observations** across "
            f"{artifact['dataset']['repeats']} repeats. Confidence intervals resample the "
            f"{artifact['dataset']['independent_case_units']} case clusters rather than treating repeated "
            "observations as independent."
        ),
        "",
        "| Metric | Legacy router | Evidence-aware candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    labels = [
        ("Pass rate", "pass_rate"),
        ("Route accuracy", "route_accuracy"),
        ("Answer accuracy", "answer_accuracy"),
        ("Evidence support", "evidence_support_rate"),
        ("Abstention accuracy", "abstention_accuracy"),
        ("Unsupported-answer rate", "unsupported_answer_rate"),
    ]
    for label, key in labels:
        delta = artifact["comparison"][key]
        lines.append(
            f"| {label} | {baseline['quality'][key]:.2%} | {candidate['quality'][key]:.2%} | {delta:+.2%} |"
        )
    lines.extend(
        [
            "",
            "## Reliability and operations",
            "",
            f"- Candidate pass-rate 95% case-cluster bootstrap interval: `{candidate['quality']['pass_rate_95ci']}`",
            f"- Repeated-run pass-rate standard deviation: `{candidate['reliability']['run_pass_rate_stddev']:.6f}`",
            f"- Observed local p95 harness latency: `{candidate['operations']['p95_latency_ms']:.4f} ms`",
            f"- Reference estimated cost per case: `${candidate['operations']['mean_estimated_cost_usd']:.8f}`",
            f"- Candidate release gates: `{'PASS' if artifact['candidate_gates']['passed'] else 'FAIL'}`",
            "",
            "## Mutation testing",
            "",
            (
                f"The gate suite detected **{mutations['caught']}/{mutations['total']}** seeded regressions "
                "covering route swaps, stale answers, missing evidence, and disabled abstention."
            ),
            "",
            "| Seeded regression | Detected | Failed gates |",
            "|---|---:|---|",
        ]
    )
    for mutation in mutations["mutations"]:
        lines.append(
            f"| `{mutation['fault']}` | {'yes' if mutation['caught'] else 'no'} | "
            f"{', '.join(mutation['failed_gates']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results validate the harness and the committed evidence-aware routing policy on the public fixture. "
            "They are not a claim about a hosted foundation model or an undisclosed production dataset. Runtime "
            "latency varies by machine and is excluded from golden-artifact equality checks.",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(artifact: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(artifact), encoding="utf-8")
