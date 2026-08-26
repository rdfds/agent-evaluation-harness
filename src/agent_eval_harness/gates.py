"""CI-friendly quality gates for candidate agent releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateThresholds:
    pass_rate: float = 0.95
    route_accuracy: float = 0.95
    answer_accuracy: float = 0.95
    evidence_support_rate: float = 0.95
    abstention_accuracy: float = 0.95
    unsupported_answer_rate_max: float = 0.02
    run_pass_rate_stddev_max: float = 0.001


def check_gates(
    report: dict[str, Any], thresholds: GateThresholds | None = None
) -> dict[str, Any]:
    thresholds = thresholds or GateThresholds()
    quality = report["quality"]
    reliability = report["reliability"]
    checks = {
        "pass_rate": quality["pass_rate"] >= thresholds.pass_rate,
        "route_accuracy": quality["route_accuracy"] >= thresholds.route_accuracy,
        "answer_accuracy": quality["answer_accuracy"] >= thresholds.answer_accuracy,
        "evidence_support_rate": quality["evidence_support_rate"]
        >= thresholds.evidence_support_rate,
        "abstention_accuracy": quality["abstention_accuracy"]
        >= thresholds.abstention_accuracy,
        "unsupported_answer_rate": quality["unsupported_answer_rate"]
        <= thresholds.unsupported_answer_rate_max,
        "run_pass_rate_stddev": reliability["run_pass_rate_stddev"]
        <= thresholds.run_pass_rate_stddev_max,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "thresholds": {
            "pass_rate_min": thresholds.pass_rate,
            "route_accuracy_min": thresholds.route_accuracy,
            "answer_accuracy_min": thresholds.answer_accuracy,
            "evidence_support_rate_min": thresholds.evidence_support_rate,
            "abstention_accuracy_min": thresholds.abstention_accuracy,
            "unsupported_answer_rate_max": thresholds.unsupported_answer_rate_max,
            "run_pass_rate_stddev_max": thresholds.run_pass_rate_stddev_max,
        },
    }

