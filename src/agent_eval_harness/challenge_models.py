"""Contracts for the atomic multi-request challenge benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ChallengeOutcome = Literal["applied", "blocked", "no_op", "failed"]
ChallengeFaultPlan = Literal[
    "none",
    "transient_chain",
    "evidence_drift",
    "concurrency_race",
    "ambiguous_commit",
    "degraded_auth",
]


@dataclass(frozen=True)
class ChallengeTask:
    case_id: str
    change_set_id: str
    actor: str


@dataclass(frozen=True)
class ChallengeCase:
    case_id: str
    change_set_id: str
    actor: str
    expected_outcome: ChallengeOutcome
    expected_change_set_status: str
    expected_initial_states: dict[str, str]
    expected_final_states: dict[str, str]
    expected_mutation_count: int
    slice_name: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChallengeCase":
        outcome = payload["expected_outcome"]
        if outcome not in {"applied", "blocked", "no_op", "failed"}:
            raise ValueError(f"Unsupported challenge outcome {outcome!r}")
        return cls(
            case_id=str(payload["case_id"]),
            change_set_id=str(payload["change_set_id"]),
            actor=str(payload["actor"]),
            expected_outcome=outcome,
            expected_change_set_status=str(payload["expected_change_set_status"]),
            expected_initial_states={
                str(key): str(value)
                for key, value in payload["expected_initial_states"].items()
            },
            expected_final_states={
                str(key): str(value)
                for key, value in payload["expected_final_states"].items()
            },
            expected_mutation_count=int(payload["expected_mutation_count"]),
            slice_name=str(payload["slice_name"]),
        )

    def task(self) -> ChallengeTask:
        return ChallengeTask(self.case_id, self.change_set_id, self.actor)


@dataclass(frozen=True)
class ChallengeRun:
    agent: str
    case_id: str
    fault_plan: ChallengeFaultPlan
    activated_faults: tuple[str, ...]
    outcome: ChallengeOutcome
    change_set_status: str
    final_states: dict[str, str]
    mutation_count: int
    tool_calls: int
    trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
