"""Contracts for stateful tool-agent workflow evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


WorkflowOutcome = Literal["applied", "blocked", "no_op", "failed"]
FaultMode = Literal[
    "none",
    "timeout_once",
    "schema_drift",
    "stale_read",
    "partial_response",
    "permission_denied",
]


@dataclass(frozen=True)
class WorkflowCase:
    case_id: str
    request_id: str
    actor: str
    target_status: str
    expected_outcome: WorkflowOutcome
    expected_final_status: str
    slice_name: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowCase":
        outcome = payload["expected_outcome"]
        if outcome not in {"applied", "blocked", "no_op", "failed"}:
            raise ValueError(f"Unsupported workflow outcome {outcome!r}")
        return cls(
            case_id=str(payload["case_id"]),
            request_id=str(payload["request_id"]),
            actor=str(payload["actor"]),
            target_status=str(payload["target_status"]),
            expected_outcome=outcome,
            expected_final_status=str(payload["expected_final_status"]),
            slice_name=str(payload["slice_name"]),
        )


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    payload: dict[str, Any]
    error: str | None = None
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowRun:
    agent: str
    case_id: str
    fault: FaultMode
    outcome: WorkflowOutcome
    final_status: str
    mutation_count: int
    tool_calls: int
    retries: int
    trace: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
