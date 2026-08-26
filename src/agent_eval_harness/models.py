"""Typed contracts shared by datasets, agents, and evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Route = Literal["retrieval", "structured_query", "abstain"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected_route: Route
    expected_answer: str
    required_evidence: tuple[str, ...]
    slice_name: str
    difficulty: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        route = payload["expected_route"]
        if route not in {"retrieval", "structured_query", "abstain"}:
            raise ValueError(f"Unsupported route {route!r}")
        return cls(
            case_id=str(payload["case_id"]),
            query=str(payload["query"]),
            expected_route=route,
            expected_answer=str(payload["expected_answer"]),
            required_evidence=tuple(payload.get("required_evidence", [])),
            slice_name=str(payload["slice_name"]),
            difficulty=str(payload["difficulty"]),
        )


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class TraceEvent:
    stage: str
    detail: dict[str, Any]


@dataclass(frozen=True)
class AgentResponse:
    route: Route
    answer: str
    evidence_ids: tuple[str, ...]
    confidence: float
    latency_ms: float
    usage: Usage
    trace: tuple[TraceEvent, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

