"""Inspectable baseline and candidate agents backed by deterministic tools."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter
from typing import Any, Protocol

from .models import AgentResponse, TraceEvent, Usage


ABSTAIN_ANSWER = "INSUFFICIENT_EVIDENCE"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUANTITATIVE_CUE = re.compile(
    r"\b(how many|count|average|mean|sum|total|number of|quantity|backlog)\b"
)
DOCUMENT_CUE = re.compile(
    r"\b(policy|note|memo|document|guidance|control|mentions|states|says|artifact)\b"
)


class Agent(Protocol):
    name: str

    def answer(self, query: str) -> AgentResponse: ...


@dataclass(frozen=True)
class CostProfile:
    input_per_million_usd: float = 0.25
    output_per_million_usd: float = 1.0

    def estimate(self, query: str, answer: str) -> Usage:
        input_tokens = max(1, len(TOKEN_PATTERN.findall(query)))
        output_tokens = max(1, len(TOKEN_PATTERN.findall(answer)))
        cost = (
            input_tokens * self.input_per_million_usd
            + output_tokens * self.output_per_million_usd
        ) / 1_000_000
        return Usage(input_tokens, output_tokens, round(cost, 8))


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


class KnowledgeBase:
    def __init__(self, payload: dict[str, Any]):
        self.documents = list(payload["documents"])
        self.structured_intents = list(payload["structured_intents"])
        self.out_of_scope_terms = tuple(payload["out_of_scope_terms"])

    @staticmethod
    def _alias_score(query: str, aliases: list[str]) -> float:
        query_normalized = " ".join(TOKEN_PATTERN.findall(query.lower()))
        query_tokens = _tokens(query)
        best = 0.0
        for alias in aliases:
            alias_normalized = " ".join(TOKEN_PATTERN.findall(alias.lower()))
            alias_tokens = _tokens(alias)
            if alias_normalized and alias_normalized in query_normalized:
                best = max(best, 1.0)
                continue
            if alias_tokens:
                best = max(best, len(query_tokens & alias_tokens) / len(alias_tokens))
        return best

    def best_document(self, query: str, *, legacy: bool = False) -> tuple[dict[str, Any] | None, float]:
        best_document: dict[str, Any] | None = None
        best_score = 0.0
        for document in self.documents:
            aliases = document["aliases"][:1] if legacy else document["aliases"]
            score = self._alias_score(query, aliases)
            if score > best_score:
                best_document = document
                best_score = score
        threshold = 1.0 if legacy else 0.66
        return (best_document, best_score) if best_score >= threshold else (None, best_score)

    def best_intent(self, query: str) -> tuple[dict[str, Any] | None, float]:
        best_intent: dict[str, Any] | None = None
        best_score = 0.0
        for intent in self.structured_intents:
            score = self._alias_score(query, intent["aliases"])
            if score > best_score:
                best_intent = intent
                best_score = score
        return (best_intent, best_score) if best_score >= 0.66 else (None, best_score)

    def is_out_of_scope(self, query: str) -> bool:
        lowered = query.lower()
        return any(term in lowered for term in self.out_of_scope_terms)


class _BaseAgent:
    name = "base"

    def __init__(self, knowledge: KnowledgeBase, cost_profile: CostProfile | None = None):
        self.knowledge = knowledge
        self.cost_profile = cost_profile or CostProfile()

    def _response(
        self,
        *,
        started: float,
        query: str,
        route: str,
        answer: str,
        evidence_ids: tuple[str, ...],
        confidence: float,
        trace: list[TraceEvent],
    ) -> AgentResponse:
        latency_ms = (perf_counter() - started) * 1_000
        return AgentResponse(
            route=route,  # type: ignore[arg-type]
            answer=answer,
            evidence_ids=evidence_ids,
            confidence=round(confidence, 4),
            latency_ms=latency_ms,
            usage=self.cost_profile.estimate(query, answer),
            trace=tuple(trace),
        )

    def _call_retrieval(
        self, query: str, *, legacy: bool = False
    ) -> tuple[str, tuple[str, ...], float]:
        document, score = self.knowledge.best_document(query, legacy=legacy)
        if document is None:
            return "NO_MATCH", (), score
        return document["answer"], (document["id"],), score

    def _call_structured(self, query: str) -> tuple[str, tuple[str, ...], float]:
        intent, score = self.knowledge.best_intent(query)
        if intent is None:
            return "NO_MATCH", (), score
        return intent["answer"], (intent["evidence_id"],), score


class LegacyAgent(_BaseAgent):
    """A keyword router retained as the comparison baseline."""

    name = "legacy_keyword_router"

    def answer(self, query: str) -> AgentResponse:
        started = perf_counter()
        route = "structured_query" if QUANTITATIVE_CUE.search(query.lower()) else "retrieval"
        trace = [TraceEvent("route", {"route": route, "policy": self.name})]
        if route == "structured_query":
            answer, evidence, confidence = self._call_structured(query)
        else:
            answer, evidence, confidence = self._call_retrieval(query, legacy=True)
        trace.append(
            TraceEvent(
                "tool",
                {"route": route, "evidence_ids": list(evidence), "confidence": confidence},
            )
        )
        return self._response(
            started=started,
            query=query,
            route=route,
            answer=answer,
            evidence_ids=evidence,
            confidence=confidence,
            trace=trace,
        )


class CandidateAgent(_BaseAgent):
    """Evidence-aware router with explicit abstention and traceable decisions."""

    name = "evidence_aware_router_v2"

    def answer(self, query: str) -> AgentResponse:
        started = perf_counter()
        trace: list[TraceEvent] = []
        if self.knowledge.is_out_of_scope(query):
            trace.append(TraceEvent("route", {"route": "abstain", "reason": "out_of_scope"}))
            return self._response(
                started=started,
                query=query,
                route="abstain",
                answer=ABSTAIN_ANSWER,
                evidence_ids=(),
                confidence=1.0,
                trace=trace,
            )

        document, document_score = self.knowledge.best_document(query)
        intent, intent_score = self.knowledge.best_intent(query)
        if DOCUMENT_CUE.search(query.lower()) and document is not None:
            route = "retrieval"
            reason = "document_cue"
        elif intent is not None and (QUANTITATIVE_CUE.search(query.lower()) or intent_score > document_score):
            route = "structured_query"
            reason = "structured_intent"
        elif document is not None:
            route = "retrieval"
            reason = "retrieval_match"
        else:
            route = "abstain"
            reason = "insufficient_evidence"

        trace.append(
            TraceEvent(
                "route",
                {
                    "route": route,
                    "reason": reason,
                    "document_score": round(document_score, 4),
                    "intent_score": round(intent_score, 4),
                },
            )
        )
        if route == "retrieval":
            answer, evidence, confidence = self._call_retrieval(query)
        elif route == "structured_query":
            answer, evidence, confidence = self._call_structured(query)
        else:
            answer, evidence, confidence = ABSTAIN_ANSWER, (), max(document_score, intent_score)
        trace.append(
            TraceEvent(
                "tool",
                {"route": route, "evidence_ids": list(evidence), "confidence": confidence},
            )
        )
        return self._response(
            started=started,
            query=query,
            route=route,
            answer=answer,
            evidence_ids=evidence,
            confidence=confidence,
            trace=trace,
        )

