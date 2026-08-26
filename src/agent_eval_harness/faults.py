"""Mutation agents used to prove that each regression gate can fail."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .models import AgentResponse, TraceEvent
from .runtime import ABSTAIN_ANSWER, Agent


FaultMode = Literal["route_swap", "stale_answer", "drop_evidence", "disable_abstain"]


class FaultInjectingAgent:
    def __init__(self, base: Agent, mode: FaultMode):
        self.base = base
        self.mode = mode
        self.name = f"{base.name}__fault_{mode}"

    def answer(self, query: str) -> AgentResponse:
        response = self.base.answer(query)
        trace = response.trace + (TraceEvent("fault_injection", {"mode": self.mode}),)
        if self.mode == "route_swap" and response.route == "retrieval":
            return replace(response, route="structured_query", trace=trace)
        if self.mode == "stale_answer" and response.route != "abstain":
            return replace(response, answer="STALE_RESULT", trace=trace)
        if self.mode == "drop_evidence" and response.route != "abstain":
            return replace(response, evidence_ids=(), trace=trace)
        if self.mode == "disable_abstain" and response.answer == ABSTAIN_ANSWER:
            return replace(
                response,
                route="retrieval",
                answer="UNSUPPORTED_GUESS",
                evidence_ids=(),
                trace=trace,
            )
        return replace(response, trace=trace)

