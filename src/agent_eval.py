"""Deterministic tool-routing agent and evaluation harness."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
import json
import re
from typing import Callable, Iterable


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    query: str
    expected_route: str
    expected_answer: str


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    name: str
    payload: dict[str, str]


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    route: str
    answer: str
    passed: bool
    failures: tuple[str, ...]
    trace: tuple[TraceEvent, ...]


CASES = [
    EvalCase("retrieval-01", "Which policy mentions badge sharing?", "retrieval", "policy-17"),
    EvalCase("query-01", "How many open findings are assigned to the quality team?", "structured_query", "7"),
    EvalCase("retrieval-02", "Find notes that mention a supplier audit delay.", "retrieval", "note-04"),
    EvalCase("query-02", "What is the average closure time for Q2 findings?", "structured_query", "4.5 days"),
    # Deliberately ambiguous case: the baseline routes it to retrieval, while the
    # expected contract requires a structured count over an indexed field.
    EvalCase("ambiguous-01", "List accounts with onboarding gaps in their notes.", "structured_query", "3"),
]


def retrieval_tool(query: str) -> str:
    if "badge" in query.lower():
        return "policy-17"
    if "supplier" in query.lower():
        return "note-04"
    return "no-match"


def structured_query_tool(query: str) -> str:
    lowered = query.lower()
    if "open findings" in lowered:
        return "7"
    if "average closure" in lowered:
        return "4.5 days"
    if "onboarding gaps" in lowered:
        return "3"
    return "0"


def route_query(query: str) -> str:
    """Choose a tool using a deliberately inspectable heuristic."""

    if re.search(r"\b(how many|count|average|sum|total)\b", query.lower()):
        return "structured_query"
    return "retrieval"


def run_agent(case: EvalCase) -> EvalResult:
    trace: list[TraceEvent] = []
    route = route_query(case.query)
    trace.append(TraceEvent("decision", "route", {"route": route}))
    tools: dict[str, Callable[[str], str]] = {
        "retrieval": retrieval_tool,
        "structured_query": structured_query_tool,
    }
    answer = tools[route](case.query)
    trace.append(TraceEvent("tool", route, {"query": case.query, "answer": answer}))
    failures: list[str] = []
    if route != case.expected_route:
        failures.append("route_mismatch")
    if answer != case.expected_answer:
        failures.append("answer_mismatch")
    trace.append(TraceEvent("result", "comparison", {"passed": str(not failures)}))
    return EvalResult(case.case_id, route, answer, not failures, tuple(failures), tuple(trace))


def evaluate(cases: Iterable[EvalCase] = CASES, repeats: int = 5) -> dict[str, object]:
    cases = list(cases)
    results = [run_agent(case) for _ in range(repeats) for case in cases]
    failures = Counter(failure for result in results for failure in result.failures)
    routes = Counter(result.route for result in results)
    return {
        "cases": len(cases),
        "repeats": repeats,
        "runs": len(results),
        "pass_rate": round(sum(result.passed for result in results) / len(results), 4),
        "routes": dict(sorted(routes.items())),
        "failures": dict(sorted(failures.items())),
        "traces": [
            {
                "case_id": result.case_id,
                "passed": result.passed,
                "failures": list(result.failures),
                "events": [asdict(event) for event in result.trace],
            }
            for result in results[: len(cases)]
        ],
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2, sort_keys=True))
