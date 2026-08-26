"""Load and validate committed evaluation fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import EvalCase


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES_PATH = REPOSITORY_ROOT / "data" / "eval_cases.jsonl"
DEFAULT_KNOWLEDGE_PATH = REPOSITORY_ROOT / "data" / "knowledge_base.json"


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    cases = [
        EvalCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation case IDs must be unique")
    if not cases:
        raise ValueError("Evaluation dataset is empty")
    return cases


def load_knowledge_base(path: Path = DEFAULT_KNOWLEDGE_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("documents") or not payload.get("structured_intents"):
        raise ValueError("Knowledge base requires documents and structured intents")
    return payload

