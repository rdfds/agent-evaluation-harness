"""Load and validate the atomic challenge fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .challenge_models import ChallengeCase


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHALLENGE_CASES_PATH = REPOSITORY_ROOT / "data" / "challenge_cases.json"
DEFAULT_CHALLENGE_SEED_PATH = REPOSITORY_ROOT / "data" / "challenge_seed.json"


def load_challenge_cases(
    path: Path = DEFAULT_CHALLENGE_CASES_PATH,
) -> list[ChallengeCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported challenge case schema")
    cases = [ChallengeCase.from_dict(item) for item in payload.get("cases", [])]
    identifiers = [case.case_id for case in cases]
    if not cases or len(identifiers) != len(set(identifiers)):
        raise ValueError("Challenge cases must be non-empty with unique IDs")
    return cases


def load_challenge_seed(
    path: Path = DEFAULT_CHALLENGE_SEED_PATH,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported challenge seed schema")
    required = {
        "requests",
        "approvals",
        "policies",
        "actors",
        "change_sets",
        "change_set_items",
    }
    if not required.issubset(payload):
        raise ValueError("Challenge seed is missing required collections")
    return payload
