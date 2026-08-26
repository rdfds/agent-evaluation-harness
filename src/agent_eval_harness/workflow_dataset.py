"""Load the versioned stateful workflow fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dataset import REPOSITORY_ROOT
from .workflow_models import WorkflowCase


DEFAULT_WORKFLOW_CASES_PATH = REPOSITORY_ROOT / "data" / "workflow_cases.json"
DEFAULT_WORKFLOW_SEED_PATH = REPOSITORY_ROOT / "data" / "workflow_seed.json"


def load_workflow_cases(path: Path = DEFAULT_WORKFLOW_CASES_PATH) -> list[WorkflowCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = [WorkflowCase.from_dict(item) for item in payload["cases"]]
    identifiers = [case.case_id for case in cases]
    if not cases:
        raise ValueError("Workflow dataset is empty")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Workflow case IDs must be unique")
    return cases


def load_workflow_seed(path: Path = DEFAULT_WORKFLOW_SEED_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"requests", "approvals", "policies", "actors"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Workflow seed is missing: {', '.join(missing)}")
    return payload
