"""Build 24 stateful change-control tasks and their SQLite seed records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


POLICIES = [
    {"risk_level": "low", "version": 1, "required_role": "team_lead", "active": True},
    {"risk_level": "medium", "version": 1, "required_role": "quality_lead", "active": True},
    {"risk_level": "high", "version": 1, "required_role": "quality_lead", "active": False},
    {"risk_level": "high", "version": 2, "required_role": "quality_director", "active": True},
]


ACTORS = [
    {"actor": "ops_agent", "can_write": True},
    {"actor": "quality_agent", "can_write": True},
    {"actor": "read_only_bot", "can_write": False},
]


def _request(identifier: str, risk: str, status: str = "pending") -> dict[str, object]:
    return {
        "request_id": identifier,
        "status": status,
        "risk_level": risk,
        "owner_team": "quality" if risk == "high" else "operations",
        "revision": 3,
    }


def _approval(identifier: str, role: str, decision: str = "approved") -> dict[str, str]:
    return {"request_id": identifier, "approver_role": role, "decision": decision}


def _case(
    identifier: str,
    actor: str,
    expected_outcome: str,
    expected_final_status: str,
    slice_name: str,
) -> dict[str, str]:
    return {
        "case_id": f"workflow-{identifier}",
        "request_id": identifier,
        "actor": actor,
        "target_status": "approved",
        "expected_outcome": expected_outcome,
        "expected_final_status": expected_final_status,
        "slice_name": slice_name,
    }


def build() -> tuple[dict[str, object], dict[str, object]]:
    requests: list[dict[str, object]] = []
    approvals: list[dict[str, str]] = []
    cases: list[dict[str, str]] = []

    risk_to_role = {"low": "team_lead", "medium": "quality_lead", "high": "quality_director"}
    for number in range(1, 7):
        risk = ["low", "medium", "high"][(number - 1) % 3]
        identifier = f"ready-{number:02d}"
        requests.append(_request(identifier, risk))
        approvals.append(_approval(identifier, risk_to_role[risk]))
        cases.append(_case(identifier, "ops_agent", "applied", "approved", "ready"))

    for number in range(1, 5):
        identifier = f"approval-blocked-{number:02d}"
        risk = "medium" if number % 2 else "high"
        requests.append(_request(identifier, risk))
        if number % 2 == 0:
            approvals.append(_approval(identifier, risk_to_role[risk], decision="rejected"))
        cases.append(_case(identifier, "quality_agent", "blocked", "pending", "approval_blocked"))

    for number in range(1, 5):
        identifier = f"permission-blocked-{number:02d}"
        risk = "low" if number % 2 else "medium"
        requests.append(_request(identifier, risk))
        approvals.append(_approval(identifier, risk_to_role[risk]))
        cases.append(_case(identifier, "read_only_bot", "blocked", "pending", "permission_blocked"))

    for number in range(1, 5):
        identifier = f"idempotent-{number:02d}"
        requests.append(_request(identifier, "low", status="approved"))
        cases.append(_case(identifier, "ops_agent", "no_op", "approved", "idempotent"))

    for number in range(1, 4):
        identifier = f"policy-v2-{number:02d}"
        requests.append(_request(identifier, "high"))
        approvals.extend(
            [
                _approval(identifier, "quality_lead"),
                _approval(identifier, "quality_director"),
            ]
        )
        cases.append(_case(identifier, "quality_agent", "applied", "approved", "policy_version"))

    for number in range(1, 4):
        identifier = f"revision-{number:02d}"
        requests.append(_request(identifier, "medium"))
        approvals.append(_approval(identifier, "quality_lead"))
        cases.append(_case(identifier, "ops_agent", "applied", "approved", "revision_sensitive"))

    if len(cases) != 24:
        raise AssertionError(f"Expected 24 workflow cases, built {len(cases)}")
    seed = {
        "schema_version": 1,
        "requests": requests,
        "approvals": approvals,
        "policies": POLICIES,
        "actors": ACTORS,
    }
    dataset = {"schema_version": 1, "cases": cases}
    return seed, dataset


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed, dataset = build()
    (DATA_DIR / "workflow_seed.json").write_text(
        json.dumps(seed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "workflow_cases.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(dataset['cases'])} workflow cases to {DATA_DIR}")


if __name__ == "__main__":
    main()
