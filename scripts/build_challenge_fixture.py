"""Build atomic multi-request change sets for the challenge benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


POLICIES = [
    {"risk_level": "low", "version": 1, "required_role": "team_lead", "active": True},
    {"risk_level": "medium", "version": 2, "required_role": "quality_lead", "active": True},
    {"risk_level": "high", "version": 2, "required_role": "quality_director", "active": False},
    {"risk_level": "high", "version": 3, "required_role": "security_director", "active": True},
]


ACTORS = [
    {"actor": "deployment_agent", "can_write": True},
    {"actor": "compliance_agent", "can_write": True},
    {"actor": "audit_viewer", "can_write": False},
]


ROLE_BY_RISK = {
    "low": "team_lead",
    "medium": "quality_lead",
    "high": "security_director",
}


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    approvals: list[dict[str, str]] = []
    change_sets: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    def add_change_set(
        identifier: str,
        specs: list[tuple[str, str, str]],
        *,
        actor: str = "deployment_agent",
        expected_outcome: str = "applied",
        expected_status: str = "committed",
        slice_name: str,
    ) -> None:
        initial_states: dict[str, str] = {}
        final_states: dict[str, str] = {}
        mutations = 0
        change_sets.append(
            {
                "change_set_id": identifier,
                "status": "open",
                "atomic": True,
                "revision": 1,
            }
        )
        for ordinal, (risk, status, approval) in enumerate(specs, 1):
            request_id = f"{identifier}-r{ordinal}"
            initial_states[request_id] = status
            final_status = "approved" if expected_outcome in {"applied", "no_op"} else status
            final_states[request_id] = final_status
            if status != final_status:
                mutations += 1
            requests.append(
                {
                    "request_id": request_id,
                    "status": status,
                    "risk_level": risk,
                    "owner_team": "security" if risk == "high" else "operations",
                    "revision": ordinal + 2,
                }
            )
            items.append(
                {
                    "change_set_id": identifier,
                    "request_id": request_id,
                    "target_status": "approved",
                    "ordinal": ordinal,
                }
            )
            if approval != "missing":
                approvals.append(
                    {
                        "request_id": request_id,
                        "approver_role": ROLE_BY_RISK[risk],
                        "decision": approval,
                    }
                )
            if risk == "high":
                approvals.append(
                    {
                        "request_id": request_id,
                        "approver_role": "quality_director",
                        "decision": "approved",
                    }
                )
        cases.append(
            {
                "case_id": f"challenge-{identifier}",
                "change_set_id": identifier,
                "actor": actor,
                "expected_outcome": expected_outcome,
                "expected_change_set_status": expected_status,
                "expected_initial_states": initial_states,
                "expected_final_states": final_states,
                "expected_mutation_count": mutations,
                "slice_name": slice_name,
            }
        )

    add_change_set(
        "atomic-ready-3",
        [("low", "pending", "approved"), ("medium", "pending", "approved"), ("high", "pending", "approved")],
        slice_name="atomic_ready",
    )
    add_change_set(
        "mixed-state-4",
        [("low", "approved", "approved"), ("medium", "pending", "approved"), ("high", "pending", "approved"), ("low", "pending", "approved")],
        slice_name="mixed_idempotent",
    )
    add_change_set(
        "policy-version-3",
        [("high", "pending", "approved"), ("high", "pending", "approved"), ("medium", "pending", "approved")],
        actor="compliance_agent",
        slice_name="policy_version",
    )
    add_change_set(
        "long-horizon-5",
        [("low", "pending", "approved"), ("medium", "pending", "approved"), ("high", "pending", "approved"), ("medium", "pending", "approved"), ("low", "pending", "approved")],
        slice_name="long_horizon",
    )
    add_change_set(
        "repeated-policy-4",
        [("low", "pending", "approved"), ("low", "pending", "approved"), ("medium", "pending", "approved"), ("medium", "pending", "approved")],
        slice_name="repeated_policy",
    )
    add_change_set(
        "approval-missing-3",
        [("low", "approved", "approved"), ("medium", "pending", "missing"), ("high", "pending", "approved")],
        actor="compliance_agent",
        expected_outcome="blocked",
        expected_status="aborted",
        slice_name="approval_blocked",
    )
    add_change_set(
        "approval-rejected-4",
        [("low", "pending", "approved"), ("medium", "pending", "rejected"), ("high", "pending", "approved"), ("low", "pending", "approved")],
        expected_outcome="blocked",
        expected_status="aborted",
        slice_name="approval_blocked",
    )
    add_change_set(
        "actor-blocked-3",
        [("low", "pending", "approved"), ("medium", "pending", "approved"), ("high", "pending", "approved")],
        actor="audit_viewer",
        expected_outcome="blocked",
        expected_status="aborted",
        slice_name="permission_blocked",
    )
    add_change_set(
        "already-satisfied-3",
        [("low", "approved", "approved"), ("medium", "approved", "approved"), ("high", "approved", "approved")],
        expected_outcome="no_op",
        expected_status="satisfied",
        slice_name="idempotent",
    )
    add_change_set(
        "mixed-large-5",
        [("high", "approved", "approved"), ("low", "pending", "approved"), ("medium", "approved", "approved"), ("high", "pending", "approved"), ("medium", "pending", "approved")],
        actor="compliance_agent",
        slice_name="mixed_long_horizon",
    )

    if len(cases) != 10:
        raise AssertionError(f"Expected 10 challenge cases, built {len(cases)}")
    seed = {
        "schema_version": 1,
        "requests": requests,
        "approvals": approvals,
        "policies": POLICIES,
        "actors": ACTORS,
        "change_sets": change_sets,
        "change_set_items": items,
    }
    dataset = {"schema_version": 1, "cases": cases}
    return seed, dataset


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed, dataset = build()
    (DATA_DIR / "challenge_seed.json").write_text(
        json.dumps(seed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (DATA_DIR / "challenge_cases.json").write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(dataset['cases'])} challenge cases and "
        f"{len(seed['requests'])} request records to {DATA_DIR}"
    )


if __name__ == "__main__":
    main()
