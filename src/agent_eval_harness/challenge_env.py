"""Atomic change-set tools with composable faults at real call boundaries."""

from __future__ import annotations

from collections import Counter
import sqlite3
from typing import Any

from .challenge_dataset import load_challenge_seed
from .challenge_models import ChallengeFaultPlan
from .workflow_models import ToolResult


FAULT_PLANS: dict[ChallengeFaultPlan, tuple[str, ...]] = {
    "none": (),
    "transient_chain": ("policy_timeout_once", "approval_rate_limit_once"),
    "evidence_drift": ("request_schema_alias", "approval_partial_once"),
    "concurrency_race": ("stale_read_once", "commit_conflict_once"),
    "ambiguous_commit": ("approval_timeout_once", "commit_timeout_after_apply"),
    "degraded_auth": ("request_schema_alias", "stage_permission_denied"),
}


class ChallengeEnvironment:
    """A fresh SQLite transaction domain for one challenge episode."""

    def __init__(
        self,
        seed: dict[str, Any] | None = None,
        *,
        fault_plan: ChallengeFaultPlan = "none",
    ) -> None:
        if fault_plan not in FAULT_PLANS:
            raise ValueError(f"Unsupported challenge fault plan {fault_plan!r}")
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.fault_plan = fault_plan
        self.fault_components = FAULT_PLANS[fault_plan]
        self.activations: Counter[str] = Counter()
        self.trace: list[dict[str, Any]] = []
        self._initialize(seed or load_challenge_seed())

    def __enter__(self) -> "ChallengeEnvironment":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @property
    def activated_faults(self) -> tuple[str, ...]:
        return tuple(
            component
            for component in self.fault_components
            if self.activations[component] > 0
        )

    def _initialize(self, seed: dict[str, Any]) -> None:
        self.connection.executescript(
            """
            CREATE TABLE requests (
                request_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                owner_team TEXT NOT NULL,
                revision INTEGER NOT NULL
            );
            CREATE TABLE approvals (
                request_id TEXT NOT NULL,
                approver_role TEXT NOT NULL,
                decision TEXT NOT NULL
            );
            CREATE TABLE policies (
                risk_level TEXT NOT NULL,
                version INTEGER NOT NULL,
                required_role TEXT NOT NULL,
                active INTEGER NOT NULL,
                PRIMARY KEY (risk_level, version)
            );
            CREATE TABLE actors (
                actor TEXT PRIMARY KEY,
                can_write INTEGER NOT NULL
            );
            CREATE TABLE change_sets (
                change_set_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                atomic INTEGER NOT NULL,
                revision INTEGER NOT NULL
            );
            CREATE TABLE change_set_items (
                change_set_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                target_status TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                PRIMARY KEY (change_set_id, request_id)
            );
            CREATE TABLE staged_changes (
                change_set_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                target_status TEXT NOT NULL,
                expected_revision INTEGER NOT NULL,
                PRIMARY KEY (change_set_id, request_id)
            );
            CREATE TABLE mutation_log (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                change_set_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                from_revision INTEGER NOT NULL,
                to_revision INTEGER NOT NULL
            );
            """
        )
        self.connection.executemany(
            "INSERT INTO requests VALUES (:request_id, :status, :risk_level, :owner_team, :revision)",
            seed["requests"],
        )
        self.connection.executemany(
            "INSERT INTO approvals VALUES (:request_id, :approver_role, :decision)",
            seed["approvals"],
        )
        self.connection.executemany(
            "INSERT INTO policies VALUES (:risk_level, :version, :required_role, :active)",
            seed["policies"],
        )
        self.connection.executemany(
            "INSERT INTO actors VALUES (:actor, :can_write)", seed["actors"]
        )
        self.connection.executemany(
            "INSERT INTO change_sets VALUES (:change_set_id, :status, :atomic, :revision)",
            seed["change_sets"],
        )
        self.connection.executemany(
            "INSERT INTO change_set_items VALUES (:change_set_id, :request_id, :target_status, :ordinal)",
            seed["change_set_items"],
        )
        self.connection.commit()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any]:
        return dict(row) if row is not None else {}

    def request_state(self, request_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT request_id, status, risk_level, owner_team, revision "
            "FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._row(row)

    def request_states(self, change_set_id: str) -> dict[str, str]:
        rows = self.connection.execute(
            "SELECT r.request_id, r.status FROM requests r "
            "JOIN change_set_items i ON i.request_id = r.request_id "
            "WHERE i.change_set_id = ? ORDER BY i.ordinal",
            (change_set_id,),
        ).fetchall()
        return {str(row["request_id"]): str(row["status"]) for row in rows}

    def change_set_status(self, change_set_id: str) -> str:
        row = self.connection.execute(
            "SELECT status FROM change_sets WHERE change_set_id = ?", (change_set_id,)
        ).fetchone()
        if row is None:
            raise KeyError(change_set_id)
        return str(row["status"])

    def mutation_count(self, change_set_id: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM mutation_log WHERE change_set_id = ?",
            (change_set_id,),
        ).fetchone()
        return int(row["count"])

    def call(self, tool_name: str, **arguments: Any) -> ToolResult:
        call_id = sum(event.get("stage") == "tool_call" for event in self.trace) + 1
        self.trace.append(
            {
                "stage": "tool_call",
                "call_id": call_id,
                "tool": tool_name,
                "arguments": arguments,
            }
        )
        result, applied = self._before_call(tool_name, arguments)
        if result is None:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                result = ToolResult(False, {}, error="unknown_tool", retryable=False)
            else:
                result = handler(**arguments)
            result, after_applied = self._after_call(tool_name, result)
            applied.extend(after_applied)
        self.trace.append(
            {
                "stage": "tool_result",
                "call_id": call_id,
                "tool": tool_name,
                "fault_plan": self.fault_plan,
                "faults_applied": applied,
                "result": result.to_dict(),
            }
        )
        return result

    def _activate(self, component: str) -> None:
        self.activations[component] += 1

    def _before_call(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[ToolResult | None, list[str]]:
        applied: list[str] = []
        one_shot_errors = {
            "policy_timeout_once": ("get_active_policy", "timeout"),
            "approval_rate_limit_once": ("list_approvals", "rate_limited"),
            "approval_timeout_once": ("list_approvals", "timeout"),
        }
        for component, (target, error) in one_shot_errors.items():
            if (
                component in self.fault_components
                and tool_name == target
                and self.activations[component] == 0
            ):
                self._activate(component)
                applied.append(component)
                return ToolResult(False, {}, error=error, retryable=True), applied

        if "stage_permission_denied" in self.fault_components and tool_name == "stage_transition":
            self._activate("stage_permission_denied")
            applied.append("stage_permission_denied")
            return ToolResult(False, {}, error="permission_denied", retryable=False), applied

        if (
            "commit_conflict_once" in self.fault_components
            and tool_name == "commit_change_set"
            and self.activations["commit_conflict_once"] == 0
        ):
            staged = self.connection.execute(
                "SELECT request_id FROM staged_changes WHERE change_set_id = ? "
                "ORDER BY request_id LIMIT 1",
                (arguments.get("change_set_id"),),
            ).fetchone()
            if staged is not None:
                request_id = str(staged["request_id"])
                self.connection.execute(
                    "UPDATE requests SET revision = revision + 1 WHERE request_id = ?",
                    (request_id,),
                )
                self.connection.execute(
                    "DELETE FROM staged_changes WHERE change_set_id = ? AND request_id = ?",
                    (arguments.get("change_set_id"), request_id),
                )
                self.connection.commit()
                self._activate("commit_conflict_once")
                applied.append("commit_conflict_once")
                return (
                    ToolResult(
                        False,
                        {"request_id": request_id},
                        error="revision_conflict",
                        retryable=True,
                    ),
                    applied,
                )
        return None, applied

    def _after_call(
        self, tool_name: str, result: ToolResult
    ) -> tuple[ToolResult, list[str]]:
        if not result.ok:
            return result, []
        applied: list[str] = []
        payload = dict(result.payload)
        if "request_schema_alias" in self.fault_components and tool_name == "get_request":
            if "status" in payload and "revision" in payload:
                payload["state"] = payload.pop("status")
                payload["version"] = payload.pop("revision")
                self._activate("request_schema_alias")
                applied.append("request_schema_alias")
        if (
            "stale_read_once" in self.fault_components
            and tool_name == "get_request"
            and self.activations["stale_read_once"] == 0
        ):
            revision_key = "version" if "version" in payload else "revision"
            payload[revision_key] = max(0, int(payload[revision_key]) - 1)
            self._activate("stale_read_once")
            applied.append("stale_read_once")
        if (
            "approval_partial_once" in self.fault_components
            and tool_name == "list_approvals"
            and self.activations["approval_partial_once"] == 0
        ):
            approvals = [dict(item) for item in payload.get("approvals", [])]
            for approval in approvals:
                approval.pop("decision", None)
            payload["approvals"] = approvals
            self._activate("approval_partial_once")
            applied.append("approval_partial_once")
        if (
            "commit_timeout_after_apply" in self.fault_components
            and tool_name == "commit_change_set"
            and self.activations["commit_timeout_after_apply"] == 0
            and result.payload.get("status") in {"committed", "satisfied"}
        ):
            self._activate("commit_timeout_after_apply")
            applied.append("commit_timeout_after_apply")
            return (
                ToolResult(
                    False,
                    {"operation": "commit_change_set", "outcome": "unknown"},
                    error="timeout",
                    retryable=True,
                ),
                applied,
            )
        return ToolResult(True, payload), applied

    def _tool_get_change_set(self, change_set_id: str) -> ToolResult:
        change_set = self.connection.execute(
            "SELECT change_set_id, status, atomic, revision FROM change_sets "
            "WHERE change_set_id = ?",
            (change_set_id,),
        ).fetchone()
        if change_set is None:
            return ToolResult(False, {}, error="change_set_not_found", retryable=False)
        items = self.connection.execute(
            "SELECT request_id, target_status, ordinal FROM change_set_items "
            "WHERE change_set_id = ? ORDER BY ordinal",
            (change_set_id,),
        ).fetchall()
        return ToolResult(
            True,
            {
                **self._row(change_set),
                "items": [dict(row) for row in items],
            },
        )

    def _tool_get_request(self, request_id: str) -> ToolResult:
        try:
            return ToolResult(True, self.request_state(request_id))
        except KeyError:
            return ToolResult(False, {}, error="request_not_found", retryable=False)

    def _tool_get_active_policy(self, risk_level: str) -> ToolResult:
        row = self.connection.execute(
            "SELECT risk_level, version, required_role, active FROM policies "
            "WHERE risk_level = ? AND active = 1 ORDER BY version DESC LIMIT 1",
            (risk_level,),
        ).fetchone()
        if row is None:
            return ToolResult(False, {}, error="policy_not_found", retryable=False)
        return ToolResult(True, self._row(row))

    def _tool_list_approvals(self, request_id: str) -> ToolResult:
        rows = self.connection.execute(
            "SELECT approver_role, decision FROM approvals WHERE request_id = ? "
            "ORDER BY approver_role",
            (request_id,),
        ).fetchall()
        return ToolResult(
            True,
            {"request_id": request_id, "approvals": [dict(row) for row in rows]},
        )

    def _can_write(self, actor: str) -> bool:
        row = self.connection.execute(
            "SELECT can_write FROM actors WHERE actor = ?", (actor,)
        ).fetchone()
        return row is not None and bool(row["can_write"])

    def _approval_valid(self, request_id: str, risk_level: str) -> bool:
        policy = self.connection.execute(
            "SELECT required_role FROM policies WHERE risk_level = ? AND active = 1 "
            "ORDER BY version DESC LIMIT 1",
            (risk_level,),
        ).fetchone()
        if policy is None:
            return False
        approval = self.connection.execute(
            "SELECT 1 FROM approvals WHERE request_id = ? AND approver_role = ? "
            "AND decision = 'approved'",
            (request_id, policy["required_role"]),
        ).fetchone()
        return approval is not None

    def _tool_stage_transition(
        self,
        change_set_id: str,
        request_id: str,
        actor: str,
        target_status: str,
        expected_revision: int,
    ) -> ToolResult:
        try:
            status = self.change_set_status(change_set_id)
        except KeyError:
            return ToolResult(False, {}, error="change_set_not_found", retryable=False)
        if status != "open":
            return ToolResult(False, {"status": status}, error="change_set_closed", retryable=False)
        if not self._can_write(actor):
            return ToolResult(False, {}, error="permission_denied", retryable=False)
        membership = self.connection.execute(
            "SELECT target_status FROM change_set_items WHERE change_set_id = ? AND request_id = ?",
            (change_set_id, request_id),
        ).fetchone()
        if membership is None or str(membership["target_status"]) != target_status:
            return ToolResult(False, {}, error="item_mismatch", retryable=False)
        try:
            request = self.request_state(request_id)
        except KeyError:
            return ToolResult(False, {}, error="request_not_found", retryable=False)
        if request["status"] == target_status:
            return ToolResult(True, {**request, "staged": False, "already_satisfied": True})
        if request["status"] != "pending" or target_status != "approved":
            return ToolResult(False, {}, error="invalid_transition", retryable=False)
        if int(request["revision"]) != int(expected_revision):
            return ToolResult(False, {}, error="revision_conflict", retryable=True)
        if not self._approval_valid(request_id, str(request["risk_level"])):
            return ToolResult(False, {}, error="approval_missing", retryable=False)
        self.connection.execute(
            "INSERT INTO staged_changes VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(change_set_id, request_id) DO UPDATE SET "
            "actor = excluded.actor, target_status = excluded.target_status, "
            "expected_revision = excluded.expected_revision",
            (change_set_id, request_id, actor, target_status, expected_revision),
        )
        self.connection.commit()
        return ToolResult(True, {**request, "staged": True})

    def _review(self, change_set_id: str) -> dict[str, Any]:
        items = self.connection.execute(
            "SELECT i.request_id, i.target_status, i.ordinal, r.status, r.risk_level, "
            "r.revision, s.expected_revision FROM change_set_items i "
            "JOIN requests r ON r.request_id = i.request_id "
            "LEFT JOIN staged_changes s ON s.change_set_id = i.change_set_id "
            "AND s.request_id = i.request_id "
            "WHERE i.change_set_id = ? ORDER BY i.ordinal",
            (change_set_id,),
        ).fetchall()
        blockers: list[dict[str, str]] = []
        staged = 0
        satisfied = 0
        for row in items:
            request_id = str(row["request_id"])
            if row["status"] == row["target_status"]:
                satisfied += 1
                continue
            if not self._approval_valid(request_id, str(row["risk_level"])):
                blockers.append({"request_id": request_id, "reason": "approval_missing"})
            if row["expected_revision"] is None:
                blockers.append({"request_id": request_id, "reason": "not_staged"})
            elif int(row["expected_revision"]) != int(row["revision"]):
                blockers.append({"request_id": request_id, "reason": "revision_conflict"})
            else:
                staged += 1
        return {
            "change_set_id": change_set_id,
            "ready": not blockers,
            "items": len(items),
            "staged": staged,
            "already_satisfied": satisfied,
            "blockers": blockers,
        }

    def _tool_review_change_set(self, change_set_id: str) -> ToolResult:
        try:
            status = self.change_set_status(change_set_id)
        except KeyError:
            return ToolResult(False, {}, error="change_set_not_found", retryable=False)
        return ToolResult(True, {"status": status, **self._review(change_set_id)})

    def _tool_commit_change_set(self, change_set_id: str, actor: str) -> ToolResult:
        try:
            status = self.change_set_status(change_set_id)
        except KeyError:
            return ToolResult(False, {}, error="change_set_not_found", retryable=False)
        if status in {"committed", "satisfied"}:
            return ToolResult(True, {"change_set_id": change_set_id, "status": status, "idempotent": True})
        if status != "open":
            return ToolResult(False, {"status": status}, error="change_set_closed", retryable=False)
        if not self._can_write(actor):
            return ToolResult(False, {}, error="permission_denied", retryable=False)
        review = self._review(change_set_id)
        if not review["ready"]:
            retryable = any(item["reason"] == "revision_conflict" for item in review["blockers"])
            return ToolResult(
                False,
                {"blockers": review["blockers"]},
                error="validation_failed",
                retryable=retryable,
            )
        staged = self.connection.execute(
            "SELECT s.request_id, s.target_status, s.expected_revision, r.status "
            "FROM staged_changes s JOIN requests r ON r.request_id = s.request_id "
            "WHERE s.change_set_id = ? ORDER BY s.request_id",
            (change_set_id,),
        ).fetchall()
        try:
            for row in staged:
                new_revision = int(row["expected_revision"]) + 1
                cursor = self.connection.execute(
                    "UPDATE requests SET status = ?, revision = ? "
                    "WHERE request_id = ? AND revision = ?",
                    (
                        row["target_status"],
                        new_revision,
                        row["request_id"],
                        row["expected_revision"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise sqlite3.IntegrityError("revision conflict")
                self.connection.execute(
                    "INSERT INTO mutation_log "
                    "(change_set_id, request_id, actor, from_status, to_status, from_revision, to_revision) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        change_set_id,
                        row["request_id"],
                        actor,
                        row["status"],
                        row["target_status"],
                        row["expected_revision"],
                        new_revision,
                    ),
                )
            final_status = "committed" if staged else "satisfied"
            self.connection.execute(
                "UPDATE change_sets SET status = ?, revision = revision + 1 WHERE change_set_id = ?",
                (final_status, change_set_id),
            )
            self.connection.commit()
        except sqlite3.IntegrityError:
            self.connection.rollback()
            return ToolResult(False, {}, error="revision_conflict", retryable=True)
        return ToolResult(
            True,
            {
                "change_set_id": change_set_id,
                "status": final_status,
                "mutations": len(staged),
            },
        )

    def _tool_abort_change_set(
        self, change_set_id: str, actor: str, reason: str
    ) -> ToolResult:
        del actor
        try:
            status = self.change_set_status(change_set_id)
        except KeyError:
            return ToolResult(False, {}, error="change_set_not_found", retryable=False)
        if status == "aborted":
            return ToolResult(True, {"change_set_id": change_set_id, "status": status, "idempotent": True})
        if status != "open":
            return ToolResult(False, {"status": status}, error="change_set_closed", retryable=False)
        self.connection.execute(
            "DELETE FROM staged_changes WHERE change_set_id = ?", (change_set_id,)
        )
        self.connection.execute(
            "UPDATE change_sets SET status = 'aborted', revision = revision + 1 "
            "WHERE change_set_id = ?",
            (change_set_id,),
        )
        self.connection.commit()
        return ToolResult(
            True,
            {"change_set_id": change_set_id, "status": "aborted", "reason": reason},
        )
