"""SQLite-backed workflow tools with faults injected at the call boundary."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Any

from .workflow_dataset import load_workflow_seed
from .workflow_models import FaultMode, ToolResult


FAULT_TARGETS = {
    "timeout_once": "get_active_policy",
    "schema_drift": "get_request",
    "stale_read": "get_request",
    "partial_response": "list_approvals",
    "permission_denied": "apply_transition",
}


@dataclass
class FaultState:
    mode: FaultMode = "none"
    activations: int = 0

    def targets(self, tool_name: str) -> bool:
        return self.mode != "none" and FAULT_TARGETS[self.mode] == tool_name


class WorkflowEnvironment:
    """A fresh transactional workflow database for one evaluation episode."""

    def __init__(
        self,
        seed: dict[str, Any] | None = None,
        *,
        fault: FaultMode = "none",
    ) -> None:
        if fault not in {"none", *FAULT_TARGETS}:
            raise ValueError(f"Unsupported workflow fault {fault!r}")
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self.fault = FaultState(fault)
        self.trace: list[dict[str, Any]] = []
        self._initialize(seed or load_workflow_seed())

    def __enter__(self) -> "WorkflowEnvironment":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

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
            CREATE TABLE mutation_log (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        self.connection.commit()

    @staticmethod
    def _row_payload(row: sqlite3.Row | None) -> dict[str, Any]:
        return dict(row) if row is not None else {}

    def request_state(self, request_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT request_id, status, risk_level, owner_team, revision "
            "FROM requests WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._row_payload(row)

    def mutation_count(self, request_id: str | None = None) -> int:
        if request_id is None:
            row = self.connection.execute("SELECT COUNT(*) AS count FROM mutation_log").fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS count FROM mutation_log WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return int(row["count"])

    def call(self, tool_name: str, **arguments: Any) -> ToolResult:
        call_id = len([event for event in self.trace if event["stage"] == "tool_call"]) + 1
        self.trace.append(
            {"stage": "tool_call", "call_id": call_id, "tool": tool_name, "arguments": arguments}
        )
        injected = self._before_call(tool_name)
        fault_applied = injected is not None
        if injected is None:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                result = ToolResult(False, {}, error="unknown_tool", retryable=False)
            else:
                result = handler(**arguments)
            result, transformed = self._after_call(tool_name, result)
            fault_applied = fault_applied or transformed
        else:
            result = injected
        self.trace.append(
            {
                "stage": "tool_result",
                "call_id": call_id,
                "tool": tool_name,
                "fault": self.fault.mode if fault_applied else "none",
                "result": result.to_dict(),
            }
        )
        return result

    def _before_call(self, tool_name: str) -> ToolResult | None:
        if not self.fault.targets(tool_name):
            return None
        if self.fault.mode == "timeout_once" and self.fault.activations == 0:
            self.fault.activations += 1
            return ToolResult(False, {}, error="timeout", retryable=True)
        if self.fault.mode == "permission_denied":
            self.fault.activations += 1
            return ToolResult(False, {}, error="permission_denied", retryable=False)
        return None

    def _after_call(self, tool_name: str, result: ToolResult) -> tuple[ToolResult, bool]:
        if not result.ok or not self.fault.targets(tool_name):
            return result, False
        if self.fault.mode == "schema_drift":
            payload = dict(result.payload)
            payload["state"] = payload.pop("status")
            payload["version"] = payload.pop("revision")
            self.fault.activations += 1
            return ToolResult(True, payload), True
        if self.fault.mode == "stale_read" and self.fault.activations == 0:
            payload = dict(result.payload)
            payload["revision"] = max(0, int(payload["revision"]) - 1)
            self.fault.activations += 1
            return ToolResult(True, payload), True
        if self.fault.mode == "partial_response" and self.fault.activations == 0:
            payload = dict(result.payload)
            approvals = [dict(item) for item in payload["approvals"]]
            for approval in approvals:
                approval.pop("decision", None)
            payload["approvals"] = approvals
            self.fault.activations += 1
            return ToolResult(True, payload), True
        return result, False

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
        return ToolResult(True, self._row_payload(row))

    def _tool_list_approvals(self, request_id: str) -> ToolResult:
        rows = self.connection.execute(
            "SELECT approver_role, decision FROM approvals WHERE request_id = ? "
            "ORDER BY approver_role",
            (request_id,),
        ).fetchall()
        return ToolResult(True, {"request_id": request_id, "approvals": [dict(row) for row in rows]})

    def _tool_apply_transition(
        self,
        request_id: str,
        actor: str,
        target_status: str,
        expected_revision: int,
    ) -> ToolResult:
        actor_row = self.connection.execute(
            "SELECT can_write FROM actors WHERE actor = ?", (actor,)
        ).fetchone()
        if actor_row is None or not bool(actor_row["can_write"]):
            return ToolResult(False, {}, error="permission_denied", retryable=False)

        try:
            request = self.request_state(request_id)
        except KeyError:
            return ToolResult(False, {}, error="request_not_found", retryable=False)
        if request["status"] == target_status:
            return ToolResult(True, {**request, "mutated": False})
        if request["status"] != "pending" or target_status != "approved":
            return ToolResult(False, {}, error="invalid_transition", retryable=False)
        if int(request["revision"]) != int(expected_revision):
            return ToolResult(False, {}, error="revision_conflict", retryable=True)

        policy = self._tool_get_active_policy(str(request["risk_level"]))
        if not policy.ok:
            return policy
        required_role = policy.payload["required_role"]
        approval = self.connection.execute(
            "SELECT 1 FROM approvals WHERE request_id = ? AND approver_role = ? "
            "AND decision = 'approved'",
            (request_id, required_role),
        ).fetchone()
        if approval is None:
            return ToolResult(False, {}, error="approval_missing", retryable=False)

        new_revision = int(request["revision"]) + 1
        cursor = self.connection.execute(
            "UPDATE requests SET status = ?, revision = ? "
            "WHERE request_id = ? AND revision = ?",
            (target_status, new_revision, request_id, expected_revision),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            return ToolResult(False, {}, error="revision_conflict", retryable=True)
        self.connection.execute(
            "INSERT INTO mutation_log "
            "(request_id, actor, from_status, to_status, from_revision, to_revision) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                request_id,
                actor,
                request["status"],
                target_status,
                expected_revision,
                new_revision,
            ),
        )
        self.connection.commit()
        return ToolResult(
            True,
            {
                **request,
                "status": target_status,
                "revision": new_revision,
                "mutated": True,
            },
        )
