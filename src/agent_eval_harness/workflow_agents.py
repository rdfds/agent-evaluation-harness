"""Inspectable baseline and recovery-aware agents for stateful workflows."""

from __future__ import annotations

from typing import Any, Protocol

from .workflow_env import WorkflowEnvironment
from .workflow_models import ToolResult, WorkflowCase, WorkflowRun


class WorkflowAgent(Protocol):
    name: str

    def run(self, case: WorkflowCase, environment: WorkflowEnvironment) -> WorkflowRun: ...


def _finalize(
    *,
    agent: str,
    case: WorkflowCase,
    environment: WorkflowEnvironment,
    outcome: str,
    retries: int,
) -> WorkflowRun:
    state = environment.request_state(case.request_id)
    return WorkflowRun(
        agent=agent,
        case_id=case.case_id,
        fault=environment.fault.mode,
        outcome=outcome,  # type: ignore[arg-type]
        final_status=str(state["status"]),
        mutation_count=environment.mutation_count(case.request_id),
        tool_calls=sum(event["stage"] == "tool_call" for event in environment.trace),
        retries=retries,
        trace=tuple(environment.trace),
    )


class LegacyWorkflowAgent:
    """A one-shot executor without policy inspection, normalization, or retries."""

    name = "legacy_one_shot_executor"

    def run(self, case: WorkflowCase, environment: WorkflowEnvironment) -> WorkflowRun:
        request = environment.call("get_request", request_id=case.request_id)
        if not request.ok:
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="failed",
                retries=0,
            )
        if "status" not in request.payload or "revision" not in request.payload:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "unrecognized_request_schema"}
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="failed",
                retries=0,
            )
        if request.payload["status"] == case.target_status:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "already_at_target"}
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="no_op",
                retries=0,
            )
        result = environment.call(
            "apply_transition",
            request_id=case.request_id,
            actor=case.actor,
            target_status=case.target_status,
            expected_revision=int(request.payload["revision"]),
        )
        if result.ok:
            outcome = "applied" if result.payload.get("mutated") else "no_op"
        elif result.retryable:
            outcome = "failed"
        else:
            outcome = "blocked"
        environment.trace.append(
            {"stage": "agent_decision", "decision": outcome, "error": result.error}
        )
        return _finalize(
            agent=self.name,
            case=case,
            environment=environment,
            outcome=outcome,
            retries=0,
        )


class RecoveryWorkflowAgent:
    """Checks policy and approval state, normalizes drift, and retries bounded faults."""

    name = "recovery_aware_executor_v1"
    max_retries = 1

    @staticmethod
    def _normalize_request(payload: dict[str, Any]) -> dict[str, Any] | None:
        status = payload.get("status", payload.get("state"))
        revision = payload.get("revision", payload.get("version"))
        if status is None or revision is None or "risk_level" not in payload:
            return None
        return {**payload, "status": status, "revision": int(revision)}

    def _call_with_retry(
        self,
        environment: WorkflowEnvironment,
        tool_name: str,
        **arguments: Any,
    ) -> tuple[ToolResult, int]:
        result = environment.call(tool_name, **arguments)
        retries = 0
        while not result.ok and result.retryable and retries < self.max_retries:
            retries += 1
            environment.trace.append(
                {
                    "stage": "agent_decision",
                    "decision": "retry",
                    "tool": tool_name,
                    "error": result.error,
                    "attempt": retries + 1,
                }
            )
            result = environment.call(tool_name, **arguments)
        return result, retries

    def run(self, case: WorkflowCase, environment: WorkflowEnvironment) -> WorkflowRun:
        retries = 0
        request_result, used = self._call_with_retry(
            environment, "get_request", request_id=case.request_id
        )
        retries += used
        if not request_result.ok:
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="failed",
                retries=retries,
            )
        request = self._normalize_request(request_result.payload)
        if request is None:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "unsafe_unknown_schema"}
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="blocked",
                retries=retries,
            )
        if request["status"] == case.target_status:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "already_at_target"}
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="no_op",
                retries=retries,
            )

        policy, used = self._call_with_retry(
            environment, "get_active_policy", risk_level=request["risk_level"]
        )
        retries += used
        if not policy.ok:
            environment.trace.append(
                {"stage": "agent_decision", "decision": "block_without_policy"}
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="blocked",
                retries=retries,
            )

        approvals, used = self._call_with_retry(
            environment, "list_approvals", request_id=case.request_id
        )
        retries += used
        if approvals.ok and any(
            "decision" not in item for item in approvals.payload.get("approvals", [])
        ):
            environment.trace.append(
                {
                    "stage": "agent_decision",
                    "decision": "refetch_incomplete_approvals",
                }
            )
            approvals = environment.call("list_approvals", request_id=case.request_id)
            retries += 1
        if not approvals.ok:
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="blocked",
                retries=retries,
            )
        required_role = policy.payload["required_role"]
        has_approval = any(
            item.get("approver_role") == required_role and item.get("decision") == "approved"
            for item in approvals.payload["approvals"]
        )
        if not has_approval:
            environment.trace.append(
                {
                    "stage": "agent_decision",
                    "decision": "required_approval_missing",
                    "required_role": required_role,
                }
            )
            return _finalize(
                agent=self.name,
                case=case,
                environment=environment,
                outcome="blocked",
                retries=retries,
            )

        transition = environment.call(
            "apply_transition",
            request_id=case.request_id,
            actor=case.actor,
            target_status=case.target_status,
            expected_revision=request["revision"],
        )
        if transition.error == "revision_conflict":
            environment.trace.append(
                {"stage": "agent_decision", "decision": "refresh_after_revision_conflict"}
            )
            refreshed = environment.call("get_request", request_id=case.request_id)
            normalized = self._normalize_request(refreshed.payload) if refreshed.ok else None
            retries += 1
            if normalized is not None:
                transition = environment.call(
                    "apply_transition",
                    request_id=case.request_id,
                    actor=case.actor,
                    target_status=case.target_status,
                    expected_revision=normalized["revision"],
                )

        if transition.ok:
            outcome = "applied" if transition.payload.get("mutated") else "no_op"
        elif transition.error in {"permission_denied", "approval_missing", "invalid_transition"}:
            outcome = "blocked"
        else:
            outcome = "failed"
        environment.trace.append(
            {"stage": "agent_decision", "decision": outcome, "error": transition.error}
        )
        return _finalize(
            agent=self.name,
            case=case,
            environment=environment,
            outcome=outcome,
            retries=retries,
        )
