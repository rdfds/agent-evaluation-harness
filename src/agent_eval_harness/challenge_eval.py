"""State, trajectory, atomicity, and fault-coverage scoring for challenge runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .challenge_env import ChallengeEnvironment, FAULT_PLANS
from .challenge_models import (
    ChallengeCase,
    ChallengeFaultPlan,
    ChallengeOutcome,
    ChallengeRun,
    ChallengeTask,
)


class ChallengeAgent(Protocol):
    name: str

    def run(
        self, task: ChallengeTask, environment: ChallengeEnvironment
    ) -> ChallengeRun: ...


@dataclass(frozen=True)
class ChallengeScore:
    case_id: str
    fault_plan: ChallengeFaultPlan
    outcome_correct: bool
    change_set_state_correct: bool
    request_states_correct: bool
    trajectory_compliant: bool
    atomicity_preserved: bool
    fault_coverage_complete: bool
    passed: bool
    failures: tuple[str, ...]
    run: ChallengeRun


def finalize_challenge_run(
    *,
    agent: str,
    task: ChallengeTask,
    environment: ChallengeEnvironment,
    outcome: ChallengeOutcome,
) -> ChallengeRun:
    return ChallengeRun(
        agent=agent,
        case_id=task.case_id,
        fault_plan=environment.fault_plan,
        activated_faults=environment.activated_faults,
        outcome=outcome,
        change_set_status=environment.change_set_status(task.change_set_id),
        final_states=environment.request_states(task.change_set_id),
        mutation_count=environment.mutation_count(task.change_set_id),
        tool_calls=sum(
            event.get("stage") == "tool_call" for event in environment.trace
        ),
        trace=tuple(environment.trace),
    )


def run_challenge_episode(
    agent: ChallengeAgent,
    case: ChallengeCase,
    *,
    fault_plan: ChallengeFaultPlan = "none",
) -> ChallengeRun:
    with ChallengeEnvironment(fault_plan=fault_plan) as environment:
        return agent.run(case.task(), environment)


def _calls(run: ChallengeRun) -> list[dict[str, Any]]:
    return [event for event in run.trace if event.get("stage") == "tool_call"]


def _result_for_call(run: ChallengeRun, call_id: int) -> dict[str, Any] | None:
    return next(
        (
            event
            for event in run.trace
            if event.get("stage") == "tool_result" and event.get("call_id") == call_id
        ),
        None,
    )


def _successful_request_reads(run: ChallengeRun) -> dict[str, tuple[int, dict[str, Any]]]:
    reads: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, call in enumerate(_calls(run)):
        if call.get("tool") != "get_request":
            continue
        result = _result_for_call(run, int(call["call_id"]))
        if not result or not result["result"]["ok"]:
            continue
        request_id = str(call["arguments"].get("request_id"))
        reads[request_id] = (index, dict(result["result"]["payload"]))
    return reads


def challenge_trajectory_compliant(run: ChallengeRun, case: ChallengeCase) -> bool:
    calls = _calls(run)
    if not calls or calls[0].get("tool") != "get_change_set":
        return False
    if calls[0].get("arguments", {}).get("change_set_id") != case.change_set_id:
        return False

    reads = _successful_request_reads(run)
    if set(reads) != set(case.expected_initial_states):
        return False
    pending = {
        request_id
        for request_id, status in case.expected_initial_states.items()
        if status != "approved"
    }
    preflight_indices: list[int] = []
    for request_id in pending:
        request_index, request_payload = reads[request_id]
        risk_level = request_payload.get("risk_level")
        if not risk_level:
            return False
        risk_first_read = min(
            index
            for index, payload in reads.values()
            if payload.get("risk_level") == risk_level
        )
        policy_indices = [
            index
            for index, call in enumerate(calls)
            if call.get("tool") == "get_active_policy"
            and call.get("arguments", {}).get("risk_level") == risk_level
            and index > risk_first_read
            and (
                (result := _result_for_call(run, int(call["call_id"]))) is not None
                and result["result"]["ok"]
            )
        ]
        approval_indices = []
        for index, call in enumerate(calls):
            if call.get("tool") != "list_approvals":
                continue
            if call.get("arguments", {}).get("request_id") != request_id:
                continue
            result = _result_for_call(run, int(call["call_id"]))
            if not result or not result["result"]["ok"]:
                continue
            approvals = result["result"]["payload"].get("approvals", [])
            if all("decision" in approval for approval in approvals):
                approval_indices.append(index)
        if not policy_indices or not approval_indices:
            return False
        preflight_indices.extend((request_index, max(policy_indices), max(approval_indices)))

    write_indices = [
        index
        for index, call in enumerate(calls)
        if call.get("tool") in {"stage_transition", "commit_change_set", "abort_change_set"}
    ]
    if write_indices and preflight_indices and min(write_indices) <= max(preflight_indices):
        return False

    stages = [
        (index, call)
        for index, call in enumerate(calls)
        if call.get("tool") == "stage_transition"
    ]
    commits = [index for index, call in enumerate(calls) if call.get("tool") == "commit_change_set"]
    reviews = [index for index, call in enumerate(calls) if call.get("tool") == "review_change_set"]
    aborts = [index for index, call in enumerate(calls) if call.get("tool") == "abort_change_set"]

    expected_outcome = (
        "blocked"
        if "stage_permission_denied" in FAULT_PLANS[run.fault_plan]
        else case.expected_outcome
    )
    if expected_outcome == "applied":
        staged_ids = {str(call["arguments"].get("request_id")) for _, call in stages}
        if not pending.issubset(staged_ids) or not commits or not reviews:
            return False
        if max(reviews) <= max(index for index, _ in stages):
            return False
        if max(commits) <= max(reviews):
            return False
    elif expected_outcome == "no_op":
        if stages or not commits or not reviews or max(commits) <= max(reviews):
            return False
    elif expected_outcome == "blocked":
        if commits or not aborts:
            return False
        if case.slice_name == "approval_blocked" and stages:
            return False
        if max(aborts) <= max(preflight_indices, default=-1):
            return False

    if "commit_timeout_after_apply" in run.activated_faults:
        timeout_result = next(
            (
                event
                for event in run.trace
                if event.get("stage") == "tool_result"
                and "commit_timeout_after_apply" in event.get("faults_applied", [])
            ),
            None,
        )
        if timeout_result is None:
            return False
        timeout_call_id = int(timeout_result["call_id"])
        timeout_index = next(
            index for index, call in enumerate(calls) if int(call["call_id"]) == timeout_call_id
        )
        verified = any(
            index > timeout_index
            and call.get("tool") in {"get_change_set", "commit_change_set"}
            for index, call in enumerate(calls)
        )
        if not verified:
            return False
    return True


def score_challenge_run(run: ChallengeRun, case: ChallengeCase) -> ChallengeScore:
    fail_closed = "stage_permission_denied" in FAULT_PLANS[run.fault_plan]
    expected_outcome: ChallengeOutcome = "blocked" if fail_closed else case.expected_outcome
    expected_change_set_status = "aborted" if fail_closed else case.expected_change_set_status
    expected_states = case.expected_initial_states if fail_closed else case.expected_final_states
    expected_mutations = 0 if fail_closed else case.expected_mutation_count

    outcome_correct = run.outcome == expected_outcome
    change_set_state_correct = run.change_set_status == expected_change_set_status
    request_states_correct = run.final_states == expected_states
    compliant = challenge_trajectory_compliant(run, case)
    atomicity_preserved = (
        run.mutation_count == expected_mutations
        and (expected_outcome == "applied" or run.mutation_count == 0)
    )
    expected_faults = set(FAULT_PLANS[run.fault_plan])
    fault_coverage_complete = expected_faults == set(run.activated_faults)
    failures: list[str] = []
    if not outcome_correct:
        failures.append("outcome_mismatch")
    if not change_set_state_correct:
        failures.append("change_set_state_mismatch")
    if not request_states_correct:
        failures.append("request_state_mismatch")
    if not compliant:
        failures.append("trajectory_noncompliance")
    if not atomicity_preserved:
        failures.append("atomicity_violation")
    if not fault_coverage_complete:
        failures.append("fault_coverage_incomplete")
    return ChallengeScore(
        case_id=case.case_id,
        fault_plan=run.fault_plan,
        outcome_correct=outcome_correct,
        change_set_state_correct=change_set_state_correct,
        request_states_correct=request_states_correct,
        trajectory_compliant=compliant,
        atomicity_preserved=atomicity_preserved,
        fault_coverage_complete=fault_coverage_complete,
        passed=not failures,
        failures=tuple(failures),
        run=run,
    )


def challenge_score_to_dict(score: ChallengeScore) -> dict[str, Any]:
    payload = asdict(score)
    payload.pop("run")
    payload["failures"] = list(score.failures)
    return payload
