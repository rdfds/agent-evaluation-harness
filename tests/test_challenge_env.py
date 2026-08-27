from __future__ import annotations

import unittest

from agent_eval_harness.challenge_dataset import load_challenge_cases
from agent_eval_harness.challenge_env import ChallengeEnvironment, FAULT_PLANS


class ChallengeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.change_set_id: case for case in load_challenge_cases()}

    @staticmethod
    def _stage_all(environment: ChallengeEnvironment, change_set_id: str, actor: str) -> None:
        change_set = environment.call("get_change_set", change_set_id=change_set_id)
        for item in change_set.payload["items"]:
            request = environment.call("get_request", request_id=item["request_id"])
            payload = request.payload
            status = payload.get("status", payload.get("state"))
            revision = payload.get("revision", payload.get("version"))
            if status == item["target_status"]:
                continue
            environment.call("get_active_policy", risk_level=payload["risk_level"])
            environment.call("list_approvals", request_id=item["request_id"])
            staged = environment.call(
                "stage_transition",
                change_set_id=change_set_id,
                request_id=item["request_id"],
                actor=actor,
                target_status=item["target_status"],
                expected_revision=revision,
            )
            if staged.error == "revision_conflict":
                refreshed = environment.call("get_request", request_id=item["request_id"])
                refreshed_revision = refreshed.payload.get(
                    "revision", refreshed.payload.get("version")
                )
                staged = environment.call(
                    "stage_transition",
                    change_set_id=change_set_id,
                    request_id=item["request_id"],
                    actor=actor,
                    target_status=item["target_status"],
                    expected_revision=refreshed_revision,
                )
            self_or_error = staged.error
            if self_or_error:
                raise AssertionError(self_or_error)

    def test_dataset_has_long_mixed_and_blocked_slices(self):
        cases = load_challenge_cases()
        self.assertEqual(len(cases), 10)
        self.assertEqual(sum(case.expected_outcome == "applied" for case in cases), 6)
        self.assertTrue(any(len(case.expected_final_states) == 5 for case in cases))
        task = cases[0].task()
        self.assertFalse(hasattr(task, "expected_outcome"))
        self.assertFalse(hasattr(task, "expected_final_states"))

    def test_atomic_commit_applies_every_staged_request_once(self):
        case = self.cases["atomic-ready-3"]
        with ChallengeEnvironment() as environment:
            self._stage_all(environment, case.change_set_id, case.actor)
            review = environment.call("review_change_set", change_set_id=case.change_set_id)
            self.assertTrue(review.payload["ready"])
            committed = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            repeated = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            self.assertEqual(committed.payload["mutations"], 3)
            self.assertTrue(repeated.payload["idempotent"])
            self.assertEqual(environment.mutation_count(case.change_set_id), 3)
            self.assertEqual(
                environment.request_states(case.change_set_id), case.expected_final_states
            )

    def test_failed_preflight_cannot_partially_commit(self):
        case = self.cases["approval-missing-3"]
        with ChallengeEnvironment() as environment:
            change_set = environment.call(
                "get_change_set", change_set_id=case.change_set_id
            )
            for item in change_set.payload["items"]:
                request = environment.call("get_request", request_id=item["request_id"])
                if request.payload["status"] == item["target_status"]:
                    continue
                environment.call(
                    "stage_transition",
                    change_set_id=case.change_set_id,
                    request_id=item["request_id"],
                    actor=case.actor,
                    target_status=item["target_status"],
                    expected_revision=request.payload["revision"],
                )
            commit = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            self.assertFalse(commit.ok)
            self.assertEqual(commit.error, "validation_failed")
            self.assertEqual(environment.mutation_count(case.change_set_id), 0)
            self.assertEqual(
                environment.request_states(case.change_set_id), case.expected_initial_states
            )

    def test_idempotent_change_set_closes_without_request_mutation(self):
        case = self.cases["already-satisfied-3"]
        with ChallengeEnvironment() as environment:
            review = environment.call("review_change_set", change_set_id=case.change_set_id)
            self.assertTrue(review.payload["ready"])
            committed = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            self.assertEqual(committed.payload["status"], "satisfied")
            self.assertEqual(environment.mutation_count(case.change_set_id), 0)

    def test_composed_fault_plans_activate_each_component(self):
        self.assertEqual(len(FAULT_PLANS), 6)

        with ChallengeEnvironment(fault_plan="transient_chain") as environment:
            first_policy = environment.call("get_active_policy", risk_level="low")
            second_policy = environment.call("get_active_policy", risk_level="low")
            first_approval = environment.call(
                "list_approvals", request_id="atomic-ready-3-r1"
            )
            second_approval = environment.call(
                "list_approvals", request_id="atomic-ready-3-r1"
            )
            self.assertEqual(first_policy.error, "timeout")
            self.assertTrue(second_policy.ok)
            self.assertEqual(first_approval.error, "rate_limited")
            self.assertTrue(second_approval.ok)
            self.assertEqual(
                set(environment.activated_faults), set(FAULT_PLANS["transient_chain"])
            )

        with ChallengeEnvironment(fault_plan="evidence_drift") as environment:
            request = environment.call("get_request", request_id="atomic-ready-3-r1")
            approvals = environment.call(
                "list_approvals", request_id="atomic-ready-3-r1"
            )
            self.assertIn("state", request.payload)
            self.assertNotIn("status", request.payload)
            self.assertNotIn("decision", approvals.payload["approvals"][0])
            self.assertEqual(
                set(environment.activated_faults), set(FAULT_PLANS["evidence_drift"])
            )

    def test_concurrency_and_ambiguous_commit_faults_change_real_state(self):
        case = self.cases["atomic-ready-3"]
        with ChallengeEnvironment(fault_plan="concurrency_race") as environment:
            self._stage_all(environment, case.change_set_id, case.actor)
            conflict = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            self.assertEqual(conflict.error, "revision_conflict")
            self.assertEqual(environment.mutation_count(case.change_set_id), 0)
            self.assertEqual(
                set(environment.activated_faults), set(FAULT_PLANS["concurrency_race"])
            )

        with ChallengeEnvironment(fault_plan="ambiguous_commit") as environment:
            first_approval = environment.call(
                "list_approvals", request_id="atomic-ready-3-r1"
            )
            self.assertEqual(first_approval.error, "timeout")
            self._stage_all(environment, case.change_set_id, case.actor)
            timeout = environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            self.assertEqual(timeout.error, "timeout")
            self.assertEqual(environment.change_set_status(case.change_set_id), "committed")
            self.assertEqual(environment.mutation_count(case.change_set_id), 3)
            self.assertEqual(
                set(environment.activated_faults), set(FAULT_PLANS["ambiguous_commit"])
            )

    def test_degraded_authorization_fails_before_staging(self):
        case = self.cases["atomic-ready-3"]
        with ChallengeEnvironment(fault_plan="degraded_auth") as environment:
            request = environment.call("get_request", request_id="atomic-ready-3-r1")
            denied = environment.call(
                "stage_transition",
                change_set_id=case.change_set_id,
                request_id="atomic-ready-3-r1",
                actor=case.actor,
                target_status="approved",
                expected_revision=request.payload["version"],
            )
            self.assertEqual(denied.error, "permission_denied")
            self.assertEqual(environment.mutation_count(case.change_set_id), 0)
            self.assertEqual(
                set(environment.activated_faults), set(FAULT_PLANS["degraded_auth"])
            )


if __name__ == "__main__":
    unittest.main()
