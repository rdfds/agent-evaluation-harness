from __future__ import annotations

import unittest

from agent_eval_harness.challenge_dataset import load_challenge_cases
from agent_eval_harness.challenge_env import ChallengeEnvironment
from agent_eval_harness.challenge_eval import finalize_challenge_run, score_challenge_run


class ChallengeScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.change_set_id: case for case in load_challenge_cases()}

    @staticmethod
    def _preflight(environment: ChallengeEnvironment, case):
        change_set = environment.call(
            "get_change_set", change_set_id=case.change_set_id
        )
        records = []
        seen_risks = set()
        for item in change_set.payload["items"]:
            request = environment.call("get_request", request_id=item["request_id"])
            payload = request.payload
            records.append((item, payload))
            if payload["status"] == item["target_status"]:
                continue
            if payload["risk_level"] not in seen_risks:
                environment.call(
                    "get_active_policy", risk_level=payload["risk_level"]
                )
                seen_risks.add(payload["risk_level"])
            environment.call("list_approvals", request_id=item["request_id"])
        return records

    @staticmethod
    def _stage(environment: ChallengeEnvironment, case, records):
        for item, payload in records:
            if payload["status"] == item["target_status"]:
                continue
            environment.call(
                "stage_transition",
                change_set_id=case.change_set_id,
                request_id=item["request_id"],
                actor=case.actor,
                target_status=item["target_status"],
                expected_revision=payload["revision"],
            )

    def test_full_state_and_preflight_trajectory_pass(self):
        case = self.cases["long-horizon-5"]
        with ChallengeEnvironment() as environment:
            records = self._preflight(environment, case)
            self._stage(environment, case, records)
            environment.call("review_change_set", change_set_id=case.change_set_id)
            environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            for request_id in case.expected_final_states:
                environment.call("get_request", request_id=request_id)
            environment.call("get_change_set", change_set_id=case.change_set_id)
            run = finalize_challenge_run(
                agent="test", task=case.task(), environment=environment, outcome="applied"
            )
        score = score_challenge_run(run, case)
        self.assertTrue(score.passed)
        self.assertTrue(score.trajectory_compliant)
        self.assertTrue(score.atomicity_preserved)

    def test_correct_terminal_state_does_not_hide_early_write(self):
        case = self.cases["atomic-ready-3"]
        with ChallengeEnvironment() as environment:
            change_set = environment.call(
                "get_change_set", change_set_id=case.change_set_id
            )
            for item in change_set.payload["items"]:
                request = environment.call("get_request", request_id=item["request_id"])
                environment.call(
                    "get_active_policy", risk_level=request.payload["risk_level"]
                )
                environment.call("list_approvals", request_id=item["request_id"])
                environment.call(
                    "stage_transition",
                    change_set_id=case.change_set_id,
                    request_id=item["request_id"],
                    actor=case.actor,
                    target_status=item["target_status"],
                    expected_revision=request.payload["revision"],
                )
            environment.call("review_change_set", change_set_id=case.change_set_id)
            environment.call(
                "commit_change_set", change_set_id=case.change_set_id, actor=case.actor
            )
            run = finalize_challenge_run(
                agent="unsafe", task=case.task(), environment=environment, outcome="applied"
            )
        score = score_challenge_run(run, case)
        self.assertTrue(score.request_states_correct)
        self.assertTrue(score.atomicity_preserved)
        self.assertFalse(score.trajectory_compliant)
        self.assertFalse(score.passed)

    def test_approval_blocked_case_requires_full_preflight_then_abort(self):
        case = self.cases["approval-rejected-4"]
        with ChallengeEnvironment() as environment:
            self._preflight(environment, case)
            environment.call(
                "abort_change_set",
                change_set_id=case.change_set_id,
                actor=case.actor,
                reason="approval evidence is insufficient",
            )
            run = finalize_challenge_run(
                agent="test", task=case.task(), environment=environment, outcome="blocked"
            )
        self.assertTrue(score_challenge_run(run, case).passed)

    def test_scheduled_faults_must_reach_their_tool_boundaries(self):
        case = self.cases["atomic-ready-3"]
        with ChallengeEnvironment(fault_plan="transient_chain") as environment:
            environment.call("get_change_set", change_set_id=case.change_set_id)
            environment.call(
                "abort_change_set",
                change_set_id=case.change_set_id,
                actor=case.actor,
                reason="gave up",
            )
            run = finalize_challenge_run(
                agent="test", task=case.task(), environment=environment, outcome="blocked"
            )
        score = score_challenge_run(run, case)
        self.assertFalse(score.fault_coverage_complete)
        self.assertIn("fault_coverage_incomplete", score.failures)


if __name__ == "__main__":
    unittest.main()
