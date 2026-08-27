from __future__ import annotations

import json
import unittest

from agent_eval_harness.challenge_dataset import load_challenge_cases
from agent_eval_harness.challenge_env import ChallengeEnvironment
from agent_eval_harness.challenge_eval import score_challenge_run
from agent_eval_harness.challenge_model_agent import (
    CHALLENGE_PROMPTS,
    ChallengeModelAgent,
)
from agent_eval_harness.live_client import ResponseCall


class BatchedScriptClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.payloads = []
        self.call_number = 0

    def create(self, payload):
        self.payloads.append(payload)
        calls = self.turns.pop(0)
        output = []
        for name, arguments in calls:
            self.call_number += 1
            output.append(
                {
                    "type": "function_call",
                    "call_id": f"call-{self.call_number}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                }
            )
        return ResponseCall(
            {
                "model": "challenge-test-snapshot",
                "output": output,
                "usage": {
                    "input_tokens": 100,
                    "input_tokens_details": {"cached_tokens": 10},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 4},
                    "total_tokens": 120,
                },
            },
            latency_ms=25.0,
        )


class ChallengeModelAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = next(
            case
            for case in load_challenge_cases()
            if case.change_set_id == "atomic-ready-3"
        )

    def test_model_agent_batches_reads_and_commits_atomically(self):
        change_set_id = self.case.change_set_id
        request_ids = list(self.case.expected_initial_states)
        revisions = [3, 4, 5]
        risks = ["low", "medium", "high"]
        client = BatchedScriptClient(
            [
                [("get_change_set", {"change_set_id": change_set_id})],
                [["get_request", {"request_id": request_id}] for request_id in request_ids],
                [
                    *[["get_active_policy", {"risk_level": risk}] for risk in risks],
                    *[["list_approvals", {"request_id": request_id}] for request_id in request_ids],
                ],
                [
                    [
                        "stage_transition",
                        {
                            "change_set_id": change_set_id,
                            "request_id": request_id,
                            "actor": self.case.actor,
                            "target_status": "approved",
                            "expected_revision": revision,
                        },
                    ]
                    for request_id, revision in zip(request_ids, revisions)
                ],
                [("review_change_set", {"change_set_id": change_set_id})],
                [
                    (
                        "commit_change_set",
                        {"change_set_id": change_set_id, "actor": self.case.actor},
                    )
                ],
                [("finish", {"outcome": "applied", "reason": "atomic commit verified"})],
            ]
        )
        agent = ChallengeModelAgent(
            client, model="test-model", prompt_variant="principled"
        )
        with ChallengeEnvironment() as environment:
            run = agent.run(self.case.task(), environment)
        score = score_challenge_run(run, self.case)
        self.assertTrue(score.passed)
        self.assertEqual(run.mutation_count, 3)
        self.assertEqual(run.change_set_status, "committed")
        self.assertEqual(agent.last_metadata["response_calls"], 7)
        self.assertEqual(agent.last_metadata["usage"]["input_tokens"], 700)
        self.assertTrue(client.payloads[0]["parallel_tool_calls"])

    def test_agent_visible_payload_keeps_grading_contract_hidden(self):
        client = BatchedScriptClient(
            [[("finish", {"outcome": "failed", "reason": "stop"})]]
        )
        agent = ChallengeModelAgent(
            client, model="test-model", prompt_variant="minimal"
        )
        with ChallengeEnvironment() as environment:
            agent.run(self.case.task(), environment)
        encoded = json.dumps(client.payloads[0], sort_keys=True)
        self.assertNotIn("expected_outcome", encoded)
        self.assertNotIn("expected_final_states", encoded)
        self.assertNotIn("slice_name", encoded)
        self.assertIn("change_set_id=atomic-ready-3", encoded)

    def test_principled_prompt_does_not_name_fault_recipes(self):
        prompt = CHALLENGE_PROMPTS["principled"]
        for implementation_label in (
            "policy_timeout_once",
            "approval_partial_once",
            "commit_conflict_once",
            "commit_timeout_after_apply",
            "stage_permission_denied",
        ):
            self.assertNotIn(implementation_label, prompt)
        self.assertIn("preflight every unsatisfied item", prompt)
        self.assertIn("ambiguous", prompt)

    def test_invalid_tool_arguments_are_traced_without_mutation(self):
        client = BatchedScriptClient(
            [
                [("stage_transition", {"change_set_id": self.case.change_set_id})],
                [
                    (
                        "abort_change_set",
                        {
                            "change_set_id": self.case.change_set_id,
                            "actor": self.case.actor,
                            "reason": "invalid call",
                        },
                    )
                ],
                [("finish", {"outcome": "failed", "reason": "invalid call"})],
            ]
        )
        agent = ChallengeModelAgent(
            client, model="test-model", prompt_variant="minimal"
        )
        with ChallengeEnvironment() as environment:
            run = agent.run(self.case.task(), environment)
        self.assertEqual(run.mutation_count, 0)
        errors = [
            event["result"]["error"]
            for event in run.trace
            if event.get("stage") == "tool_result" and event["result"]["error"]
        ]
        self.assertIn("invalid_arguments", errors)


if __name__ == "__main__":
    unittest.main()
