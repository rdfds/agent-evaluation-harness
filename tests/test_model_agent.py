from __future__ import annotations

import json
import unittest

from agent_eval_harness.live_client import ResponseCall
from agent_eval_harness.model_agent import ModelWorkflowAgent
from agent_eval_harness.workflow_dataset import load_workflow_cases
from agent_eval_harness.workflow_env import WorkflowEnvironment


class ScriptedResponsesClient:
    def __init__(self, calls):
        self.calls = list(calls)
        self.payloads = []

    def create(self, payload):
        self.payloads.append(payload)
        name, arguments = self.calls.pop(0)
        return ResponseCall(
            {
                "model": "test-model-snapshot",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"call-{len(self.payloads)}",
                        "name": name,
                        "arguments": json.dumps(arguments),
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 2},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 1},
                    "total_tokens": 25,
                },
            },
            latency_ms=12.5,
        )


class ModelWorkflowAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = next(
            case for case in load_workflow_cases() if case.request_id == "ready-01"
        )

    def test_model_agent_drives_real_tools_and_records_provider_usage(self):
        client = ScriptedResponsesClient(
            [
                ("get_request", {"request_id": "ready-01"}),
                ("get_active_policy", {"risk_level": "low"}),
                ("list_approvals", {"request_id": "ready-01"}),
                (
                    "apply_transition",
                    {
                        "request_id": "ready-01",
                        "actor": "ops_agent",
                        "target_status": "approved",
                        "expected_revision": 3,
                    },
                ),
                ("finish", {"outcome": "applied", "reason": "transition committed"}),
            ]
        )
        agent = ModelWorkflowAgent(
            client, model="test-model", prompt_variant="explicit"
        )
        with WorkflowEnvironment() as environment:
            run = agent.run(self.case.task(), environment)
        self.assertEqual(run.outcome, "applied")
        self.assertEqual(run.final_status, "approved")
        self.assertEqual(run.mutation_count, 1)
        self.assertEqual(agent.last_metadata["response_calls"], 5)
        self.assertEqual(agent.last_metadata["usage"]["input_tokens"], 100)
        self.assertEqual(agent.last_metadata["usage"]["output_tokens"], 25)
        self.assertEqual(agent.last_metadata["resolved_models"], ["test-model-snapshot"])

    def test_agent_visible_payload_excludes_grading_labels(self):
        client = ScriptedResponsesClient(
            [("finish", {"outcome": "blocked", "reason": "cannot continue"})]
        )
        agent = ModelWorkflowAgent(
            client, model="test-model", prompt_variant="concise"
        )
        with WorkflowEnvironment() as environment:
            agent.run(self.case.task(), environment)
        encoded = json.dumps(client.payloads[0], sort_keys=True)
        self.assertNotIn("expected_outcome", encoded)
        self.assertNotIn("expected_final_status", encoded)
        self.assertNotIn("slice_name", encoded)
        self.assertIn("request_id=ready-01", encoded)

    def test_invalid_tool_arguments_fail_without_mutation(self):
        client = ScriptedResponsesClient(
            [
                ("apply_transition", {"request_id": "ready-01"}),
                ("finish", {"outcome": "failed", "reason": "invalid call"}),
            ]
        )
        agent = ModelWorkflowAgent(
            client, model="test-model", prompt_variant="concise"
        )
        with WorkflowEnvironment() as environment:
            run = agent.run(self.case.task(), environment)
        self.assertEqual(run.outcome, "failed")
        self.assertEqual(run.mutation_count, 0)
        tool_results = [
            event for event in run.trace if event.get("stage") == "tool_result"
        ]
        self.assertEqual(tool_results[0]["result"]["error"], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
