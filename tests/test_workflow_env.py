from __future__ import annotations

import unittest

from agent_eval_harness.workflow_env import WorkflowEnvironment


class WorkflowEnvironmentTests(unittest.TestCase):
    def test_each_episode_starts_from_fresh_state(self):
        with WorkflowEnvironment() as first:
            result = first.call(
                "apply_transition",
                request_id="ready-01",
                actor="ops_agent",
                target_status="approved",
                expected_revision=3,
            )
            self.assertTrue(result.ok)
            self.assertEqual(first.request_state("ready-01")["status"], "approved")
        with WorkflowEnvironment() as second:
            self.assertEqual(second.request_state("ready-01")["status"], "pending")

    def test_database_enforces_approval_and_permissions(self):
        with WorkflowEnvironment() as environment:
            missing = environment.call(
                "apply_transition",
                request_id="approval-blocked-01",
                actor="ops_agent",
                target_status="approved",
                expected_revision=3,
            )
            denied = environment.call(
                "apply_transition",
                request_id="permission-blocked-01",
                actor="read_only_bot",
                target_status="approved",
                expected_revision=3,
            )
            self.assertEqual(missing.error, "approval_missing")
            self.assertEqual(denied.error, "permission_denied")
            self.assertEqual(environment.mutation_count(), 0)

    def test_active_policy_supersedes_old_approval_rule(self):
        with WorkflowEnvironment() as environment:
            policy = environment.call("get_active_policy", risk_level="high")
            self.assertEqual(policy.payload["version"], 2)
            self.assertEqual(policy.payload["required_role"], "quality_director")

    def test_timeout_is_retryable_and_only_fires_once(self):
        with WorkflowEnvironment(fault="timeout_once") as environment:
            first = environment.call("get_active_policy", risk_level="low")
            second = environment.call("get_active_policy", risk_level="low")
            self.assertFalse(first.ok)
            self.assertEqual(first.error, "timeout")
            self.assertTrue(first.retryable)
            self.assertTrue(second.ok)

    def test_schema_and_partial_response_faults_modify_real_tool_results(self):
        with WorkflowEnvironment(fault="schema_drift") as schema_environment:
            result = schema_environment.call("get_request", request_id="ready-01")
            self.assertIn("state", result.payload)
            self.assertNotIn("status", result.payload)
        with WorkflowEnvironment(fault="partial_response") as partial_environment:
            first = partial_environment.call("list_approvals", request_id="ready-01")
            second = partial_environment.call("list_approvals", request_id="ready-01")
            self.assertNotIn("decision", first.payload["approvals"][0])
            self.assertIn("decision", second.payload["approvals"][0])

    def test_stale_read_causes_a_real_revision_conflict(self):
        with WorkflowEnvironment(fault="stale_read") as environment:
            stale = environment.call("get_request", request_id="ready-01")
            applied = environment.call(
                "apply_transition",
                request_id="ready-01",
                actor="ops_agent",
                target_status="approved",
                expected_revision=stale.payload["revision"],
            )
            self.assertEqual(stale.payload["revision"], 2)
            self.assertEqual(applied.error, "revision_conflict")
            self.assertEqual(environment.mutation_count(), 0)


if __name__ == "__main__":
    unittest.main()
