from __future__ import annotations

import unittest

from agent_eval_harness.workflow_agents import LegacyWorkflowAgent, RecoveryWorkflowAgent
from agent_eval_harness.workflow_dataset import load_workflow_cases
from agent_eval_harness.workflow_env import WorkflowEnvironment


class WorkflowAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_workflow_cases()
        cls.ready = next(case for case in cls.cases if case.request_id == "ready-01")

    def test_recovery_agent_matches_all_clean_terminal_outcomes(self):
        agent = RecoveryWorkflowAgent()
        for case in self.cases:
            with self.subTest(case=case.case_id), WorkflowEnvironment() as environment:
                run = agent.run(case, environment)
                self.assertEqual(run.outcome, case.expected_outcome)
                self.assertEqual(run.final_status, case.expected_final_status)
                self.assertLessEqual(run.mutation_count, 1)

    def test_recovery_agent_handles_transient_and_contract_faults(self):
        agent = RecoveryWorkflowAgent()
        for fault in ["timeout_once", "schema_drift", "stale_read", "partial_response"]:
            with self.subTest(fault=fault), WorkflowEnvironment(fault=fault) as environment:
                run = agent.run(self.ready, environment)
                self.assertEqual(run.outcome, "applied")
                self.assertEqual(run.final_status, "approved")
                self.assertEqual(run.mutation_count, 1)

    def test_permission_fault_fails_closed_without_mutation(self):
        with WorkflowEnvironment(fault="permission_denied") as environment:
            run = RecoveryWorkflowAgent().run(self.ready, environment)
            self.assertEqual(run.outcome, "blocked")
            self.assertEqual(run.final_status, "pending")
            self.assertEqual(run.mutation_count, 0)

    def test_legacy_agent_has_no_policy_check_or_retry(self):
        with WorkflowEnvironment() as clean_environment:
            run = LegacyWorkflowAgent().run(self.ready, clean_environment)
            tools = [event["tool"] for event in run.trace if event["stage"] == "tool_call"]
            self.assertNotIn("get_active_policy", tools)
            self.assertNotIn("list_approvals", tools)
        with WorkflowEnvironment(fault="stale_read") as stale_environment:
            stale_run = LegacyWorkflowAgent().run(self.ready, stale_environment)
            self.assertEqual(stale_run.outcome, "failed")
            self.assertEqual(stale_run.retries, 0)

    def test_trace_records_faults_retries_and_agent_decisions(self):
        with WorkflowEnvironment(fault="timeout_once") as environment:
            run = RecoveryWorkflowAgent().run(self.ready, environment)
            self.assertEqual(run.retries, 1)
            self.assertTrue(any(event.get("fault") == "timeout_once" for event in run.trace))
            self.assertTrue(
                any(event.get("decision") == "retry" for event in run.trace)
            )


if __name__ == "__main__":
    unittest.main()
