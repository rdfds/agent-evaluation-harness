from __future__ import annotations

import unittest

from agent_eval_harness.workflow_agents import LegacyWorkflowAgent, RecoveryWorkflowAgent
from agent_eval_harness.workflow_dataset import load_workflow_cases
from agent_eval_harness.workflow_eval import run_episode
from agent_eval_harness.workflow_replay import (
    attribute_counterfactual,
    evaluate_attribution,
    first_divergent_tool,
    strict_replay,
)


class WorkflowReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_workflow_cases()
        cls.ready = next(case for case in cls.cases if case.request_id == "ready-01")

    def test_strict_replay_reproduces_complete_episode_digest(self):
        agent = RecoveryWorkflowAgent()
        recorded = run_episode(agent, self.ready, fault="stale_read")
        check = strict_replay(agent, self.ready, recorded)
        self.assertTrue(check.matched)
        self.assertEqual(check.recorded_digest, check.replayed_digest)

    def test_first_divergence_localizes_faulted_tool_without_reading_label(self):
        agent = RecoveryWorkflowAgent()
        clean = run_episode(agent, self.ready)
        stale = run_episode(agent, self.ready, fault="stale_read")
        denied = run_episode(agent, self.ready, fault="permission_denied")
        self.assertEqual(first_divergent_tool(stale, clean), "get_request")
        self.assertEqual(first_divergent_tool(denied, clean), "apply_transition")

    def test_counterfactual_attribution_matches_injected_component(self):
        for fault, tool in [
            ("timeout_once", "get_active_policy"),
            ("schema_drift", "get_request"),
            ("stale_read", "get_request"),
            ("partial_response", "list_approvals"),
            ("permission_denied", "apply_transition"),
        ]:
            with self.subTest(fault=fault):
                record = attribute_counterfactual(
                    RecoveryWorkflowAgent(), self.ready, fault
                )
                self.assertTrue(record.fault_activated)
                self.assertEqual(record.attributed_tool, tool)
                self.assertTrue(record.localized)

    def test_attribution_summary_excludes_faults_the_legacy_path_never_exercises(self):
        summary = evaluate_attribution(LegacyWorkflowAgent(), self.cases)
        self.assertEqual(summary["probes"], 60)
        self.assertEqual(summary["activated_probes"], 36)
        self.assertEqual(summary["localization_accuracy"], 1.0)
        self.assertEqual(summary["by_fault"]["timeout_once"]["activated"], 0)
        self.assertEqual(summary["by_fault"]["partial_response"]["activated"], 0)


if __name__ == "__main__":
    unittest.main()
