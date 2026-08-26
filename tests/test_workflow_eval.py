from __future__ import annotations

import unittest

from agent_eval_harness.workflow_agents import LegacyWorkflowAgent, RecoveryWorkflowAgent
from agent_eval_harness.workflow_dataset import load_workflow_cases
from agent_eval_harness.workflow_eval import (
    evaluate_workflow_agent,
    run_episode,
    score_clean_run,
    score_fault_run,
)


class WorkflowEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_workflow_cases()
        cls.ready = next(case for case in cls.cases if case.request_id == "ready-01")

    def test_clean_score_requires_trajectory_and_terminal_state(self):
        legacy_run = run_episode(LegacyWorkflowAgent(), self.ready)
        recovery_run = run_episode(RecoveryWorkflowAgent(), self.ready)
        legacy_score = score_clean_run(legacy_run, self.ready)
        recovery_score = score_clean_run(recovery_run, self.ready)
        self.assertTrue(legacy_score.terminal_state_correct)
        self.assertFalse(legacy_score.trajectory_compliant)
        self.assertFalse(legacy_score.passed)
        self.assertTrue(recovery_score.passed)

    def test_fault_score_distinguishes_recovery_from_safe_blocking(self):
        recovered = score_fault_run(
            run_episode(RecoveryWorkflowAgent(), self.ready, fault="stale_read"),
            self.ready,
        )
        blocked = score_fault_run(
            run_episode(RecoveryWorkflowAgent(), self.ready, fault="permission_denied"),
            self.ready,
        )
        self.assertTrue(recovered.passed)
        self.assertTrue(blocked.passed)
        self.assertEqual(recovered.run.outcome, "applied")
        self.assertEqual(blocked.run.outcome, "blocked")

    def test_recovery_agent_passes_clean_and_fault_suites(self):
        report, clean_scores, fault_scores = evaluate_workflow_agent(
            RecoveryWorkflowAgent(), self.cases
        )
        self.assertEqual(len(clean_scores), 24)
        self.assertEqual(len(fault_scores), 60)
        self.assertEqual(report["clean"]["pass_rate"], 1.0)
        self.assertEqual(report["faults"]["fault_handling_rate"], 1.0)
        self.assertEqual(report["faults"]["terminal_recovery_rate"], 1.0)
        self.assertEqual(report["faults"]["fail_closed_rate"], 1.0)

    def test_legacy_agent_terminal_success_does_not_hide_shortcuts(self):
        report, _, _ = evaluate_workflow_agent(LegacyWorkflowAgent(), self.cases)
        self.assertEqual(report["clean"]["outcome_accuracy"], 1.0)
        self.assertLess(report["clean"]["trajectory_compliance_rate"], 0.5)
        self.assertLess(report["faults"]["fault_handling_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
