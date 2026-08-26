from __future__ import annotations

import unittest

from agent_eval_harness.workflow_benchmark import (
    build_workflow_benchmark,
    render_workflow_markdown,
)


class WorkflowBenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = build_workflow_benchmark()

    def test_artifact_records_state_fault_and_attribution_results(self):
        self.assertEqual(self.artifact["dataset"]["clean_cases"], 24)
        self.assertEqual(self.artifact["dataset"]["fault_episodes_per_agent"], 60)
        self.assertEqual(self.artifact["candidate"]["clean"]["pass_rate"], 1.0)
        self.assertEqual(
            self.artifact["candidate"]["faults"]["fault_handling_rate"], 1.0
        )
        self.assertEqual(
            self.artifact["attribution"]["candidate"]["localization_accuracy"],
            1.0,
        )
        self.assertTrue(self.artifact["release_gates"]["passed"])

    def test_artifact_exposes_outcome_only_baseline_shortcut(self):
        baseline = self.artifact["baseline"]["clean"]
        self.assertEqual(baseline["terminal_state_accuracy"], 1.0)
        self.assertLess(baseline["trajectory_compliance_rate"], 0.5)

    def test_markdown_contains_reproducible_core_results(self):
        markdown = render_workflow_markdown(self.artifact)
        self.assertIn("24 clean stateful tasks", markdown)
        self.assertIn("60 fault episodes per agent", markdown)
        self.assertIn("counterfactual localization", markdown)
        self.assertIn("do not measure a hosted language model", markdown)


if __name__ == "__main__":
    unittest.main()
