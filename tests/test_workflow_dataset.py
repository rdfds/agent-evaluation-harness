from __future__ import annotations

import unittest

from agent_eval_harness.workflow_dataset import load_workflow_cases, load_workflow_seed


class WorkflowDatasetTests(unittest.TestCase):
    def test_fixture_has_twenty_four_unique_cases(self):
        cases = load_workflow_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case.case_id for case in cases}), 24)

    def test_fixture_covers_decisions_and_stateful_slices(self):
        cases = load_workflow_cases()
        self.assertEqual(
            {case.expected_outcome for case in cases},
            {"applied", "blocked", "no_op"},
        )
        self.assertEqual(
            {case.slice_name for case in cases},
            {
                "ready",
                "approval_blocked",
                "permission_blocked",
                "idempotent",
                "policy_version",
                "revision_sensitive",
            },
        )

    def test_every_case_references_a_seeded_request_and_actor(self):
        cases = load_workflow_cases()
        seed = load_workflow_seed()
        request_ids = {item["request_id"] for item in seed["requests"]}
        actors = {item["actor"] for item in seed["actors"]}
        self.assertTrue(all(case.request_id in request_ids for case in cases))
        self.assertTrue(all(case.actor in actors for case in cases))

    def test_seed_retains_active_and_superseded_policy_versions(self):
        policies = load_workflow_seed()["policies"]
        high_risk_versions = [
            item for item in policies if item["risk_level"] == "high"
        ]
        self.assertEqual({item["version"] for item in high_risk_versions}, {1, 2})
        self.assertEqual(sum(item["active"] for item in high_risk_versions), 1)

    def test_agent_task_does_not_expose_grading_labels(self):
        task = load_workflow_cases()[0].task()
        self.assertFalse(hasattr(task, "expected_outcome"))
        self.assertFalse(hasattr(task, "expected_final_status"))
        self.assertFalse(hasattr(task, "slice_name"))


if __name__ == "__main__":
    unittest.main()
