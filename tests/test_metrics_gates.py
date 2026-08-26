from __future__ import annotations

import unittest

from agent_eval_harness import CandidateAgent, KnowledgeBase, LegacyAgent, load_cases, load_knowledge_base
from agent_eval_harness.faults import FaultInjectingAgent
from agent_eval_harness.gates import check_gates
from agent_eval_harness.metrics import evaluate


class MetricsAndGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.candidate = CandidateAgent(KnowledgeBase(load_knowledge_base()))
        cls.legacy = LegacyAgent(KnowledgeBase(load_knowledge_base()))

    def test_candidate_scores_all_metrics_and_slices(self):
        report = evaluate(self.candidate, self.cases, repeats=2)
        self.assertEqual(report["runs"], 160)
        self.assertGreaterEqual(report["quality"]["pass_rate"], 0.95)
        self.assertEqual(report["reliability"]["run_pass_rate_stddev"], 0.0)
        self.assertEqual(
            set(report["by_slice"]),
            {"retrieval", "structured_query", "abstention", "challenge"},
        )

    def test_candidate_outperforms_legacy_baseline(self):
        candidate = evaluate(self.candidate, self.cases, repeats=1)
        legacy = evaluate(self.legacy, self.cases, repeats=1)
        self.assertGreater(candidate["quality"]["pass_rate"], legacy["quality"]["pass_rate"])
        self.assertLess(
            candidate["quality"]["unsupported_answer_rate"],
            legacy["quality"]["unsupported_answer_rate"],
        )

    def test_candidate_passes_release_gates(self):
        report = evaluate(self.candidate, self.cases, repeats=2)
        self.assertTrue(check_gates(report)["passed"])

    def test_each_seeded_regression_fails_at_least_one_gate(self):
        for mode in ["route_swap", "stale_answer", "drop_evidence", "disable_abstain"]:
            with self.subTest(mode=mode):
                report = evaluate(FaultInjectingAgent(self.candidate, mode), self.cases, repeats=1)
                self.assertFalse(check_gates(report)["passed"])

    def test_latency_cost_and_intervals_are_reported(self):
        report = evaluate(self.candidate, self.cases, repeats=1)
        self.assertGreaterEqual(report["operations"]["p95_latency_ms"], 0.0)
        self.assertGreater(report["operations"]["mean_estimated_cost_usd"], 0.0)
        self.assertEqual(len(report["quality"]["pass_rate_95ci"]), 2)


if __name__ == "__main__":
    unittest.main()

