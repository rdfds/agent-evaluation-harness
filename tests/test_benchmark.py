from __future__ import annotations

from copy import deepcopy
import unittest

from agent_eval_harness.benchmark import build_benchmark, deterministic_projection, render_markdown


class BenchmarkArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = build_benchmark(repeats=2)

    def test_artifact_records_comparison_and_mutation_coverage(self):
        self.assertEqual(self.artifact["dataset"]["cases"], 80)
        self.assertEqual(self.artifact["dataset"]["independent_case_units"], 80)
        self.assertEqual(self.artifact["dataset"]["candidate_observations"], 160)
        self.assertGreater(self.artifact["comparison"]["pass_rate"], 0.0)
        self.assertEqual(self.artifact["mutation_testing"]["caught"], 4)
        self.assertEqual(self.artifact["mutation_testing"]["total"], 4)

    def test_deterministic_projection_excludes_machine_latency(self):
        changed = deepcopy(self.artifact)
        changed["candidate"]["operations"]["p95_latency_ms"] = 999.0
        changed["baseline"]["operations"]["p50_latency_ms"] = 999.0
        self.assertEqual(
            deterministic_projection(self.artifact), deterministic_projection(changed)
        )

    def test_markdown_includes_core_results(self):
        markdown = render_markdown(self.artifact)
        self.assertIn("80 cases", markdown)
        self.assertIn("case-cluster bootstrap", markdown)
        self.assertIn("Mutation testing", markdown)
        self.assertIn("Unsupported-answer rate", markdown)


if __name__ == "__main__":
    unittest.main()
