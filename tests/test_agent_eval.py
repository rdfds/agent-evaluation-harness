import unittest

from src.agent_eval import CASES, evaluate, route_query, run_agent


class AgentEvaluationTests(unittest.TestCase):
    def test_routes_counting_queries_to_structured_query(self):
        self.assertEqual(route_query("How many open findings are there?"), "structured_query")
        self.assertEqual(route_query("Find notes about an audit"), "retrieval")

    def test_trace_contains_decision_tool_and_result(self):
        result = run_agent(CASES[0])
        self.assertEqual([event.kind for event in result.trace], ["decision", "tool", "result"])

    def test_failure_taxonomy_keeps_ambiguous_case_visible(self):
        report = evaluate(repeats=2)
        self.assertIn("route_mismatch", report["failures"])
        self.assertLess(report["pass_rate"], 1.0)

    def test_repeated_runs_are_deterministic(self):
        self.assertEqual(evaluate(repeats=3), evaluate(repeats=3))


if __name__ == "__main__":
    unittest.main()
