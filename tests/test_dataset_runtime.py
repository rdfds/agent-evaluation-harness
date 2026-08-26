from __future__ import annotations

import unittest

from agent_eval_harness import CandidateAgent, KnowledgeBase, LegacyAgent, load_cases, load_knowledge_base
from agent_eval_harness.runtime import ABSTAIN_ANSWER


class DatasetAndRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.knowledge = KnowledgeBase(load_knowledge_base())

    def test_fixture_has_eighty_unique_cases(self):
        self.assertEqual(len(self.cases), 80)
        self.assertEqual(len({case.case_id for case in self.cases}), 80)

    def test_fixture_covers_required_slices(self):
        self.assertEqual(
            {case.slice_name for case in self.cases},
            {"retrieval", "structured_query", "abstention", "challenge"},
        )

    def test_candidate_retrieval_returns_required_evidence(self):
        case = next(case for case in self.cases if case.case_id == "retrieval-01-1")
        response = CandidateAgent(self.knowledge).answer(case.query)
        self.assertEqual(response.route, case.expected_route)
        self.assertEqual(response.answer, case.expected_answer)
        self.assertTrue(set(case.required_evidence) <= set(response.evidence_ids))

    def test_candidate_structured_query_returns_record_evidence(self):
        case = next(case for case in self.cases if case.case_id == "structured-01-1")
        response = CandidateAgent(self.knowledge).answer(case.query)
        self.assertEqual(response.route, "structured_query")
        self.assertEqual(response.answer, "7")
        self.assertEqual(response.evidence_ids, ("records:findings-v1",))

    def test_candidate_abstains_on_out_of_scope_request(self):
        case = next(case for case in self.cases if case.case_id == "abstain-01-1")
        response = CandidateAgent(self.knowledge).answer(case.query)
        self.assertEqual(response.route, "abstain")
        self.assertEqual(response.answer, ABSTAIN_ANSWER)
        self.assertEqual(response.evidence_ids, ())

    def test_trace_captures_route_and_tool_evidence(self):
        case = next(case for case in self.cases if case.case_id == "retrieval-02-2")
        response = CandidateAgent(self.knowledge).answer(case.query)
        self.assertEqual([event.stage for event in response.trace], ["route", "tool"])
        self.assertIn("reason", response.trace[0].detail)
        self.assertIn("evidence_ids", response.trace[1].detail)

    def test_legacy_agent_has_no_abstention_path(self):
        response = LegacyAgent(self.knowledge).answer("Tell me tomorrow's weather in Boston.")
        self.assertNotEqual(response.route, "abstain")


if __name__ == "__main__":
    unittest.main()

