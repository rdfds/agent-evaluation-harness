from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from agent_eval_harness.live_benchmark import (
    DEFAULT_FAULT_REQUEST_IDS,
    build_live_artifact,
    build_live_episode_specs,
    render_live_markdown,
    verify_live_artifact,
)
from agent_eval_harness.workflow_eval import FAULT_SUITE


def _record(model: str, prompt: str, case_id: str, fault: str, passed: bool = True):
    terminal = passed
    return {
        "episode_id": f"{model}::{prompt}::{case_id}::{fault}",
        "requested_model": model,
        "prompt_variant": prompt,
        "reasoning_effort": "low",
        "case_id": case_id,
        "request_id": case_id,
        "slice": "ready",
        "fault": fault,
        "score": {
            "passed": passed,
            "outcome_correct": terminal,
            "terminal_state_correct": terminal,
            "trajectory_compliant": terminal,
            "mutation_safe": True,
            "failures": [] if passed else ["outcome_mismatch"],
        },
        "run": {},
        "provider": {
            "resolved_models": [model],
            "response_calls": 2,
            "provider_latency_ms": 100.0,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 10,
                "output_tokens": 20,
                "reasoning_tokens": 5,
            },
        },
        "episode_latency_ms": 120.0,
    }


class LiveBenchmarkTests(unittest.TestCase):
    def test_default_matrix_has_39_unique_episodes_per_config(self):
        specs = build_live_episode_specs(models=["gpt-5.6-luna"], prompts=["concise"])
        self.assertEqual(len(specs), 39)
        self.assertEqual(len({spec.episode_id for spec in specs}), 39)
        fault_specs = [spec for spec in specs if spec.fault != "none"]
        self.assertEqual(len(fault_specs), len(DEFAULT_FAULT_REQUEST_IDS) * len(FAULT_SUITE))

    def test_artifact_summarizes_prompt_delta_usage_and_cost(self):
        records = []
        for prompt in ("concise", "explicit"):
            records.append(_record("gpt-5.6-luna", prompt, "clean-01", "none"))
            for fault in FAULT_SUITE:
                records.append(_record("gpt-5.6-luna", prompt, "fault-01", fault))
        artifact = build_live_artifact(
            records,
            models=["gpt-5.6-luna"],
            prompts=["concise", "explicit"],
        )
        self.assertEqual(artifact["totals"]["episodes"], 12)
        self.assertEqual(artifact["totals"]["response_calls"], 24)
        self.assertEqual(len(artifact["prompt_deltas_explicit_minus_concise"]), 1)
        self.assertGreater(artifact["totals"]["estimated_cost_usd"], 0)
        markdown = render_live_markdown(artifact)
        self.assertIn("Live model workflow benchmark", markdown)
        self.assertIn("Prompt ablation", markdown)
        self.assertIn("one run per task/fault pair", markdown)


class CommittedLiveArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.artifact = json.loads(
            (root / "results/live_model_benchmark.json").read_text(encoding="utf-8")
        )
        cls.markdown = (root / "results/live_model_benchmark.md").read_text(
            encoding="utf-8"
        )

    def test_committed_artifact_recomputes_without_provider_access(self):
        summary = verify_live_artifact(self.artifact)
        self.assertEqual(summary, {"episodes": 234, "passed": 233, "failed": 1})
        self.assertEqual(render_live_markdown(self.artifact), self.markdown)

    def test_score_tampering_is_detected(self):
        altered = copy.deepcopy(self.artifact)
        altered["episodes"][0]["score"]["passed"] = not altered["episodes"][0][
            "score"
        ]["passed"]
        with self.assertRaisesRegex(ValueError, "Recorded score drift"):
            verify_live_artifact(altered)


if __name__ == "__main__":
    unittest.main()
