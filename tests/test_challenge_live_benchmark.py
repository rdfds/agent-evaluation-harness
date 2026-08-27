from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent_eval_harness.challenge_live_benchmark import (
    CHALLENGE_FAULT_SUITE,
    append_challenge_checkpoint,
    build_challenge_artifact,
    build_challenge_specs,
    load_challenge_checkpoint,
    render_challenge_markdown,
)


def _record(
    model: str,
    prompt: str,
    case_id: str,
    fault_plan: str,
    trial: int,
    passed: bool,
):
    return {
        "episode_id": f"{model}::{prompt}::{case_id}::{fault_plan}::trial-{trial}",
        "requested_model": model,
        "prompt_variant": prompt,
        "reasoning_effort": "low",
        "trial": trial,
        "case_id": case_id,
        "change_set_id": case_id,
        "slice": "atomic_ready",
        "fault_plan": fault_plan,
        "score": {
            "passed": passed,
            "outcome_correct": passed,
            "change_set_state_correct": passed,
            "request_states_correct": passed,
            "trajectory_compliant": passed,
            "atomicity_preserved": True,
            "fault_coverage_complete": True,
            "failures": [] if passed else ["trajectory_noncompliance"],
        },
        "run": {},
        "provider": {
            "requested_model": model,
            "resolved_models": [model],
            "prompt_variant": prompt,
            "response_calls": 4,
            "provider_latency_ms": 200.0,
            "usage": {
                "input_tokens": 500,
                "cached_input_tokens": 50,
                "output_tokens": 100,
                "reasoning_tokens": 20,
            },
        },
        "episode_latency_ms": 250.0,
    }


class ChallengeLiveBenchmarkTests(unittest.TestCase):
    def test_default_matrix_repeats_every_clean_and_fault_cell(self):
        specs = build_challenge_specs()
        self.assertEqual(len(specs), 360)
        self.assertEqual(len({spec.episode_id for spec in specs}), 360)
        one_config = [
            spec
            for spec in specs
            if spec.model == "gpt-5.6-luna" and spec.prompt_variant == "minimal"
        ]
        self.assertEqual(len(one_config), 60)
        self.assertEqual(
            sum(spec.fault_plan != "none" for spec in one_config),
            2 * len(CHALLENGE_FAULT_SUITE) * 3,
        )

    def test_repeated_summary_exposes_mixed_cells_and_prompt_delta(self):
        records = []
        for prompt in ("minimal", "principled"):
            for trial in range(1, 4):
                records.append(
                    _record(
                        "gpt-5.6-luna",
                        prompt,
                        "clean-01",
                        "none",
                        trial,
                        passed=prompt == "principled" or trial != 2,
                    )
                )
                records.append(
                    _record(
                        "gpt-5.6-luna",
                        prompt,
                        "fault-01",
                        "transient_chain",
                        trial,
                        passed=True,
                    )
                )
        artifact = build_challenge_artifact(
            records,
            models=["gpt-5.6-luna"],
            prompts=["minimal", "principled"],
            repetitions=3,
            fault_change_set_ids=["atomic-ready-3"],
        )
        minimal = artifact["configs"][0]
        self.assertEqual(minimal["clean"]["pass_rate"], 0.6667)
        self.assertEqual(
            minimal["clean"]["reliability"]["mixed_result_cell_rate"], 1.0
        )
        self.assertEqual(
            artifact["prompt_deltas_principled_minus_minimal"][0][
                "clean_pass_rate"
            ],
            0.3333,
        )
        low, high = minimal["clean"]["pass_rate_wilson_95"]
        self.assertLess(low, minimal["clean"]["pass_rate"])
        self.assertGreater(high, minimal["clean"]["pass_rate"])
        markdown = render_challenge_markdown(artifact)
        self.assertIn("Repeated atomic-workflow challenge", markdown)
        self.assertIn("all-trials", markdown)
        self.assertIn("Concurrency race", markdown)
        reversed_artifact = build_challenge_artifact(
            list(reversed(records)),
            models=["gpt-5.6-luna"],
            prompts=["minimal", "principled"],
            repetitions=3,
            fault_change_set_ids=["atomic-ready-3"],
        )
        self.assertEqual(artifact["configs"], reversed_artifact["configs"])

    def test_checkpoint_round_trip_keeps_latest_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            first = _record(
                "gpt-5.6-luna", "minimal", "clean-01", "none", 1, True
            )
            second = {**first, "episode_latency_ms": 300.0}
            append_challenge_checkpoint(path, first)
            append_challenge_checkpoint(path, second)
            loaded = load_challenge_checkpoint(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[first["episode_id"]]["episode_latency_ms"], 300.0)


if __name__ == "__main__":
    unittest.main()
