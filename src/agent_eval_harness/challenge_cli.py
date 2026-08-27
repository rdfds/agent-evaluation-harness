"""Run or verify the repeated atomic-workflow challenge benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from .challenge_live_benchmark import (
    DEFAULT_CHALLENGE_MODELS,
    DEFAULT_CHALLENGE_PROMPTS,
    DEFAULT_REPETITIONS,
    build_challenge_artifact,
    build_challenge_specs,
    execute_challenge_specs,
    render_challenge_markdown,
    verify_challenge_artifact,
    write_challenge_artifacts,
)
from .challenge_model_agent import ChallengePromptVariant
from .live_client import ResponsesClient, load_api_key


DEFAULT_CHECKPOINT = Path(".challenge_eval_checkpoint.jsonl")
DEFAULT_JSON = Path("results/challenge_model_benchmark.json")
DEFAULT_MARKDOWN = Path("results/challenge_model_benchmark.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_CHALLENGE_MODELS))
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(DEFAULT_CHALLENGE_PROMPTS),
        default=list(DEFAULT_CHALLENGE_PROMPTS),
    )
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Recompute recorded scores and summaries without an API key.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        if not args.output.exists() or not args.markdown.exists():
            raise SystemExit("Missing committed challenge benchmark artifacts")
        artifact = json.loads(args.output.read_text(encoding="utf-8"))
        try:
            summary = verify_challenge_artifact(artifact)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if render_challenge_markdown(artifact) != args.markdown.read_text(encoding="utf-8"):
            raise SystemExit("Challenge benchmark Markdown drift detected")
        print(
            f"PASS: {summary['episodes']} challenge episodes; "
            f"{summary['passed']} passed; {summary['failed']} failed"
        )
        return 0

    prompts = cast(list[ChallengePromptVariant], args.prompts)
    specs = build_challenge_specs(
        models=args.models,
        prompts=prompts,
        repetitions=args.repetitions,
    )
    client = ResponsesClient(load_api_key())

    def progress(completed: int, total: int, record: dict) -> None:
        if record.get("resumed"):
            print(f"Resuming with {completed}/{total} checkpointed challenge episodes")
            return
        status = "PASS" if record["score"]["passed"] else "FAIL"
        print(
            f"[{completed:>3}/{total}] {status} "
            f"{record['requested_model']}/{record['prompt_variant']} "
            f"{record['case_id']} fault={record['fault_plan']} trial={record['trial']}"
        )

    records = execute_challenge_specs(
        specs,
        client=client,
        checkpoint_path=args.checkpoint,
        max_workers=args.max_workers,
        reasoning_effort=args.reasoning_effort,
        progress=progress,
    )
    artifact = build_challenge_artifact(
        records,
        models=args.models,
        prompts=prompts,
        repetitions=args.repetitions,
    )
    write_challenge_artifacts(artifact, args.output, args.markdown)
    totals = artifact["totals"]
    print(
        f"Wrote {args.output} and {args.markdown}: "
        f"episodes={totals['episodes']}, passed={totals['passed']}, "
        f"cost=${totals['estimated_cost_usd']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
