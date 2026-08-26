"""Run the resumable model and prompt workflow benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, cast

from .live_benchmark import (
    DEFAULT_MODELS,
    DEFAULT_PROMPTS,
    build_live_artifact,
    build_live_episode_specs,
    execute_live_specs,
    write_live_artifacts,
)
from .live_client import ResponsesClient, load_api_key
from .model_agent import PromptVariant


DEFAULT_CHECKPOINT = Path(".live_eval_checkpoint.jsonl")
DEFAULT_JSON = Path("results/live_model_benchmark.json")
DEFAULT_MARKDOWN = Path("results/live_model_benchmark.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(DEFAULT_PROMPTS),
        default=list(DEFAULT_PROMPTS),
    )
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prompts = cast(list[PromptVariant], args.prompts)
    specs = build_live_episode_specs(models=args.models, prompts=prompts)
    client = ResponsesClient(load_api_key())

    def progress(completed: int, total: int, record: dict) -> None:
        if record.get("resumed"):
            print(f"Resuming with {completed}/{total} checkpointed episodes")
            return
        status = "PASS" if record["score"]["passed"] else "FAIL"
        print(
            f"[{completed:>3}/{total}] {status} "
            f"{record['requested_model']}/{record['prompt_variant']} "
            f"{record['case_id']} fault={record['fault']}"
        )

    records = execute_live_specs(
        specs,
        client=client,
        checkpoint_path=args.checkpoint,
        max_workers=args.max_workers,
        reasoning_effort=args.reasoning_effort,
        progress=progress,
    )
    artifact = build_live_artifact(records, models=args.models, prompts=prompts)
    write_live_artifacts(artifact, args.output, args.markdown)
    totals = artifact["totals"]
    print(
        f"Wrote {args.output} and {args.markdown}: "
        f"episodes={totals['episodes']}, cost=${totals['estimated_cost_usd']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
