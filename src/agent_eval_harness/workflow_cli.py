"""Run or verify the committed stateful workflow benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .workflow_benchmark import build_workflow_benchmark, write_workflow_artifacts


DEFAULT_JSON = Path("results/workflow_benchmark.json")
DEFAULT_MARKDOWN = Path("results/workflow_benchmark.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the deterministic benchmark differs from the committed artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_workflow_benchmark()
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing committed workflow artifact: {args.output}")
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        if artifact != committed:
            raise SystemExit(
                "Workflow benchmark drift detected; regenerate and review the artifact"
            )
        if not artifact["release_gates"]["passed"]:
            raise SystemExit("Workflow release gates failed")
        print(
            "PASS: "
            f"{artifact['dataset']['clean_cases']} clean tasks; "
            f"{artifact['dataset']['fault_episodes_per_agent']} fault episodes per agent"
        )
        return 0
    write_workflow_artifacts(artifact, args.output, args.markdown)
    print(
        f"Wrote {args.output} and {args.markdown}: "
        f"clean={artifact['candidate']['clean']['pass_rate']:.2%}, "
        f"faults={artifact['candidate']['faults']['fault_handling_rate']:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
