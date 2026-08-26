"""Command-line interface for running and checking benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .benchmark import build_benchmark, deterministic_projection, write_artifacts


DEFAULT_JSON = Path("results/benchmark.json")
DEFAULT_MARKDOWN = Path("results/benchmark.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-run the benchmark and fail if deterministic fields drift from the committed artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_benchmark(repeats=args.repeats)
    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing committed benchmark artifact: {args.output}")
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        if deterministic_projection(artifact) != deterministic_projection(committed):
            raise SystemExit("Benchmark drift detected; regenerate and review results/benchmark.json")
        if not artifact["candidate_gates"]["passed"]:
            raise SystemExit("Candidate release gates failed")
        mutation_testing = artifact["mutation_testing"]
        if mutation_testing["caught"] != mutation_testing["total"]:
            raise SystemExit("Mutation testing did not detect every seeded regression")
        print(
            f"PASS: {artifact['dataset']['candidate_observations']} candidate observations; "
            f"{mutation_testing['caught']}/{mutation_testing['total']} seeded regressions detected"
        )
        return 0
    write_artifacts(artifact, args.output, args.markdown)
    quality = artifact["candidate"]["quality"]
    print(
        f"Wrote {args.output} and {args.markdown}: "
        f"route={quality['route_accuracy']:.2%}, answer={quality['answer_accuracy']:.2%}, "
        f"unsupported={quality['unsupported_answer_rate']:.2%}"
    )
    return 0
