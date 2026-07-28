#!/usr/bin/env python3
"""Evaluate CysticCare research-paper or supporting-passage retrieval.

Examples:

  # Existing generated benchmark output: evaluate final retrieved papers.
  python evaluate_retrieval.py \
    --benchmark app/pipeline_results.json \
    --predictions app/pipeline_results_with_cysticcare.json \
    --k 1,3,5,10 --label-level paper

  # A future trace file with ``chunk_id`` fields: evaluate exact passages.
  python evaluate_retrieval.py \
    --benchmark app/pipeline_results.json --predictions retrieval_trace.jsonl \
    --label-level chunk --per-query --output chunk_metrics.json

Inputs can be JSON, JSONL/NDJSON, or CSV.  See ``app.retrieval_evaluation`` for
the accepted prediction shapes and metric definitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.retrieval_evaluation import (
    DEFAULT_CUTOFFS,
    SUPPORTED_LABEL_LEVELS,
    evaluate_records,
    load_records,
    parse_cutoffs,
    write_report,
)


HERE = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = HERE / "app" / "pipeline_results.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK),
        help="Benchmark with retrieval_targets/supporting_passages labels (default: %(default)s)",
    )
    parser.add_argument(
        "--predictions",
        help=(
            "Ranked retrieval output. Omit only when the benchmark itself contains "
            "retrieved sources (for example cysticcare_metadata.sources)."
        ),
    )
    parser.add_argument(
        "--k",
        default=",".join(str(k) for k in DEFAULT_CUTOFFS),
        help="Comma-separated cutoffs, e.g. 1,3,5,10 (default: %(default)s)",
    )
    parser.add_argument(
        "--label-level",
        choices=sorted(SUPPORTED_LABEL_LEVELS),
        default="paper",
        help=(
            "paper = retrieval_targets; chunk = supporting_passages; auto uses chunk "
            "only when predictions include stable chunk IDs (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--per-query",
        action="store_true",
        help="Include each query's metric values in the JSON report.",
    )
    parser.add_argument(
        "--output",
        help="Write JSON report to this path; otherwise print it to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        benchmark_records = load_records(args.benchmark)
        prediction_records = load_records(args.predictions) if args.predictions else None
        report = evaluate_records(
            benchmark_records,
            prediction_records,
            cutoffs=parse_cutoffs(args.k),
            label_level=args.label_level,
            include_per_query=args.per_query,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        try:
            write_report(report, args.output)
        except OSError as exc:
            print(f"error: could not write {args.output}: {exc}", file=sys.stderr)
            return 2
        print(f"Wrote retrieval evaluation to {args.output}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
