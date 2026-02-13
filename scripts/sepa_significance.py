#!/usr/bin/env python3
"""
Compute statistical significance for SEPA vs baseline run groups.

Example:
  uv run python scripts/sepa_significance.py \
    --baseline results/sepa_cmp_hicra_confirm_v1 \
    --candidate results/sepa_cmp_linear8_confirm_v1 \
    --output results/sepa_significance_confirm_v1.json \
    --markdown results/sepa_significance_confirm_v1.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textpolicy.analysis import (
    build_sepa_significance_markdown,
    evaluate_sepa_significance,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SEPA-vs-baseline statistical significance tests."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Baseline run directory (repeat flag for multiple runs).",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate run directory (repeat flag for multiple runs).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=20000,
        help="Permutation/bootstrap resamples (default: 20000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for resampling.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional markdown output path.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    report = evaluate_sepa_significance(
        baseline_run_dirs=args.baseline,
        candidate_run_dirs=args.candidate,
        alpha=float(args.alpha),
        num_resamples=int(args.resamples),
        seed=int(args.seed),
    )
    payload = report.to_dict()

    print(f"Recommendation: {report.recommendation}")
    for test in report.mean_tests:
        verdict = "SIG+" if test.significant_improvement else "n.s."
        print(
            f"[{verdict}] {test.metric}: delta={test.delta:+.6f} "
            f"p={test.p_value:.4f} ci=[{test.ci_low:+.6f},{test.ci_high:+.6f}]"
        )
    rate = report.rate_test
    rate_verdict = "SIG+" if rate.significant_improvement else "n.s."
    print(
        f"[{rate_verdict}] {rate.metric}: delta={rate.delta:+.6f} "
        f"p={rate.p_value:.4f} ci=[{rate.ci_low:+.6f},{rate.ci_high:+.6f}] "
        f"({rate.baseline_success}/{rate.baseline_total} -> "
        f"{rate.candidate_success}/{rate.candidate_total})"
    )

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {out}")

    if args.markdown:
        out_md = Path(args.markdown).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_sepa_significance_markdown(report), encoding="utf-8")
        print(f"Wrote markdown report: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
