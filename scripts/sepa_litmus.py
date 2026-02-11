#!/usr/bin/env python3
"""
Run a thresholded SEPA litmus test against baseline runs.

Example:
  uv run python scripts/sepa_litmus.py \
    --baseline results/sepa_cmp_hicra_confirm_v1 \
    --candidate results/sepa_cmp_linear8_confirm_v1 \
    --output results/sepa_litmus_confirm_v1.json \
    --markdown results/sepa_litmus_confirm_v1.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from textpolicy.analysis import (
    SEPALitmusEvidence,
    SEPALitmusThresholds,
    build_litmus_markdown,
    evaluate_sepa_litmus,
    get_sepa_litmus_profile,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether candidate SEPA runs clear explicit thresholds "
            "vs baseline runs, with evidence sufficiency gating."
        )
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
        "--profile",
        default="official_v1",
        help="Named litmus profile (default: official_v1).",
    )
    parser.add_argument(
        "--min-reward-lift",
        type=float,
        default=None,
        help="Override required minimum lift in mean_reward_steps (candidate - baseline).",
    )
    parser.add_argument(
        "--min-correct-rate-lift",
        type=float,
        default=None,
        help="Override required minimum lift in overall_correct_rate.",
    )
    parser.add_argument(
        "--min-gram-entropy-delta-lift",
        type=float,
        default=None,
        help="Override required minimum lift in mean_gram_entropy_delta.",
    )
    parser.add_argument(
        "--min-run-pairs",
        type=int,
        default=None,
        help="Override minimum number of baseline and candidate runs required.",
    )
    parser.add_argument(
        "--min-steps-per-run",
        type=int,
        default=None,
        help="Override minimum step count required for each run.",
    )
    parser.add_argument(
        "--min-generations-per-run",
        type=int,
        default=None,
        help="Override minimum generation count required for each run.",
    )
    parser.add_argument(
        "--min-gram-pairs-per-run",
        type=int,
        default=None,
        help="Override minimum gram-pair count required for each run.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional markdown report output path.",
    )
    return parser


def _status_exit_code(status: str) -> int:
    if status == "CONFIRMED":
        return 0
    if status == "FAILED":
        return 1
    return 2


def _print_summary(
    *,
    baseline_runs: Sequence[str],
    candidate_runs: Sequence[str],
    status: str,
    checks: Sequence[dict],
    evidence_ok: bool,
    evidence_failures: Sequence[str],
) -> None:
    print(f"SEPA litmus status: {status}")
    print(f"Baseline runs ({len(baseline_runs)}):")
    for path in baseline_runs:
        print(f"  - {Path(path).resolve()}")
    print(f"Candidate runs ({len(candidate_runs)}):")
    for path in candidate_runs:
        print(f"  - {Path(path).resolve()}")
    print(f"Evidence ok: {evidence_ok}")
    for failure in evidence_failures:
        print(f"  evidence-failure: {failure}")
    for check in checks:
        marker = "PASS" if check["passed"] else "FAIL"
        print(
            f"  [{marker}] {check['metric']}: "
            f"delta={check['delta']:.6f} "
            f"(threshold={check['threshold']:.6f})"
        )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    profile = get_sepa_litmus_profile(args.profile)
    thresholds = profile.thresholds
    evidence = profile.evidence

    if args.min_reward_lift is not None:
        thresholds = replace(thresholds, min_mean_reward_lift=float(args.min_reward_lift))
    if args.min_correct_rate_lift is not None:
        thresholds = replace(
            thresholds, min_correct_rate_lift=float(args.min_correct_rate_lift)
        )
    if args.min_gram_entropy_delta_lift is not None:
        thresholds = replace(
            thresholds,
            min_gram_entropy_delta_lift=float(args.min_gram_entropy_delta_lift),
        )

    if args.min_run_pairs is not None:
        evidence = replace(evidence, min_run_pairs=int(args.min_run_pairs))
    if args.min_steps_per_run is not None:
        evidence = replace(evidence, min_steps_per_run=int(args.min_steps_per_run))
    if args.min_generations_per_run is not None:
        evidence = replace(
            evidence,
            min_generations_per_run=int(args.min_generations_per_run),
        )
    if args.min_gram_pairs_per_run is not None:
        evidence = replace(
            evidence,
            min_gram_pairs_per_run=int(args.min_gram_pairs_per_run),
        )

    result = evaluate_sepa_litmus(
        baseline_run_dirs=args.baseline,
        candidate_run_dirs=args.candidate,
        thresholds=thresholds,
        evidence=evidence,
    )
    payload = result.to_dict()

    _print_summary(
        baseline_runs=args.baseline,
        candidate_runs=args.candidate,
        status=result.status,
        checks=payload["checks"],
        evidence_ok=result.evidence_ok,
        evidence_failures=result.evidence_failures,
    )
    print(f"Profile: {profile.name}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON report: {output_path}")

    if args.markdown:
        md_path = Path(args.markdown).expanduser().resolve()
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(build_litmus_markdown(result), encoding="utf-8")
        print(f"Wrote markdown report: {md_path}")

    return _status_exit_code(result.status)


if __name__ == "__main__":
    raise SystemExit(main())
