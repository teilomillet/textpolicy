#!/usr/bin/env python3
"""
Map SEPA dynamics from a completed paired campaign.

This script focuses on *why* a campaign passed/failed by analyzing:
- phase behavior (early/mid/late) for key metrics
- stepwise coupling/decoupling between entropy and outcomes
- trend slopes and lagged relationships

Example:
  uv run python scripts/sepa_dynamics_map.py \
    --campaign-root results/sepa_campaign_2026_02_11_consequential_v1 \
    --output results/sepa_campaign_2026_02_11_consequential_v1/analysis/dynamics.json \
    --markdown results/sepa_campaign_2026_02_11_consequential_v1/analysis/dynamics.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _fraction(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return sum(1 for v in values if v) / len(values)


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(vx * vy)


def _slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    if varx <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / varx


@dataclass(frozen=True)
class StepPoint:
    seed: int
    step: int
    mean_reward: float
    correct_rate: float
    planning_ratio: Optional[float]
    gram_entropy_delta: Optional[float]
    sepa_lambda: Optional[float]


@dataclass(frozen=True)
class PhaseSummary:
    phase: str
    baseline_mean_reward: Optional[float]
    candidate_mean_reward: Optional[float]
    delta_mean_reward: Optional[float]
    baseline_correct_rate: Optional[float]
    candidate_correct_rate: Optional[float]
    delta_correct_rate: Optional[float]
    baseline_gram_entropy_delta: Optional[float]
    candidate_gram_entropy_delta: Optional[float]
    delta_gram_entropy_delta: Optional[float]
    baseline_planning_ratio: Optional[float]
    candidate_planning_ratio: Optional[float]
    delta_planning_ratio: Optional[float]
    candidate_sepa_lambda: Optional[float]


@dataclass(frozen=True)
class CorrelationSummary:
    label: str
    corr: Optional[float]
    n: int


@dataclass(frozen=True)
class DynamicsReport:
    campaign_root: str
    paired_seeds: List[int]
    max_step: int
    phase_summaries: List[PhaseSummary]
    trend_slopes: Dict[str, Optional[float]]
    correlations: List[CorrelationSummary]
    decoupling: Dict[str, Optional[float]]
    interpretation: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_root": self.campaign_root,
            "paired_seeds": list(self.paired_seeds),
            "max_step": self.max_step,
            "phase_summaries": [asdict(row) for row in self.phase_summaries],
            "trend_slopes": dict(self.trend_slopes),
            "correlations": [asdict(row) for row in self.correlations],
            "decoupling": dict(self.decoupling),
            "interpretation": list(self.interpretation),
        }


def _collect_paired_steps(campaign_root: Path) -> Tuple[List[int], List[StepPoint], List[StepPoint]]:
    seeds: List[int] = []
    baseline_points: List[StepPoint] = []
    candidate_points: List[StepPoint] = []

    baseline_dirs = sorted(campaign_root.glob("baseline_seed*"))
    for bdir in baseline_dirs:
        suffix = bdir.name.replace("baseline_seed", "")
        if not suffix.isdigit():
            continue
        seed = int(suffix)
        cdir = campaign_root / f"candidate_seed{seed}"
        if not cdir.exists():
            continue

        b_rows = _read_jsonl(bdir / "emergence" / "steps.jsonl")
        c_rows = _read_jsonl(cdir / "emergence" / "steps.jsonl")
        if not b_rows or not c_rows:
            continue

        b_by_step: Dict[int, Dict[str, Any]] = {}
        c_by_step: Dict[int, Dict[str, Any]] = {}
        for rec in b_rows:
            step = int(rec.get("step", 0))
            b_by_step[step] = rec
        for rec in c_rows:
            step = int(rec.get("step", 0))
            c_by_step[step] = rec

        shared_steps = sorted(set(b_by_step.keys()) & set(c_by_step.keys()))
        if not shared_steps:
            continue
        seeds.append(seed)

        for step in shared_steps:
            b = b_by_step[step]
            c = c_by_step[step]

            b_total = max(float(b.get("total_count", 0.0)), 1.0)
            c_total = max(float(c.get("total_count", 0.0)), 1.0)
            baseline_points.append(
                StepPoint(
                    seed=seed,
                    step=step,
                    mean_reward=float(b.get("mean_reward", 0.0)),
                    correct_rate=float(b.get("correct_count", 0.0)) / b_total,
                    planning_ratio=_safe_float(b.get("planning_token_ratio")),
                    gram_entropy_delta=_safe_float(b.get("gram_entropy_delta")),
                    sepa_lambda=_safe_float(b.get("sepa_lambda")),
                )
            )
            candidate_points.append(
                StepPoint(
                    seed=seed,
                    step=step,
                    mean_reward=float(c.get("mean_reward", 0.0)),
                    correct_rate=float(c.get("correct_count", 0.0)) / c_total,
                    planning_ratio=_safe_float(c.get("planning_token_ratio")),
                    gram_entropy_delta=_safe_float(c.get("gram_entropy_delta")),
                    sepa_lambda=_safe_float(c.get("sepa_lambda")),
                )
            )
    return sorted(seeds), baseline_points, candidate_points


def _phase_for_step(step: int, max_step: int) -> str:
    if max_step <= 0:
        return "unknown"
    third = max(max_step / 3.0, 1.0)
    if step <= third:
        return "early"
    if step <= 2.0 * third:
        return "mid"
    return "late"


def _phase_summaries(
    baseline_points: Sequence[StepPoint],
    candidate_points: Sequence[StepPoint],
    max_step: int,
) -> List[PhaseSummary]:
    phase_names = ("early", "mid", "late")
    summaries: List[PhaseSummary] = []
    for phase in phase_names:
        b = [p for p in baseline_points if _phase_for_step(p.step + 1, max_step) == phase]
        c = [p for p in candidate_points if _phase_for_step(p.step + 1, max_step) == phase]

        b_reward = [p.mean_reward for p in b]
        c_reward = [p.mean_reward for p in c]
        b_correct = [p.correct_rate for p in b]
        c_correct = [p.correct_rate for p in c]
        b_entropy = [p.gram_entropy_delta for p in b if p.gram_entropy_delta is not None]
        c_entropy = [p.gram_entropy_delta for p in c if p.gram_entropy_delta is not None]
        b_plan = [p.planning_ratio for p in b if p.planning_ratio is not None]
        c_plan = [p.planning_ratio for p in c if p.planning_ratio is not None]
        c_lambda = [p.sepa_lambda for p in c if p.sepa_lambda is not None]

        def _delta(a: Optional[float], b_: Optional[float]) -> Optional[float]:
            if a is None or b_ is None:
                return None
            return b_ - a

        mb_reward = _mean(b_reward)
        mc_reward = _mean(c_reward)
        mb_correct = _mean(b_correct)
        mc_correct = _mean(c_correct)
        mb_entropy = _mean(b_entropy)
        mc_entropy = _mean(c_entropy)
        mb_plan = _mean([float(x) for x in b_plan])
        mc_plan = _mean([float(x) for x in c_plan])
        mc_lambda = _mean([float(x) for x in c_lambda])

        summaries.append(
            PhaseSummary(
                phase=phase,
                baseline_mean_reward=mb_reward,
                candidate_mean_reward=mc_reward,
                delta_mean_reward=_delta(mb_reward, mc_reward),
                baseline_correct_rate=mb_correct,
                candidate_correct_rate=mc_correct,
                delta_correct_rate=_delta(mb_correct, mc_correct),
                baseline_gram_entropy_delta=mb_entropy,
                candidate_gram_entropy_delta=mc_entropy,
                delta_gram_entropy_delta=_delta(mb_entropy, mc_entropy),
                baseline_planning_ratio=mb_plan,
                candidate_planning_ratio=mc_plan,
                delta_planning_ratio=_delta(mb_plan, mc_plan),
                candidate_sepa_lambda=mc_lambda,
            )
        )
    return summaries


def _paired_by_seed_step(points: Sequence[StepPoint]) -> Dict[Tuple[int, int], StepPoint]:
    return {(p.seed, p.step): p for p in points}


def _trend_slopes(
    baseline_points: Sequence[StepPoint],
    candidate_points: Sequence[StepPoint],
) -> Dict[str, Optional[float]]:
    b_by = _paired_by_seed_step(baseline_points)
    c_by = _paired_by_seed_step(candidate_points)
    keys = sorted(set(b_by.keys()) & set(c_by.keys()))
    if not keys:
        return {}

    steps_reward: List[float] = []
    delta_reward: List[float] = []
    steps_correct: List[float] = []
    delta_correct: List[float] = []
    steps_entropy: List[float] = []
    delta_entropy: List[float] = []
    steps_planning: List[float] = []
    delta_planning: List[float] = []
    steps_lambda: List[float] = []
    lambda_values: List[float] = []

    for key in keys:
        b = b_by[key]
        c = c_by[key]
        step_val = float(b.step + 1)
        steps_reward.append(step_val)
        delta_reward.append(c.mean_reward - b.mean_reward)
        steps_correct.append(step_val)
        delta_correct.append(c.correct_rate - b.correct_rate)

        if b.gram_entropy_delta is not None and c.gram_entropy_delta is not None:
            steps_entropy.append(step_val)
            delta_entropy.append(c.gram_entropy_delta - b.gram_entropy_delta)
        if b.planning_ratio is not None and c.planning_ratio is not None:
            steps_planning.append(step_val)
            delta_planning.append(c.planning_ratio - b.planning_ratio)
        if c.sepa_lambda is not None:
            steps_lambda.append(step_val)
            lambda_values.append(c.sepa_lambda)

    slopes: Dict[str, Optional[float]] = {
        "delta_reward_per_step": _slope(steps_reward, delta_reward),
        "delta_correct_rate_per_step": _slope(steps_correct, delta_correct),
        "delta_gram_entropy_per_step": _slope(steps_entropy, delta_entropy),
        "delta_planning_ratio_per_step": _slope(steps_planning, delta_planning),
        "candidate_lambda_per_step": _slope(steps_lambda, lambda_values),
    }
    return slopes


def _collect_pairs(
    points: Sequence[StepPoint],
    x_getter,
    y_getter,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for p in points:
        x = _safe_float(x_getter(p))
        y = _safe_float(y_getter(p))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _lag_pairs(points: Sequence[StepPoint], x_key: str, y_key: str, lag: int = 1) -> Tuple[List[float], List[float]]:
    by_seed: Dict[int, Dict[int, StepPoint]] = {}
    for p in points:
        by_seed.setdefault(p.seed, {})[p.step] = p
    xs: List[float] = []
    ys: List[float] = []
    for seed_steps in by_seed.values():
        for step, p in seed_steps.items():
            q = seed_steps.get(step + lag)
            if q is None:
                continue
            x = _safe_float(getattr(p, x_key))
            y = _safe_float(getattr(q, y_key))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
    return xs, ys


def _correlations(
    baseline_points: Sequence[StepPoint],
    candidate_points: Sequence[StepPoint],
) -> List[CorrelationSummary]:
    rows: List[CorrelationSummary] = []

    def add(label: str, xs: List[float], ys: List[float]) -> None:
        rows.append(CorrelationSummary(label=label, corr=_pearson(xs, ys), n=min(len(xs), len(ys))))

    xs, ys = _collect_pairs(candidate_points, lambda p: p.sepa_lambda, lambda p: p.gram_entropy_delta)
    add("candidate: sepa_lambda -> gram_entropy_delta", xs, ys)

    xs, ys = _collect_pairs(candidate_points, lambda p: p.gram_entropy_delta, lambda p: p.mean_reward)
    add("candidate: gram_entropy_delta -> mean_reward", xs, ys)

    xs, ys = _collect_pairs(candidate_points, lambda p: p.planning_ratio, lambda p: p.mean_reward)
    add("candidate: planning_ratio -> mean_reward", xs, ys)

    xs, ys = _collect_pairs(candidate_points, lambda p: p.gram_entropy_delta, lambda p: p.correct_rate)
    add("candidate: gram_entropy_delta -> correct_rate", xs, ys)

    xs, ys = _collect_pairs(baseline_points, lambda p: p.gram_entropy_delta, lambda p: p.mean_reward)
    add("baseline: gram_entropy_delta -> mean_reward", xs, ys)

    xs, ys = _lag_pairs(candidate_points, "gram_entropy_delta", "mean_reward", lag=1)
    add("candidate lag1: gram_entropy_delta_t -> mean_reward_t+1", xs, ys)

    xs, ys = _lag_pairs(candidate_points, "gram_entropy_delta", "correct_rate", lag=1)
    add("candidate lag1: gram_entropy_delta_t -> correct_rate_t+1", xs, ys)

    xs, ys = _lag_pairs(candidate_points, "planning_ratio", "mean_reward", lag=1)
    add("candidate lag1: planning_ratio_t -> mean_reward_t+1", xs, ys)

    return rows


def _decoupling(
    baseline_points: Sequence[StepPoint],
    candidate_points: Sequence[StepPoint],
) -> Dict[str, Optional[float]]:
    b_by = _paired_by_seed_step(baseline_points)
    c_by = _paired_by_seed_step(candidate_points)
    keys = sorted(set(b_by.keys()) & set(c_by.keys()))
    if not keys:
        return {}

    entropy_up: List[bool] = []
    reward_up: List[bool] = []
    correct_up: List[bool] = []
    entropy_up_reward_not_up: List[bool] = []
    entropy_up_correct_not_up: List[bool] = []
    same_sign_entropy_reward: List[bool] = []

    for key in keys:
        b = b_by[key]
        c = c_by[key]
        d_reward = c.mean_reward - b.mean_reward
        d_correct = c.correct_rate - b.correct_rate
        d_entropy = None
        if b.gram_entropy_delta is not None and c.gram_entropy_delta is not None:
            d_entropy = c.gram_entropy_delta - b.gram_entropy_delta
        if d_entropy is None:
            continue

        eu = d_entropy > 0.0
        ru = d_reward > 0.0
        cu = d_correct > 0.0

        entropy_up.append(eu)
        reward_up.append(ru)
        correct_up.append(cu)
        entropy_up_reward_not_up.append(eu and not ru)
        entropy_up_correct_not_up.append(eu and not cu)
        same_sign_entropy_reward.append((d_entropy >= 0.0 and d_reward >= 0.0) or (d_entropy < 0.0 and d_reward < 0.0))

    return {
        "entropy_up_rate": _fraction(entropy_up),
        "reward_up_rate": _fraction(reward_up),
        "correct_up_rate": _fraction(correct_up),
        "entropy_up_and_reward_not_up_rate": _fraction(entropy_up_reward_not_up),
        "entropy_up_and_correct_not_up_rate": _fraction(entropy_up_correct_not_up),
        "entropy_reward_same_sign_rate": _fraction(same_sign_entropy_reward),
    }


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _interpretation(report: DynamicsReport) -> List[str]:
    out: List[str] = []

    phases = {row.phase: row for row in report.phase_summaries}
    late = phases.get("late")
    if late is not None:
        d_entropy = late.delta_gram_entropy_delta
        d_reward = late.delta_mean_reward
        if d_entropy is not None and d_reward is not None:
            if d_entropy > 0 and d_reward <= 0:
                out.append(
                    "Late training shows entropy lift without reward lift (mechanistic shift, no payoff)."
                )
            elif d_entropy > 0 and d_reward > 0:
                out.append("Late training aligns entropy lift with reward lift.")
            elif d_entropy <= 0 and d_reward > 0:
                out.append("Reward lift appears independent of entropy lift in late phase.")

    corr_map = {row.label: row for row in report.correlations}
    lambda_entropy = corr_map.get("candidate: sepa_lambda -> gram_entropy_delta")
    if lambda_entropy and lambda_entropy.corr is not None:
        if lambda_entropy.corr > 0.2:
            out.append("Higher sepa_lambda is positively associated with gram entropy delta.")
        elif lambda_entropy.corr < -0.2:
            out.append("Higher sepa_lambda is associated with lower gram entropy delta.")
        else:
            out.append("sepa_lambda has weak association with gram entropy delta in this run.")

    entropy_reward = corr_map.get("candidate: gram_entropy_delta -> mean_reward")
    if entropy_reward and entropy_reward.corr is not None:
        if abs(entropy_reward.corr) < 0.1:
            out.append("Entropy shifts are weakly coupled to reward at the same step.")
        elif entropy_reward.corr > 0:
            out.append("Entropy shifts are positively coupled with reward.")
        else:
            out.append("Entropy shifts are negatively coupled with reward.")

    lag_entropy_reward = corr_map.get("candidate lag1: gram_entropy_delta_t -> mean_reward_t+1")
    if lag_entropy_reward and lag_entropy_reward.corr is not None:
        if abs(lag_entropy_reward.corr) < 0.1:
            out.append("Lagged entropy->reward signal is weak (little delayed payoff).")
        elif lag_entropy_reward.corr > 0:
            out.append("Entropy increase tends to precede next-step reward increase.")
        else:
            out.append("Entropy increase tends to precede next-step reward decrease.")

    dec = report.decoupling
    d = dec.get("entropy_up_and_reward_not_up_rate")
    if d is not None:
        if d >= 0.5:
            out.append("Decoupling is strong: entropy rises often occur without reward improvement.")
        elif d >= 0.3:
            out.append("Decoupling is moderate: many entropy rises do not translate to reward.")
        else:
            out.append("Decoupling is limited: entropy rises often co-occur with reward rises.")

    if not out:
        out.append("No strong dynamic pattern detected from available step-level signals.")
    return out


def evaluate_dynamics(campaign_root: Path) -> DynamicsReport:
    seeds, baseline_points, candidate_points = _collect_paired_steps(campaign_root)
    if not seeds:
        raise ValueError("No paired baseline/candidate runs with step logs were found.")

    max_step = 0
    for p in baseline_points:
        max_step = max(max_step, p.step + 1)
    for p in candidate_points:
        max_step = max(max_step, p.step + 1)

    phase_rows = _phase_summaries(baseline_points, candidate_points, max_step)
    slopes = _trend_slopes(baseline_points, candidate_points)
    corrs = _correlations(baseline_points, candidate_points)
    dec = _decoupling(baseline_points, candidate_points)

    interpretation = _interpretation(
        DynamicsReport(
            campaign_root=str(campaign_root),
            paired_seeds=seeds,
            max_step=max_step,
            phase_summaries=phase_rows,
            trend_slopes=slopes,
            correlations=corrs,
            decoupling=dec,
            interpretation=[],
        )
    )
    report = DynamicsReport(
        campaign_root=str(campaign_root),
        paired_seeds=seeds,
        max_step=max_step,
        phase_summaries=phase_rows,
        trend_slopes=slopes,
        correlations=corrs,
        decoupling=dec,
        interpretation=interpretation,
    )
    return report


def build_markdown(report: DynamicsReport) -> str:
    lines: List[str] = []
    lines.append("# SEPA Dynamics Map")
    lines.append("")
    lines.append(f"- campaign: `{report.campaign_root}`")
    lines.append(f"- paired seeds: {len(report.paired_seeds)} ({', '.join(str(s) for s in report.paired_seeds)})")
    lines.append(f"- max step: {report.max_step}")
    lines.append("")
    lines.append("## Interpretation")
    for msg in report.interpretation:
        lines.append(f"- {msg}")
    lines.append("")
    lines.append("## Phase Summary (candidate - baseline)")
    lines.append(
        "| phase | d_reward | d_correct | d_gram_entropy | d_planning | candidate_lambda |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in report.phase_summaries:
        lines.append(
            f"| {row.phase} | {_fmt(row.delta_mean_reward)} | {_fmt(row.delta_correct_rate)} | "
            f"{_fmt(row.delta_gram_entropy_delta)} | {_fmt(row.delta_planning_ratio)} | "
            f"{_fmt(row.candidate_sepa_lambda)} |"
        )
    lines.append("")
    lines.append("## Trend Slopes")
    for key, value in report.trend_slopes.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    lines.append("## Correlations")
    lines.append("| label | corr | n |")
    lines.append("|---|---:|---:|")
    for row in report.correlations:
        lines.append(f"| {row.label} | {_fmt(row.corr)} | {row.n} |")
    lines.append("")
    lines.append("## Decoupling Diagnostics")
    for key, value in report.decoupling.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    return "\n".join(lines)


def _default_output_paths(campaign_root: Path) -> Tuple[Path, Path]:
    analysis_dir = campaign_root / "analysis"
    return analysis_dir / "dynamics.json", analysis_dir / "dynamics.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map step-level dynamics behind SEPA campaign outcomes."
    )
    parser.add_argument(
        "--campaign-root",
        required=True,
        help="Campaign directory containing baseline_seed*/candidate_seed* folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path (default: <campaign-root>/analysis/dynamics.json).",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional markdown output path (default: <campaign-root>/analysis/dynamics.md).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    report = evaluate_dynamics(campaign_root)
    payload = report.to_dict()
    md = build_markdown(report)

    default_json, default_md = _default_output_paths(campaign_root)
    json_path = Path(args.output).expanduser().resolve() if args.output else default_json
    md_path = Path(args.markdown).expanduser().resolve() if args.markdown else default_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md, encoding="utf-8")

    print(f"Wrote dynamics JSON: {json_path}")
    print(f"Wrote dynamics markdown: {md_path}")
    print("Interpretation:")
    for item in report.interpretation:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
