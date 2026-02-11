"""
Statistical significance utilities for SEPA vs baseline comparisons.

Consumes emergence logs from one or more run directories and computes:
- mean-difference tests (permutation p-value + bootstrap CI)
- correctness-rate comparison (Fisher exact test)
- rough per-arm sample-size estimates at 80% power
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class MeanDiffTestResult:
    metric: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    n_baseline: int
    n_candidate: int
    significant: bool
    significant_improvement: bool
    cohens_d: float


@dataclass(frozen=True)
class RateDiffTestResult:
    metric: str
    baseline_rate: float
    candidate_rate: float
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    baseline_success: int
    baseline_total: int
    candidate_success: int
    candidate_total: int
    significant: bool
    significant_improvement: bool


@dataclass(frozen=True)
class SampleSizeEstimate:
    metric: str
    observed_effect: float
    n_per_arm_80pct: Optional[int]
    unit: str


@dataclass(frozen=True)
class SEPASignificanceReport:
    alpha: float
    baseline_runs: List[str]
    candidate_runs: List[str]
    mean_tests: List[MeanDiffTestResult]
    rate_test: RateDiffTestResult
    sample_size_estimates: List[SampleSizeEstimate]
    significant_improvement_metrics: List[str]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "baseline_runs": list(self.baseline_runs),
            "candidate_runs": list(self.candidate_runs),
            "mean_tests": [asdict(t) for t in self.mean_tests],
            "rate_test": asdict(self.rate_test),
            "sample_size_estimates": [asdict(s) for s in self.sample_size_estimates],
            "significant_improvement_metrics": list(self.significant_improvement_metrics),
            "recommendation": self.recommendation,
        }


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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)


def _pooled_sd(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n1 = len(a)
    n2 = len(b)
    pooled_var = (
        ((n1 - 1) * _sample_variance(a)) + ((n2 - 1) * _sample_variance(b))
    ) / (n1 + n2 - 2)
    return math.sqrt(max(pooled_var, 0.0))


def _permutation_p_value(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    num_resamples: int,
    seed: int,
) -> float:
    if not baseline or not candidate:
        return 1.0

    rng = random.Random(seed)
    obs = _mean(candidate) - _mean(baseline)
    pooled = list(baseline) + list(candidate)
    n_base = len(baseline)
    extreme = 0
    for _ in range(num_resamples):
        rng.shuffle(pooled)
        base_s = pooled[:n_base]
        cand_s = pooled[n_base:]
        delta = _mean(cand_s) - _mean(base_s)
        if abs(delta) >= abs(obs):
            extreme += 1
    return (extreme + 1) / (num_resamples + 1)


def _bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    num_resamples: int,
    seed: int,
) -> tuple[float, float]:
    if not baseline or not candidate:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n_base = len(baseline)
    n_cand = len(candidate)
    deltas: List[float] = []
    for _ in range(num_resamples):
        base_s = [baseline[rng.randrange(n_base)] for _ in range(n_base)]
        cand_s = [candidate[rng.randrange(n_cand)] for _ in range(n_cand)]
        deltas.append(_mean(cand_s) - _mean(base_s))
    deltas.sort()
    lo = deltas[int(0.025 * num_resamples)]
    hi = deltas[int(0.975 * num_resamples)]
    return (lo, hi)


def _fisher_exact_two_sided(
    baseline_success: int,
    baseline_fail: int,
    candidate_success: int,
    candidate_fail: int,
) -> float:
    n1 = baseline_success + baseline_fail
    n2 = candidate_success + candidate_fail
    k = baseline_success + candidate_success
    n = n1 + n2
    if n1 <= 0 or n2 <= 0:
        return 1.0

    def _hyper(x: int) -> float:
        return (
            math.comb(n1, x) * math.comb(n2, k - x)
        ) / math.comb(n, k)

    xmin = max(0, k - n2)
    xmax = min(n1, k)
    p_obs = _hyper(baseline_success)
    p = 0.0
    for x in range(xmin, xmax + 1):
        px = _hyper(x)
        if px <= p_obs + 1e-15:
            p += px
    return float(min(max(p, 0.0), 1.0))


def _normal_ci_rate_delta(
    baseline_rate: float,
    baseline_total: int,
    candidate_rate: float,
    candidate_total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if baseline_total <= 0 or candidate_total <= 0:
        return (0.0, 0.0)
    se = math.sqrt(
        (baseline_rate * (1.0 - baseline_rate) / baseline_total)
        + (candidate_rate * (1.0 - candidate_rate) / candidate_total)
    )
    delta = candidate_rate - baseline_rate
    return (delta - z * se, delta + z * se)


def _required_n_per_arm_for_mean(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> Optional[int]:
    if len(baseline) < 2 or len(candidate) < 2:
        return None
    delta = _mean(candidate) - _mean(baseline)
    sd = _pooled_sd(baseline, candidate)
    if sd <= 1e-12:
        return None
    d = abs(delta / sd)
    if d <= 1e-12:
        return None
    z_alpha = 1.959963984540054  # two-sided alpha=0.05
    z_beta = 0.8416212335729143  # power=0.80
    n = 2.0 * ((z_alpha + z_beta) ** 2) / (d**2)
    return max(int(math.ceil(n)), 1)


def _required_n_per_arm_for_rate(
    baseline_rate: float,
    candidate_rate: float,
) -> Optional[int]:
    delta = abs(candidate_rate - baseline_rate)
    if delta <= 1e-12:
        return None
    z_alpha = 1.959963984540054  # two-sided alpha=0.05
    z_beta = 0.8416212335729143  # power=0.80
    p_bar = 0.5 * (baseline_rate + candidate_rate)
    num = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_beta
        * math.sqrt(
            baseline_rate * (1.0 - baseline_rate)
            + candidate_rate * (1.0 - candidate_rate)
        )
    ) ** 2
    n = num / (delta**2)
    return max(int(math.ceil(n)), 1)


def _collect_metrics(run_dirs: Sequence[PathLike]) -> Dict[str, List[float]]:
    metrics: Dict[str, List[float]] = {
        "step_mean_reward": [],
        "step_gram_entropy_delta": [],
        "gen_reward": [],
        "gen_gram_entropy_delta": [],
        "gen_correctness": [],
    }
    for run_dir in run_dirs:
        root = Path(run_dir).expanduser().resolve()
        step_rows = _read_jsonl(root / "emergence" / "steps.jsonl")
        gen_rows = _read_jsonl(root / "emergence" / "generations.jsonl")

        for rec in step_rows:
            metrics["step_mean_reward"].append(float(rec.get("mean_reward", 0.0)))
            if rec.get("gram_entropy_delta") is not None:
                metrics["step_gram_entropy_delta"].append(
                    float(rec.get("gram_entropy_delta", 0.0))
                )

        for rec in gen_rows:
            metrics["gen_reward"].append(float(rec.get("reward", 0.0)))
            if rec.get("gram_entropy_delta") is not None:
                metrics["gen_gram_entropy_delta"].append(
                    float(rec.get("gram_entropy_delta", 0.0))
                )
            metadata = rec.get("metadata")
            if isinstance(metadata, dict) and "correctness" in metadata:
                correct = bool(metadata.get("correctness", False))
            else:
                correct = float(rec.get("reward", 0.0)) >= 0.99
            metrics["gen_correctness"].append(1.0 if correct else 0.0)

    return metrics


def _eval_mean_metric(
    metric: str,
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    alpha: float,
    num_resamples: int,
    seed: int,
) -> MeanDiffTestResult:
    b_mean = _mean(baseline_values)
    c_mean = _mean(candidate_values)
    delta = c_mean - b_mean
    p_val = _permutation_p_value(
        baseline_values,
        candidate_values,
        num_resamples=num_resamples,
        seed=seed,
    )
    ci_low, ci_high = _bootstrap_ci(
        baseline_values,
        candidate_values,
        num_resamples=num_resamples,
        seed=seed + 17,
    )
    pooled = _pooled_sd(baseline_values, candidate_values)
    d = (delta / pooled) if pooled > 1e-12 else 0.0
    significant = p_val < alpha
    significant_improvement = significant and delta > 0.0 and ci_low > 0.0
    return MeanDiffTestResult(
        metric=metric,
        baseline_mean=b_mean,
        candidate_mean=c_mean,
        delta=delta,
        p_value=p_val,
        ci_low=ci_low,
        ci_high=ci_high,
        n_baseline=len(baseline_values),
        n_candidate=len(candidate_values),
        significant=significant,
        significant_improvement=significant_improvement,
        cohens_d=d,
    )


def evaluate_sepa_significance(
    baseline_run_dirs: Sequence[PathLike],
    candidate_run_dirs: Sequence[PathLike],
    *,
    alpha: float = 0.05,
    num_resamples: int = 20000,
    seed: int = 0,
) -> SEPASignificanceReport:
    """Evaluate statistical significance for SEPA-vs-baseline run groups."""
    if not baseline_run_dirs:
        raise ValueError("baseline_run_dirs must not be empty.")
    if not candidate_run_dirs:
        raise ValueError("candidate_run_dirs must not be empty.")
    if num_resamples < 100:
        raise ValueError("num_resamples should be >= 100 for stable estimates.")

    baseline = _collect_metrics(baseline_run_dirs)
    candidate = _collect_metrics(candidate_run_dirs)

    mean_tests = [
        _eval_mean_metric(
            "step_mean_reward",
            baseline["step_mean_reward"],
            candidate["step_mean_reward"],
            alpha=alpha,
            num_resamples=num_resamples,
            seed=seed + 11,
        ),
        _eval_mean_metric(
            "gen_reward",
            baseline["gen_reward"],
            candidate["gen_reward"],
            alpha=alpha,
            num_resamples=num_resamples,
            seed=seed + 23,
        ),
        _eval_mean_metric(
            "gen_gram_entropy_delta",
            baseline["gen_gram_entropy_delta"],
            candidate["gen_gram_entropy_delta"],
            alpha=alpha,
            num_resamples=num_resamples,
            seed=seed + 37,
        ),
    ]

    base_total = len(baseline["gen_correctness"])
    cand_total = len(candidate["gen_correctness"])
    base_success = int(sum(baseline["gen_correctness"]))
    cand_success = int(sum(candidate["gen_correctness"]))
    base_rate = (base_success / base_total) if base_total > 0 else 0.0
    cand_rate = (cand_success / cand_total) if cand_total > 0 else 0.0
    rate_delta = cand_rate - base_rate
    fisher_p = _fisher_exact_two_sided(
        baseline_success=base_success,
        baseline_fail=max(base_total - base_success, 0),
        candidate_success=cand_success,
        candidate_fail=max(cand_total - cand_success, 0),
    )
    rate_ci_low, rate_ci_high = _normal_ci_rate_delta(
        base_rate, base_total, cand_rate, cand_total
    )
    rate_significant = fisher_p < alpha
    rate_significant_improvement = (
        rate_significant and rate_delta > 0.0 and rate_ci_low > 0.0
    )
    rate_test = RateDiffTestResult(
        metric="gen_correct_rate",
        baseline_rate=base_rate,
        candidate_rate=cand_rate,
        delta=rate_delta,
        p_value=fisher_p,
        ci_low=rate_ci_low,
        ci_high=rate_ci_high,
        baseline_success=base_success,
        baseline_total=base_total,
        candidate_success=cand_success,
        candidate_total=cand_total,
        significant=rate_significant,
        significant_improvement=rate_significant_improvement,
    )

    sample_size_estimates = [
        SampleSizeEstimate(
            metric="step_mean_reward",
            observed_effect=mean_tests[0].delta,
            n_per_arm_80pct=_required_n_per_arm_for_mean(
                baseline["step_mean_reward"], candidate["step_mean_reward"]
            ),
            unit="steps",
        ),
        SampleSizeEstimate(
            metric="gen_reward",
            observed_effect=mean_tests[1].delta,
            n_per_arm_80pct=_required_n_per_arm_for_mean(
                baseline["gen_reward"], candidate["gen_reward"]
            ),
            unit="generations",
        ),
        SampleSizeEstimate(
            metric="gen_gram_entropy_delta",
            observed_effect=mean_tests[2].delta,
            n_per_arm_80pct=_required_n_per_arm_for_mean(
                baseline["gen_gram_entropy_delta"], candidate["gen_gram_entropy_delta"]
            ),
            unit="gram_pairs",
        ),
        SampleSizeEstimate(
            metric="gen_correct_rate",
            observed_effect=rate_test.delta,
            n_per_arm_80pct=_required_n_per_arm_for_rate(base_rate, cand_rate),
            unit="generations",
        ),
    ]

    significant_improvement_metrics = [
        test.metric for test in mean_tests if test.significant_improvement
    ]
    if rate_test.significant_improvement:
        significant_improvement_metrics.append(rate_test.metric)

    if (
        mean_tests[0].significant_improvement
        and rate_test.significant_improvement
    ):
        recommendation = "statistically_significant_improvement"
    else:
        recommendation = "insufficient_statistical_evidence"

    return SEPASignificanceReport(
        alpha=float(alpha),
        baseline_runs=[str(Path(p).expanduser().resolve()) for p in baseline_run_dirs],
        candidate_runs=[str(Path(p).expanduser().resolve()) for p in candidate_run_dirs],
        mean_tests=mean_tests,
        rate_test=rate_test,
        sample_size_estimates=sample_size_estimates,
        significant_improvement_metrics=significant_improvement_metrics,
        recommendation=recommendation,
    )


def build_sepa_significance_markdown(report: SEPASignificanceReport) -> str:
    """Render a compact markdown report."""
    lines: List[str] = []
    lines.append("# SEPA Significance Report")
    lines.append("")
    lines.append(f"- alpha: {report.alpha}")
    lines.append(f"- recommendation: {report.recommendation}")
    lines.append(
        f"- significant improvement metrics: "
        f"{', '.join(report.significant_improvement_metrics) if report.significant_improvement_metrics else 'none'}"
    )
    lines.append("")
    lines.append("## Mean-Difference Tests")
    lines.append("| metric | baseline mean | candidate mean | delta | p-value | 95% CI | significant |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for test in report.mean_tests:
        lines.append(
            f"| {test.metric} | {test.baseline_mean:.6f} | {test.candidate_mean:.6f} | "
            f"{test.delta:+.6f} | {test.p_value:.4f} | "
            f"[{test.ci_low:+.6f}, {test.ci_high:+.6f}] | "
            f"{'yes' if test.significant_improvement else 'no'} |"
        )
    lines.append("")
    rt = report.rate_test
    lines.append("## Correctness Rate Test")
    lines.append(
        f"- baseline: {rt.baseline_rate:.6f} ({rt.baseline_success}/{rt.baseline_total})"
    )
    lines.append(
        f"- candidate: {rt.candidate_rate:.6f} ({rt.candidate_success}/{rt.candidate_total})"
    )
    lines.append(f"- delta: {rt.delta:+.6f}")
    lines.append(f"- p-value (Fisher exact): {rt.p_value:.4f}")
    lines.append(f"- 95% CI (normal approx): [{rt.ci_low:+.6f}, {rt.ci_high:+.6f}]")
    lines.append(
        f"- significant improvement: {'yes' if rt.significant_improvement else 'no'}"
    )
    lines.append("")
    lines.append("## Sample Size Estimates (80% power, rough)")
    lines.append("| metric | observed effect | n per arm | unit |")
    lines.append("|---|---:|---:|---|")
    for est in report.sample_size_estimates:
        n_txt = "n/a" if est.n_per_arm_80pct is None else str(est.n_per_arm_80pct)
        lines.append(
            f"| {est.metric} | {est.observed_effect:+.6f} | {n_txt} | {est.unit} |"
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "MeanDiffTestResult",
    "RateDiffTestResult",
    "SampleSizeEstimate",
    "SEPASignificanceReport",
    "evaluate_sepa_significance",
    "build_sepa_significance_markdown",
]
