"""
SEPA vs baseline litmus evaluation with explicit thresholds and evidence gates.

This module consumes per-run emergence logs and answers:
"Do we have enough evidence, and did SEPA clear the configured thresholds?"
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class SEPALitmusThresholds:
    """Minimum metric lifts (candidate - baseline) required to pass."""

    min_mean_reward_lift: float = 0.01
    min_correct_rate_lift: float = 0.005
    min_gram_entropy_delta_lift: float = 0.01


@dataclass(frozen=True)
class SEPALitmusEvidence:
    """Minimum sample sizes required before any pass/fail claim is valid."""

    min_run_pairs: int = 1
    min_steps_per_run: int = 8
    min_generations_per_run: int = 32
    min_gram_pairs_per_run: int = 8


@dataclass(frozen=True)
class SEPALitmusProfile:
    """Named threshold+evidence bundle for repeatable comparisons."""

    name: str
    thresholds: SEPALitmusThresholds
    evidence: SEPALitmusEvidence


@dataclass(frozen=True)
class EmergenceRunStats:
    """Aggregated metrics for one experiment run directory."""

    run_dir: str
    num_steps: int
    num_generations: int
    gram_pairs: int
    mean_reward_steps: float
    overall_correct_rate: float
    mean_gram_entropy_delta: float
    mean_planning_ratio: float
    mean_sepa_lambda: Optional[float]


@dataclass(frozen=True)
class SEPALitmusGroupStats:
    """Weighted aggregate of multiple runs (baseline or candidate group)."""

    run_count: int
    total_steps: int
    total_generations: int
    total_gram_pairs: int
    mean_reward_steps: float
    overall_correct_rate: float
    mean_gram_entropy_delta: float
    mean_planning_ratio: float
    mean_sepa_lambda: Optional[float]


@dataclass(frozen=True)
class MetricCheck:
    """Single threshold check result."""

    metric: str
    baseline: float
    candidate: float
    delta: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class SEPALitmusResult:
    """Final litmus result, including evidence and threshold checks."""

    status: str  # CONFIRMED | FAILED | INCONCLUSIVE
    baseline: SEPALitmusGroupStats
    candidate: SEPALitmusGroupStats
    thresholds: SEPALitmusThresholds
    evidence: SEPALitmusEvidence
    checks: List[MetricCheck]
    evidence_ok: bool
    evidence_failures: List[str]
    baseline_runs: List[EmergenceRunStats]
    candidate_runs: List[EmergenceRunStats]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "baseline": asdict(self.baseline),
            "candidate": asdict(self.candidate),
            "thresholds": asdict(self.thresholds),
            "evidence": asdict(self.evidence),
            "checks": [asdict(check) for check in self.checks],
            "evidence_ok": self.evidence_ok,
            "evidence_failures": list(self.evidence_failures),
            "baseline_runs": [asdict(run) for run in self.baseline_runs],
            "candidate_runs": [asdict(run) for run in self.candidate_runs],
        }


OFFICIAL_SEPA_LITMUS_PROFILE = SEPALitmusProfile(
    name="official_v1",
    thresholds=SEPALitmusThresholds(
        min_mean_reward_lift=0.01,
        min_correct_rate_lift=0.005,
        min_gram_entropy_delta_lift=0.005,
    ),
    evidence=SEPALitmusEvidence(
        min_run_pairs=1,
        min_steps_per_run=8,
        min_generations_per_run=32,
        min_gram_pairs_per_run=8,
    ),
)


def get_sepa_litmus_profile(name: str = "official_v1") -> SEPALitmusProfile:
    """Return a named litmus profile."""
    normalized = str(name).strip().lower()
    if normalized in {"official_v1", "official", "default"}:
        return OFFICIAL_SEPA_LITMUS_PROFILE
    raise ValueError(
        f"Unknown SEPA litmus profile {name!r}. "
        "Supported profiles: official_v1."
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                rows.append(record)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _weighted_mean(pairs: Sequence[Tuple[float, float]], fallback: float = 0.0) -> float:
    numer = 0.0
    denom = 0.0
    for value, weight in pairs:
        if weight <= 0.0:
            continue
        numer += value * weight
        denom += weight
    if denom <= 0.0:
        return fallback
    return numer / denom


def load_emergence_run_stats(run_dir: PathLike) -> EmergenceRunStats:
    """Load one run directory and compute aggregate metrics from emergence JSONL."""
    root = Path(run_dir).expanduser().resolve()
    steps_path = root / "emergence" / "steps.jsonl"
    gens_path = root / "emergence" / "generations.jsonl"

    steps = _read_jsonl(steps_path)
    if not steps:
        raise ValueError(f"No step records found in {steps_path}")

    generations = _read_jsonl(gens_path)

    mean_reward_steps = _mean([_safe_float(step.get("mean_reward")) for step in steps])
    mean_planning_ratio = _mean(
        [_safe_float(step.get("planning_token_ratio")) for step in steps]
    )

    lambda_values = [
        _safe_float(step.get("sepa_lambda"))
        for step in steps
        if step.get("sepa_lambda") is not None
    ]
    mean_sepa_lambda: Optional[float] = _mean(lambda_values) if lambda_values else None

    gram_weighted: List[Tuple[float, float]] = []
    gram_deltas_unweighted: List[float] = []
    gram_pairs = 0
    for step in steps:
        delta = step.get("gram_entropy_delta")
        if delta is None:
            continue
        delta_f = _safe_float(delta)
        pair_count = max(_safe_int(step.get("gram_entropy_pair_count"), 1), 0)
        gram_pairs += pair_count
        gram_weighted.append((delta_f, float(pair_count)))
        gram_deltas_unweighted.append(delta_f)

    mean_gram_entropy_delta = _weighted_mean(
        gram_weighted,
        fallback=_mean(gram_deltas_unweighted),
    )

    total_correct = sum(_safe_int(step.get("correct_count")) for step in steps)
    total_count = sum(_safe_int(step.get("total_count")) for step in steps)

    correctness_values: List[bool] = []
    for rec in generations:
        metadata = rec.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if "correctness" not in metadata:
            continue
        correctness_values.append(bool(metadata.get("correctness")))

    if correctness_values:
        overall_correct_rate = (
            sum(1 for c in correctness_values if c) / len(correctness_values)
        )
    elif total_count > 0:
        overall_correct_rate = total_correct / total_count
    else:
        overall_correct_rate = 0.0

    return EmergenceRunStats(
        run_dir=str(root),
        num_steps=len(steps),
        num_generations=len(generations),
        gram_pairs=gram_pairs,
        mean_reward_steps=mean_reward_steps,
        overall_correct_rate=float(overall_correct_rate),
        mean_gram_entropy_delta=float(mean_gram_entropy_delta),
        mean_planning_ratio=mean_planning_ratio,
        mean_sepa_lambda=(
            float(mean_sepa_lambda) if mean_sepa_lambda is not None else None
        ),
    )


def aggregate_group_stats(runs: Sequence[EmergenceRunStats]) -> SEPALitmusGroupStats:
    """Aggregate multiple runs into one weighted group summary."""
    if not runs:
        raise ValueError("Cannot aggregate empty run list.")

    reward_pairs = [(run.mean_reward_steps, float(max(run.num_steps, 1))) for run in runs]
    correct_pairs = [
        (run.overall_correct_rate, float(max(run.num_generations, 1))) for run in runs
    ]
    gram_pairs = [(run.mean_gram_entropy_delta, float(max(run.gram_pairs, 0))) for run in runs]
    planning_pairs = [(run.mean_planning_ratio, float(max(run.num_steps, 1))) for run in runs]

    lambda_pairs = [
        (run.mean_sepa_lambda, float(max(run.num_steps, 1)))
        for run in runs
        if run.mean_sepa_lambda is not None
    ]

    return SEPALitmusGroupStats(
        run_count=len(runs),
        total_steps=sum(run.num_steps for run in runs),
        total_generations=sum(run.num_generations for run in runs),
        total_gram_pairs=sum(run.gram_pairs for run in runs),
        mean_reward_steps=_weighted_mean(reward_pairs, fallback=0.0),
        overall_correct_rate=_weighted_mean(correct_pairs, fallback=0.0),
        mean_gram_entropy_delta=_weighted_mean(
            gram_pairs,
            fallback=_mean([run.mean_gram_entropy_delta for run in runs]),
        ),
        mean_planning_ratio=_weighted_mean(planning_pairs, fallback=0.0),
        mean_sepa_lambda=(
            _weighted_mean(
                [(float(value), weight) for value, weight in lambda_pairs], fallback=0.0
            )
            if lambda_pairs
            else None
        ),
    )


def _build_metric_checks(
    baseline: SEPALitmusGroupStats,
    candidate: SEPALitmusGroupStats,
    thresholds: SEPALitmusThresholds,
) -> List[MetricCheck]:
    checks: List[MetricCheck] = []

    reward_delta = candidate.mean_reward_steps - baseline.mean_reward_steps
    checks.append(
        MetricCheck(
            metric="mean_reward_steps",
            baseline=baseline.mean_reward_steps,
            candidate=candidate.mean_reward_steps,
            delta=reward_delta,
            threshold=thresholds.min_mean_reward_lift,
            passed=reward_delta >= thresholds.min_mean_reward_lift,
        )
    )

    correct_delta = candidate.overall_correct_rate - baseline.overall_correct_rate
    checks.append(
        MetricCheck(
            metric="overall_correct_rate",
            baseline=baseline.overall_correct_rate,
            candidate=candidate.overall_correct_rate,
            delta=correct_delta,
            threshold=thresholds.min_correct_rate_lift,
            passed=correct_delta >= thresholds.min_correct_rate_lift,
        )
    )

    gram_delta = candidate.mean_gram_entropy_delta - baseline.mean_gram_entropy_delta
    checks.append(
        MetricCheck(
            metric="mean_gram_entropy_delta",
            baseline=baseline.mean_gram_entropy_delta,
            candidate=candidate.mean_gram_entropy_delta,
            delta=gram_delta,
            threshold=thresholds.min_gram_entropy_delta_lift,
            passed=gram_delta >= thresholds.min_gram_entropy_delta_lift,
        )
    )
    return checks


def _check_evidence(
    baseline_runs: Sequence[EmergenceRunStats],
    candidate_runs: Sequence[EmergenceRunStats],
    evidence: SEPALitmusEvidence,
) -> Tuple[bool, List[str]]:
    failures: List[str] = []

    if len(baseline_runs) < evidence.min_run_pairs:
        failures.append(
            f"Baseline runs={len(baseline_runs)} is below min_run_pairs={evidence.min_run_pairs}."
        )
    if len(candidate_runs) < evidence.min_run_pairs:
        failures.append(
            f"Candidate runs={len(candidate_runs)} is below min_run_pairs={evidence.min_run_pairs}."
        )

    for group_name, runs in (("baseline", baseline_runs), ("candidate", candidate_runs)):
        for run in runs:
            run_label = Path(run.run_dir).name
            if run.num_steps < evidence.min_steps_per_run:
                failures.append(
                    f"{group_name}:{run_label} has {run.num_steps} steps; "
                    f"needs >= {evidence.min_steps_per_run}."
                )
            if run.num_generations < evidence.min_generations_per_run:
                failures.append(
                    f"{group_name}:{run_label} has {run.num_generations} generations; "
                    f"needs >= {evidence.min_generations_per_run}."
                )
            if run.gram_pairs < evidence.min_gram_pairs_per_run:
                failures.append(
                    f"{group_name}:{run_label} has {run.gram_pairs} gram pairs; "
                    f"needs >= {evidence.min_gram_pairs_per_run}."
                )

    return (len(failures) == 0), failures


def evaluate_sepa_litmus(
    baseline_run_dirs: Sequence[PathLike],
    candidate_run_dirs: Sequence[PathLike],
    *,
    thresholds: Optional[SEPALitmusThresholds] = None,
    evidence: Optional[SEPALitmusEvidence] = None,
) -> SEPALitmusResult:
    """
    Evaluate whether candidate runs clear SEPA litmus thresholds vs baseline.

    Status semantics:
    - INCONCLUSIVE: evidence requirements were not met.
    - CONFIRMED: evidence was sufficient and all threshold checks passed.
    - FAILED: evidence was sufficient but one or more thresholds failed.
    """
    if not baseline_run_dirs:
        raise ValueError("baseline_run_dirs must contain at least one run directory.")
    if not candidate_run_dirs:
        raise ValueError("candidate_run_dirs must contain at least one run directory.")

    resolved_thresholds = thresholds or SEPALitmusThresholds()
    resolved_evidence = evidence or SEPALitmusEvidence()

    baseline_runs = [load_emergence_run_stats(path) for path in baseline_run_dirs]
    candidate_runs = [load_emergence_run_stats(path) for path in candidate_run_dirs]

    baseline_group = aggregate_group_stats(baseline_runs)
    candidate_group = aggregate_group_stats(candidate_runs)
    checks = _build_metric_checks(baseline_group, candidate_group, resolved_thresholds)

    evidence_ok, evidence_failures = _check_evidence(
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
        evidence=resolved_evidence,
    )

    if not evidence_ok:
        status = "INCONCLUSIVE"
    elif all(check.passed for check in checks):
        status = "CONFIRMED"
    else:
        status = "FAILED"

    return SEPALitmusResult(
        status=status,
        baseline=baseline_group,
        candidate=candidate_group,
        thresholds=resolved_thresholds,
        evidence=resolved_evidence,
        checks=checks,
        evidence_ok=evidence_ok,
        evidence_failures=evidence_failures,
        baseline_runs=baseline_runs,
        candidate_runs=candidate_runs,
    )


def build_litmus_markdown(result: SEPALitmusResult) -> str:
    """Render a compact markdown report for sharing with research teams."""
    lines: List[str] = []
    lines.append(f"# SEPA Litmus Result: {result.status}")
    lines.append("")
    lines.append("## Evidence")
    lines.append(
        f"- baseline runs: {result.baseline.run_count} "
        f"(steps={result.baseline.total_steps}, generations={result.baseline.total_generations}, "
        f"gram_pairs={result.baseline.total_gram_pairs})"
    )
    lines.append(
        f"- candidate runs: {result.candidate.run_count} "
        f"(steps={result.candidate.total_steps}, generations={result.candidate.total_generations}, "
        f"gram_pairs={result.candidate.total_gram_pairs})"
    )
    lines.append(f"- evidence ok: {result.evidence_ok}")
    if result.evidence_failures:
        lines.append("- evidence failures:")
        for failure in result.evidence_failures:
            lines.append(f"  - {failure}")
    lines.append("")
    lines.append("## Threshold Checks")
    for check in result.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(
            f"- [{marker}] {check.metric}: "
            f"baseline={check.baseline:.6f}, "
            f"candidate={check.candidate:.6f}, "
            f"delta={check.delta:.6f}, "
            f"threshold={check.threshold:.6f}"
        )
    lines.append("")
    lines.append("## Aggregate Metrics")
    lines.append(
        f"- baseline: mean_reward={result.baseline.mean_reward_steps:.6f}, "
        f"correct_rate={result.baseline.overall_correct_rate:.6f}, "
        f"gram_delta={result.baseline.mean_gram_entropy_delta:.6f}, "
        f"planning_ratio={result.baseline.mean_planning_ratio:.6f}, "
        f"sepa_lambda={result.baseline.mean_sepa_lambda if result.baseline.mean_sepa_lambda is not None else 'n/a'}"
    )
    lines.append(
        f"- candidate: mean_reward={result.candidate.mean_reward_steps:.6f}, "
        f"correct_rate={result.candidate.overall_correct_rate:.6f}, "
        f"gram_delta={result.candidate.mean_gram_entropy_delta:.6f}, "
        f"planning_ratio={result.candidate.mean_planning_ratio:.6f}, "
        f"sepa_lambda={result.candidate.mean_sepa_lambda if result.candidate.mean_sepa_lambda is not None else 'n/a'}"
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "SEPALitmusThresholds",
    "SEPALitmusEvidence",
    "SEPALitmusProfile",
    "OFFICIAL_SEPA_LITMUS_PROFILE",
    "get_sepa_litmus_profile",
    "EmergenceRunStats",
    "SEPALitmusGroupStats",
    "MetricCheck",
    "SEPALitmusResult",
    "load_emergence_run_stats",
    "aggregate_group_stats",
    "evaluate_sepa_litmus",
    "build_litmus_markdown",
]
