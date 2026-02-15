#!/usr/bin/env python3
"""Compute AUC over training curves and paired deltas with CIs.

Uses existing metrics.jsonl from pilot + lean campaigns.
No new compute — just reanalysis of existing data.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PILOT_DIR = Path(
    "/Users/teilomillet/Code/textpolicy/results/paper_campaign_pilot_20260214_152911"
)
LEAN_DIR = Path(
    "/Users/teilomillet/Code/textpolicy/results/paper_campaign_lean_20260214_172726"
)

# Conditions in the lean campaign (the 5 we care about)
LEAN_CONDITIONS = [
    "grpo_none",
    "grpo_gtpo_hicra",
    "grpo_gtpo_sepa",
    "maxrl_none",
    "maxrl_gtpo_sepa",
]

N_BOOTSTRAP = 10_000
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(campaign_dir: Path) -> dict[str, dict[int, list[float]]]:
    """Load per-step correct_rate from all runs.

    Returns: {condition: {step: [rate_seed1, rate_seed2, ...]}}
    """
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run_dir in sorted(campaign_dir.iterdir()):
        metrics_file = run_dir / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        # Parse condition name from directory (e.g. grpo_none_seed601)
        name = run_dir.name
        # Strip _seed\d+ suffix
        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        condition = parts[0]

        with open(metrics_file) as f:
            for line in f:
                row = json.loads(line)
                step = row["step"]
                rate = row["correct_rate"]
                data[condition][step].append(rate)
    return dict(data)


def compute_auc(steps: list[int], values: list[float]) -> float:
    """Trapezoidal AUC over (step, value) pairs."""
    if len(steps) < 2:
        return 0.0
    auc = 0.0
    for i in range(len(steps) - 1):
        dt = steps[i + 1] - steps[i]
        auc += 0.5 * (values[i] + values[i + 1]) * dt
    return auc


def per_seed_auc(
    data: dict[str, dict[int, list[float]]],
    condition: str,
    max_step: int,
) -> list[float]:
    """Compute AUC for each seed independently, up to max_step.

    Returns list of AUC values (one per seed).
    """
    step_data = data.get(condition, {})
    if not step_data:
        return []

    # Find steps up to max_step
    valid_steps = sorted(s for s in step_data if s <= max_step)
    if len(valid_steps) < 2:
        return []

    # How many seeds at each step?
    n_seeds = min(len(step_data[s]) for s in valid_steps)
    aucs = []
    for seed_idx in range(n_seeds):
        values = [step_data[s][seed_idx] for s in valid_steps]
        aucs.append(compute_auc(valid_steps, values))
    return aucs


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap mean and CI."""
    means = np.array(
        [RNG.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_delta_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI on the mean difference (b - a)."""
    # Pool all seeds and resample
    n = min(len(a), len(b))
    deltas = []
    for _ in range(n_boot):
        idx_a = RNG.integers(0, len(a), size=n)
        idx_b = RNG.integers(0, len(b), size=n)
        deltas.append(b[idx_b].mean() - a[idx_a].mean())
    deltas = np.array(deltas)
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(deltas, [alpha, 1 - alpha])
    return float(np.mean(deltas)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("AUC ANALYSIS: Training Curve Area Under the Curve")
    print("=" * 70)

    # Load data
    pilot_data = load_metrics(PILOT_DIR)
    lean_data = load_metrics(LEAN_DIR)

    # Merge: lean has 4 seeds, pilot has 2 seeds (different seeds)
    # For the 5 lean conditions, combine both campaigns
    merged: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cond in LEAN_CONDITIONS:
        # Pilot data (seeds 601, 602)
        if cond in pilot_data:
            for step, vals in pilot_data[cond].items():
                merged[cond][step].extend(vals)
        # Lean data (seeds 601, 602, 603, 604)
        if cond in lean_data:
            for step, vals in lean_data[cond].items():
                merged[cond][step].extend(vals)
    merged = dict(merged)

    # Find common max step across lean runs
    lean_max_steps = {}
    for cond in LEAN_CONDITIONS:
        if cond in lean_data:
            lean_max_steps[cond] = max(lean_data[cond].keys())
    lean_common_max = min(lean_max_steps.values()) if lean_max_steps else 0
    print(f"\nLean campaign: common max step = {lean_common_max}")
    print(f"Pilot campaign: all runs complete to step 19")

    # Per-seed counts
    print(f"\nSeeds per condition (merged, up to step {lean_common_max}):")
    for cond in LEAN_CONDITIONS:
        if cond in merged:
            n = min(len(merged[cond].get(s, [])) for s in range(lean_common_max + 1))
            print(f"  {cond}: {n} seeds")

    # -----------------------------------------------------------------------
    # AUC Analysis 1: Steps 0-10 (early training, where ranking appears)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUC over steps 0-10 (early training)")
    print("=" * 70)

    auc_10: dict[str, np.ndarray] = {}
    for cond in LEAN_CONDITIONS:
        aucs = per_seed_auc(merged, cond, max_step=10)
        if aucs:
            auc_10[cond] = np.array(aucs)

    # Print per-condition AUC
    print(f"\n{'Condition':<25} {'AUC mean':>10} {'95% CI':>20} {'N seeds':>8}")
    print("-" * 65)
    ranked = sorted(auc_10.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        mean, lo, hi = bootstrap_ci(aucs)
        print(f"{cond:<25} {mean:10.3f} [{lo:8.3f}, {hi:8.3f}] {len(aucs):8d}")

    # Paired deltas vs baseline (grpo_none)
    baseline_key = "grpo_none"
    if baseline_key in auc_10:
        print(f"\nPaired deltas vs {baseline_key}:")
        print(f"{'Comparison':<35} {'Delta':>8} {'95% CI':>20} {'Contains 0?':>12}")
        print("-" * 77)
        baseline = auc_10[baseline_key]
        for cond in LEAN_CONDITIONS:
            if cond == baseline_key or cond not in auc_10:
                continue
            delta, lo, hi = bootstrap_delta_ci(baseline, auc_10[cond])
            contains_zero = "yes" if lo <= 0 <= hi else "NO"
            print(f"{cond} - {baseline_key:<15} {delta:8.3f} [{lo:8.3f}, {hi:8.3f}] {contains_zero:>12}")

    # Key comparison: C8 vs C1 (full stack vs baseline)
    if "maxrl_gtpo_sepa" in auc_10 and baseline_key in auc_10:
        print(f"\nKey comparison: maxrl_gtpo_sepa (C8) vs grpo_none (C1):")
        delta, lo, hi = bootstrap_delta_ci(auc_10[baseline_key], auc_10["maxrl_gtpo_sepa"])
        print(f"  Delta = {delta:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]")

    # -----------------------------------------------------------------------
    # AUC Analysis 2: Steps 0-{lean_common_max} (full available data)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"AUC over steps 0-{lean_common_max} (full available lean data)")
    print("=" * 70)

    auc_full: dict[str, np.ndarray] = {}
    for cond in LEAN_CONDITIONS:
        aucs = per_seed_auc(merged, cond, max_step=lean_common_max)
        if aucs:
            auc_full[cond] = np.array(aucs)

    print(f"\n{'Condition':<25} {'AUC mean':>10} {'95% CI':>20} {'N seeds':>8}")
    print("-" * 65)
    ranked = sorted(auc_full.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        mean, lo, hi = bootstrap_ci(aucs)
        print(f"{cond:<25} {mean:10.3f} [{lo:8.3f}, {hi:8.3f}] {len(aucs):8d}")

    if baseline_key in auc_full:
        print(f"\nPaired deltas vs {baseline_key}:")
        print(f"{'Comparison':<35} {'Delta':>8} {'95% CI':>20} {'Contains 0?':>12}")
        print("-" * 77)
        baseline = auc_full[baseline_key]
        for cond in LEAN_CONDITIONS:
            if cond == baseline_key or cond not in auc_full:
                continue
            delta, lo, hi = bootstrap_delta_ci(baseline, auc_full[cond])
            contains_zero = "yes" if lo <= 0 <= hi else "NO"
            print(f"{cond} - {baseline_key:<15} {delta:8.3f} [{lo:8.3f}, {hi:8.3f}] {contains_zero:>12}")

    # -----------------------------------------------------------------------
    # AUC Analysis 3: Pilot only, steps 0-19 (full convergence)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUC over steps 0-19 (pilot campaign only, 2 seeds)")
    print("=" * 70)

    auc_pilot: dict[str, np.ndarray] = {}
    for cond in LEAN_CONDITIONS:
        aucs = per_seed_auc(pilot_data, cond, max_step=19)
        if aucs:
            auc_pilot[cond] = np.array(aucs)

    print(f"\n{'Condition':<25} {'AUC mean':>10} {'95% CI':>20} {'N seeds':>8}")
    print("-" * 65)
    ranked = sorted(auc_pilot.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        mean, lo, hi = bootstrap_ci(aucs)
        print(f"{cond:<25} {mean:10.3f} [{lo:8.3f}, {hi:8.3f}] {len(aucs):8d}")

    # -----------------------------------------------------------------------
    # Normalized AUC (divide by number of steps for interpretable %)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Normalized AUC (mean correct rate over training window)")
    print("=" * 70)

    print("\nSteps 0-10 (early training):")
    print(f"{'Condition':<25} {'Mean rate':>10} {'95% CI':>20}")
    print("-" * 57)
    ranked = sorted(auc_10.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        normed = aucs / 10.0  # 10 step-units
        mean, lo, hi = bootstrap_ci(normed)
        print(f"{cond:<25} {mean*100:9.1f}% [{lo*100:7.1f}%, {hi*100:7.1f}%]")

    print(f"\nSteps 0-{lean_common_max} (full lean window):")
    print(f"{'Condition':<25} {'Mean rate':>10} {'95% CI':>20}")
    print("-" * 57)
    ranked = sorted(auc_full.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        normed = aucs / float(lean_common_max)
        mean, lo, hi = bootstrap_ci(normed)
        print(f"{cond:<25} {mean*100:9.1f}% [{lo*100:7.1f}%, {hi*100:7.1f}%]")

    print("\nSteps 0-19 (pilot, full convergence):")
    print(f"{'Condition':<25} {'Mean rate':>10} {'95% CI':>20}")
    print("-" * 57)
    ranked = sorted(auc_pilot.items(), key=lambda x: -x[1].mean())
    for cond, aucs in ranked:
        normed = aucs / 19.0
        mean, lo, hi = bootstrap_ci(normed)
        print(f"{cond:<25} {mean*100:9.1f}% [{lo*100:7.1f}%, {hi*100:7.1f}%]")


if __name__ == "__main__":
    main()
