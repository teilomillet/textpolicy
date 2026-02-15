#!/usr/bin/env python3
"""Generate SEPA diagnostic figure: surprisal distributions before/after pooling.

Uses per-token entropy data from sepa_campaign_final_maxrl_20260213.
Reconstructs planning mask via strategic gram detection, then shows:
  - Panel A: Raw surprisal distributions (planning vs execution), lambda=0
  - Panel B: After SEPA pooling (lambda=1), same tokens

No new compute — just reanalysis of existing logs.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CAMPAIGN_DIR = Path(
    "/Users/teilomillet/Code/textpolicy/results/sepa_campaign_final_maxrl_20260213"
)
OUTPUT_PATH = Path("/Users/teilomillet/Code/textpolicy/paper/sepa_diagnostic.pdf")

# Strategic grams (same as in the training code)
STRATEGIC_GRAMS = [
    "wait let me",
    "let me think",
    "on second thought",
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    "another way to",
    "or we could",
    "what if we",
    "notice that",
    "the key is",
    "the key insight",
]

# Compile regex patterns (word boundary matching)
GRAM_PATTERNS = [
    re.compile(r"\b" + re.escape(gram) + r"\b", re.IGNORECASE)
    for gram in STRATEGIC_GRAMS
]


def detect_planning_mask(completion: str, n_tokens: int) -> list[int]:
    """Simple planning mask: mark tokens covered by strategic gram spans.

    Since we don't have the tokenizer, we approximate by character-level
    detection and map back to token positions proportionally.
    """
    # Find all strategic gram character spans
    planning_chars = set()
    for pattern in GRAM_PATTERNS:
        for m in pattern.finditer(completion):
            for i in range(m.start(), m.end()):
                planning_chars.add(i)

    if not planning_chars or n_tokens == 0:
        return [0] * n_tokens

    # Map character positions to token positions (proportional)
    chars_per_token = max(len(completion) / n_tokens, 1)
    mask = [0] * n_tokens
    for char_pos in planning_chars:
        tok_idx = min(int(char_pos / chars_per_token), n_tokens - 1)
        mask[tok_idx] = 1
    return mask


def load_per_token_data(
    campaign_dir: Path,
    prefix: str = "candidate",
    max_seeds: int = 16,
) -> list[dict]:
    """Load all per-token entropy data from generations.jsonl."""
    records = []
    for seed_dir in sorted(campaign_dir.iterdir()):
        if not seed_dir.name.startswith(prefix):
            continue
        gen_file = seed_dir / "emergence" / "generations.jsonl"
        if not gen_file.exists():
            continue
        with open(gen_file) as f:
            for line in f:
                d = json.loads(line)
                records.append(d)
        if len(set(r["step"] for r in records)) > 0 and len(records) > 5000:
            break  # enough data
    return records


def apply_sepa_pooling(
    entropies: list[float],
    planning_mask: list[int],
    lambda_t: float,
) -> list[float]:
    """Apply SEPA pooling: pull execution token entropies toward their mean."""
    exec_entropies = [h for h, m in zip(entropies, planning_mask) if not m]
    if not exec_entropies:
        return list(entropies)
    mean_h_exec = sum(exec_entropies) / len(exec_entropies)
    result = []
    for h, m in zip(entropies, planning_mask):
        if m:
            result.append(h)
        else:
            result.append(lambda_t * mean_h_exec + (1.0 - lambda_t) * h)
    return result


def main():
    print("Loading per-token entropy data...")
    records = load_per_token_data(CAMPAIGN_DIR, prefix="candidate")
    print(f"  Loaded {len(records)} generations")

    # Collect per-token values from ALL steps (lambda=0 perspective)
    # We use the raw entropy_per_token and reconstruct what SEPA would do
    plan_before = []
    exec_before = []
    plan_after = []
    exec_after = []

    n_plan_tokens = 0
    n_exec_tokens = 0
    n_gens_with_planning = 0

    for rec in records:
        ept = rec.get("entropy_per_token", [])
        completion = rec.get("completion", "")
        if not ept or not completion:
            continue

        n_tokens = len(ept)
        mask = detect_planning_mask(completion, n_tokens)

        has_planning = any(m == 1 for m in mask)
        if has_planning:
            n_gens_with_planning += 1

        # Before SEPA (raw)
        for h, m in zip(ept, mask):
            if h < 0:
                h = 0.0  # clamp -0.0
            if m:
                plan_before.append(h)
                n_plan_tokens += 1
            else:
                exec_before.append(h)
                n_exec_tokens += 1

        # After SEPA (lambda=1)
        pooled = apply_sepa_pooling(ept, mask, lambda_t=1.0)
        for h, m in zip(pooled, mask):
            if h < 0:
                h = 0.0
            if m:
                plan_after.append(h)
            else:
                exec_after.append(h)

    plan_before = np.array(plan_before)
    exec_before = np.array(exec_before)
    plan_after = np.array(plan_after)
    exec_after = np.array(exec_after)

    print(f"  Planning tokens: {n_plan_tokens:,}")
    print(f"  Execution tokens: {n_exec_tokens:,}")
    print(f"  Generations with planning phrases: {n_gens_with_planning}/{len(records)}")
    print(f"  Planning token ratio: {n_plan_tokens/(n_plan_tokens+n_exec_tokens):.1%}")

    print(f"\n  Before SEPA:")
    print(f"    Exec mean={exec_before.mean():.3f}, var={exec_before.var():.4f}")
    print(f"    Plan mean={plan_before.mean():.3f}, var={plan_before.var():.4f}")
    print(f"  After SEPA (lambda=1):")
    print(f"    Exec mean={exec_after.mean():.3f}, var={exec_after.var():.4f}")
    print(f"    Plan mean={plan_after.mean():.3f}, var={plan_after.var():.4f}")
    print(f"  Exec variance reduction: {(1 - exec_after.var()/exec_before.var())*100:.1f}%")

    # -----------------------------------------------------------------------
    # Figure: 2-panel histogram
    # -----------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    bins = np.linspace(0, 4.0, 60)
    alpha = 0.65

    # Panel A: Before SEPA
    ax1.hist(exec_before, bins=bins, alpha=alpha, color="#4C72B0",
             label=f"Execution (n={n_exec_tokens:,})", density=True)
    ax1.hist(plan_before, bins=bins, alpha=alpha, color="#DD8452",
             label=f"Planning (n={n_plan_tokens:,})", density=True)
    ax1.set_xlabel("Surprisal $S(t) = -\\log\\, p_\\theta(t)$")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Before SEPA ($\\lambda = 0$)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 4.0)

    # Annotate exec variance
    ax1.text(0.97, 0.95,
             f"Exec var = {exec_before.var():.3f}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#4C72B0", alpha=0.15))

    # Panel B: After SEPA
    ax2.hist(exec_after, bins=bins, alpha=alpha, color="#4C72B0",
             label="Execution (pooled)", density=True)
    ax2.hist(plan_after, bins=bins, alpha=alpha, color="#DD8452",
             label="Planning (unchanged)", density=True)
    ax2.set_xlabel("Surprisal $S(t)$")
    ax2.set_title("(b) After SEPA ($\\lambda = 1$)")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 4.0)

    # Annotate exec variance
    reduction = (1 - exec_after.var() / exec_before.var()) * 100
    ax2.text(0.97, 0.95,
             f"Exec var = {exec_after.var():.3f}\n({reduction:.0f}% reduction)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#4C72B0", alpha=0.15))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=150)
    print(f"\nFigure saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
