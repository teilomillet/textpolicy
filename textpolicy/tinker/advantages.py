# textpolicy/tinker/advantages.py
"""
Pure-Python advantage pipeline for Tinker GPU training.

Ports the core formulas from textpolicy's MLX-based algorithms into
plain Python (lists + math).  No tensor framework dependencies — the
output lists get packed directly into Tinker's Datum objects.

Functions:
    compute_maxrl_advantages — MaxRL inverse success-rate reweighting
    apply_gtpo_weighting     — GTPO entropy-weighted credit assignment
    apply_hicra              — HICRA planning token amplification
    apply_sepa_pooling       — SEPA selective entropy pooling
    identify_planning_tokens — Strategic gram detection via tokenizer

Source references (MLX originals, read-only):
    grpo.py:111-229    — compute_advantages_maxrl
    grpo.py:561-662    — apply_entropy_weighting
    hicra.py:194-239   — apply_hicra_amplification
    hicra.py:66-187    — identify_planning_tokens
    sepa.py:153-182    — SEPAController.apply
"""

from __future__ import annotations

import math
import re
from typing import List, Optional


# ---------------------------------------------------------------------------
# 0. Baseline GRPO advantages (simple reward centering)
# ---------------------------------------------------------------------------
# Source: Tinker cookbook rl_loop.py (the default)
#
# Formula:
#   A_i = r_i - mean(r)
#
# This is the simplest group-relative advantage: subtract the group mean.
# No reweighting by success rate (MaxRL), no token-level shaping (GTPO),
# no planning amplification (HICRA), no entropy pooling (SEPA).
# Used as the baseline arm in A/B comparisons.

def compute_grpo_advantages(rewards: List[float]) -> List[float]:
    """
    Compute vanilla GRPO advantages: simple reward centering.

    This is the Tinker cookbook default. Each completion's advantage is
    its reward minus the group mean.

    Args:
        rewards: Per-completion rewards for one prompt group.

    Returns:
        List of advantages, same length as rewards.
    """
    if not rewards:
        return []

    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


# ---------------------------------------------------------------------------
# 1. MaxRL advantages (per-group inverse success-rate reweighting)
# ---------------------------------------------------------------------------
# Source: grpo.py:111-229, compute_advantages_maxrl (global fallback path)
#
# Formula:
#   A_i = (r_i - mean(r)) / (mean(r) + eps)
#
# For binary rewards with K successes out of N rollouts:
#   Correct:   A = (N-K)/K   (large when K small → hard problem)
#   Incorrect: A = -1         (constant)
#
# When mean(r) ~ 0 (no correct rollout), all advantages are zero.

def compute_maxrl_advantages(
    rewards: List[float],
    eps: float = 1e-6,
) -> List[float]:
    """
    Compute MaxRL advantages: inverse success-rate reweighting.

    Args:
        rewards: Per-completion rewards for one prompt group.
                 Expected non-negative (typically binary {0, 1}).
        eps: Numerical stability constant.

    Returns:
        List of advantages, same length as rewards.

    References:
        Maximum Likelihood RL (Tajwar et al., 2026), Eq. 10.
    """
    if not rewards:
        return []

    n = len(rewards)
    mean_r = sum(rewards) / n

    # No signal: all advantages zero (nothing to learn from).
    if mean_r <= eps:
        return [0.0] * n

    return [(r - mean_r) / (mean_r + eps) for r in rewards]


# ---------------------------------------------------------------------------
# 2. GTPO entropy-weighted credit assignment
# ---------------------------------------------------------------------------
# Source: grpo.py:561-662, apply_entropy_weighting
#
# Formula:
#   H_norm(t) = H(t) / mean(H)
#   w(t) = max(0, 1 + β * (H_norm(t) - 1))
#   A_GTPO(t) = A(t) * w(t)
#
# High-entropy tokens (decision points) get amplified advantages;
# low-entropy tokens (routine execution) get dampened advantages.
# Weights clamped to [0, ∞) so advantage signs are never flipped.

def apply_gtpo_weighting(
    advantage: float,
    entropies: List[float],
    beta: float = 0.1,
) -> List[float]:
    """
    Apply GTPO-style entropy weighting to produce token-level advantages.

    Takes a scalar episode advantage and a per-token entropy list,
    returns per-token weighted advantages.

    Args:
        advantage: Scalar advantage for this completion (from MaxRL).
        entropies: Per-token entropies (e.g. -logprob as entropy proxy).
                   Length = number of completion tokens.
        beta: Entropy weighting strength. 0.0 disables weighting.

    Returns:
        List of token-level advantages, same length as entropies.

    References:
        GTPO (arXiv 2508.04349), simplified multiplicative variant.
        See grpo.py:561-662 for the MLX original.
    """
    n = len(entropies)
    if n == 0:
        return []

    # β=0 → uniform weighting (no entropy effect)
    if beta == 0.0:
        return [advantage] * n

    # Mean-normalize entropies
    mean_h = sum(entropies) / n

    # All-zero or near-zero entropy → uniform (no signal to weight by)
    if mean_h < 1e-7:
        return [advantage] * n

    token_advs = []
    for h in entropies:
        h_norm = h / (mean_h + 1e-8)
        # GTPO weight: clamped to non-negative
        weight = max(0.0, 1.0 + beta * (h_norm - 1.0))
        token_advs.append(advantage * weight)

    return token_advs


# ---------------------------------------------------------------------------
# 3. HICRA planning token amplification
# ---------------------------------------------------------------------------
# Source: hicra.py:194-239, apply_hicra_amplification
#
# Formula:
#   A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)
#
# Sign behavior:
#   A > 0, mask=1 → A*(1+alpha)  — good planning amplified
#   A < 0, mask=1 → A*(1-alpha)  — bad planning gets less blame
#   alpha=0 or mask=0 → unchanged

def apply_hicra(
    token_advs: List[float],
    planning_mask: List[int],
    alpha: float = 0.2,
) -> List[float]:
    """
    Amplify advantages at planning tokens using the HICRA formula.

    Args:
        token_advs: Per-token advantages (from GTPO weighting).
        planning_mask: Binary mask (0 or 1) per token. 1 = planning token.
        alpha: Amplification factor. 0 disables.

    Returns:
        Amplified advantages, same length as input.

    References:
        Issue #11 — HICRA Planning Token Amplification.
        See hicra.py:194-239 for the MLX original.
    """
    if len(token_advs) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: token_advs ({len(token_advs)}) vs "
            f"planning_mask ({len(planning_mask)})"
        )

    if alpha == 0.0:
        return list(token_advs)

    result = []
    for adv, mask in zip(token_advs, planning_mask):
        if mask:
            # A_HICRA(t) = A(t) + alpha * |A(t)| * 1
            result.append(adv + alpha * abs(adv))
        else:
            result.append(adv)
    return result


# ---------------------------------------------------------------------------
# 4. SEPA selective entropy pooling
# ---------------------------------------------------------------------------
# Source: sepa.py:153-182, SEPAController.apply
#
# For execution tokens (non-planning), pool entropy toward execution mean:
#   H_pooled(t) = lambda_t * mean(H_exec) + (1 - lambda_t) * H(t)
# Planning tokens are left unchanged.

def apply_sepa_pooling(
    entropies: List[float],
    planning_mask: List[int],
    lambda_t: float,
) -> List[float]:
    """
    Apply SEPA pooling: pull execution token entropies toward their mean.

    Planning tokens keep their original entropy; execution tokens are
    interpolated toward the execution-token mean. This reduces entropy
    variance in execution regions, making GTPO weighting focus more
    on planning tokens.

    Args:
        entropies: Per-token entropies.
        planning_mask: Binary mask (0 or 1). 1 = planning, 0 = execution.
        lambda_t: Pooling strength in [0, 1]. 0 = no pooling, 1 = full pool.

    Returns:
        Pooled entropies, same length as input.

    References:
        SEPA (Selective Entropy Pooling with Annealing).
        See sepa.py:153-182 for the MLX original.
    """
    if len(entropies) != len(planning_mask):
        raise ValueError(
            f"Length mismatch: entropies ({len(entropies)}) vs "
            f"planning_mask ({len(planning_mask)})"
        )

    lambda_t = max(0.0, min(float(lambda_t), 1.0))

    if lambda_t == 0.0:
        return list(entropies)

    # Compute execution-token mean entropy
    exec_entropies = [h for h, m in zip(entropies, planning_mask) if not m]
    if not exec_entropies:
        return list(entropies)

    mean_h_exec = sum(exec_entropies) / len(exec_entropies)

    result = []
    for h, m in zip(entropies, planning_mask):
        if m:
            # Planning tokens: unchanged
            result.append(h)
        else:
            # Execution tokens: interpolate toward mean
            result.append(lambda_t * mean_h_exec + (1.0 - lambda_t) * h)
    return result


# ---------------------------------------------------------------------------
# 5. Planning token identification (strategic gram matching)
# ---------------------------------------------------------------------------
# Source: hicra.py:66-187, identify_planning_tokens
#
# Decodes token IDs to text fragments, then slides a window to match
# strategic grams via word-boundary regex. Pure Python port — no MLX.

# Default strategic grams (mirrored from analysis/strategic_grams.py)
DEFAULT_STRATEGIC_GRAMS: List[str] = [
    # Hesitation
    "wait let me",
    "let me think",
    "on second thought",
    # Verification
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    # Backtracking
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    # Alternatives
    "another way to",
    "or we could",
    "what if we",
    # Metacognition
    "notice that",
    "the key is",
    "the key insight",
]


def identify_planning_tokens(
    token_ids: List[int],
    tokenizer,
    strategic_grams: Optional[List[str]] = None,
    max_window: int = 5,
) -> List[int]:
    """
    Produce a binary mask marking tokens that participate in strategic grams.

    Uses tokenizer.convert_ids_to_tokens() to decode, then slides a window
    matching strategic grams with word-boundary regex.

    Args:
        token_ids: Token ID list for one completion.
        tokenizer: HuggingFace-compatible tokenizer with
                   convert_ids_to_tokens().
        strategic_grams: Phrases to detect. Defaults to DEFAULT_STRATEGIC_GRAMS.
        max_window: Maximum sliding window size in tokens.

    Returns:
        Binary mask (list of 0/1 ints), same length as token_ids.

    References:
        See hicra.py:66-187 for the MLX original.
    """
    if strategic_grams is None:
        strategic_grams = DEFAULT_STRATEGIC_GRAMS

    n_tokens = len(token_ids)
    if n_tokens == 0 or not strategic_grams:
        return [0] * n_tokens

    # Ensure window covers longest gram
    max_gram_words = max(len(g.split()) for g in strategic_grams)
    max_window = max(max_window, max_gram_words)

    # Decode token IDs to string fragments
    tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

    mask = [0] * n_tokens

    # Pre-compile word-boundary regex for each gram
    # Same escape logic as hicra.py:151-155
    gram_patterns = []
    for g in strategic_grams:
        escaped = re.escape(g.lower())
        escaped = re.sub(r"\\ ", r"\\s+", escaped)
        gram_patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))

    # Sliding window: shortest-match-first strategy
    for start in range(n_tokens):
        window_text = ""
        for end in range(start, min(start + max_window, n_tokens)):
            fragment = tokens_str[end]
            if fragment is None:
                fragment = ""
            # Strip common subword prefixes
            cleaned = fragment.replace("\u2581", " ").replace("\u0120", " ").strip()
            if cleaned:
                if window_text:
                    window_text += " " + cleaned
                else:
                    window_text = cleaned

            matched = False
            for pattern in gram_patterns:
                if pattern.search(window_text):
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break

    return mask
