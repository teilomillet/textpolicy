# textpolicy/algorithms/hicra.py
"""
HICRA — Planning Token Amplification for MLX.

HICRA (High-Impact Credit Re-Assignment) amplifies advantage signal at
tokens that participate in strategic reasoning phrases (hesitation,
verification, backtracking, etc.).  This is a content-based alternative
to GTPO's entropy-based proxy.

Core formula (per-token):
    A_HICRA(t) = A(t) + alpha * |A(t)| * planning_mask(t)

Sign behavior:
    A > 0, mask=1 → A*(1+alpha)  — good planning amplified
    A < 0, mask=1 → A*(1-alpha)  — bad planning gets less blame
    alpha=0 or mask=0 → unchanged

The planning mask is detached via mx.stop_gradient to prevent the model
from gaming advantage signal through token content.

All functions are pure, stateless, and follow the flat 1D data layout
invariant of the training pipeline.

Subword tokenization limitation:
    Planning token identification uses ``convert_ids_to_tokens()`` to
    reconstruct text from subword fragments, joining them with spaces.
    This works correctly when each word in a strategic gram maps to a
    single token (the common case for standard LLM tokenizers and the
    default grams, which use only common English words like "let",
    "think", "approach", etc.).

    However, if a tokenizer aggressively splits a gram's words into
    multiple subwords — e.g. "another" → ["an", "other"] — the
    reconstructed text becomes "an other" (two words), which will NOT
    match the gram regex ``\\banother\\b``.  This is a known limitation
    of text-reconstruction-based matching.  For custom grams containing
    rare or domain-specific words, verify that your tokenizer keeps them
    as single tokens, or use ``tokenizer.decode()`` externally and pass
    pre-computed masks instead.

References:
    Issue #11 — HICRA Planning Token Amplification
    Issue #15 — Strategic Gram Mining Pipeline (produces the vocabulary)
"""

from __future__ import annotations

import re
from typing import List, Optional, Union

try:
    import mlx.core as mx  # type: ignore
except ImportError:
    # Unlike grpo.py, this module has no @mx.compile decorators that
    # need a dummy at import time.  Raise immediately so the error
    # message is clear instead of a confusing AttributeError later.
    raise ImportError(
        "MLX is required for HICRA. Install it with: pip install mlx"
    )


# ---------------------------------------------------------------------------
# 1. Token identification — sliding window over decoded tokens
# ---------------------------------------------------------------------------

def identify_planning_tokens(
    token_ids: mx.array,
    tokenizer,
    strategic_grams: List[str],
    max_window: int = 5,
) -> mx.array:
    """
    Produce a binary mask marking tokens that participate in strategic grams.

    Uses ``tokenizer.convert_ids_to_tokens()`` to decode token IDs, then
    slides a window of width [1, max_window] over the decoded fragments.
    Matching is word-boundary-aware (``\\b`` regex anchors) and case-
    insensitive.  When a match is found, only the tokens in the *shortest*
    matching window are marked (the window stops expanding on first match
    to avoid over-marking adjacent tokens).

    Sliding window behavior:
        For each start position, the window expands one token at a time.
        On the first gram match, only tokens [start, end] are marked and
        expansion stops.  This "shortest match first" strategy ensures
        that non-participating tokens adjacent to a match are not falsely
        included.  Subsequent start positions will independently attempt
        matches from their own position.

    Subword tokenization:
        Decoded token fragments are joined with spaces after stripping
        common subword prefixes (``▁`` for SentencePiece, ``Ġ`` for
        GPT-style BPE).  This works correctly when each word in a gram
        is a single token — the common case for standard LLM tokenizers
        (LLaMA, GPT, Mistral) with the default strategic grams, which
        use only common English words (max 8 chars: "approach").

        **Limitation:** If a tokenizer splits a gram word into multiple
        subwords (e.g. "another" → ["an", "other"]), the reconstructed
        window text becomes "an other" (two words), which will not match
        the regex ``\\banother\\b``.  Increasing ``max_window`` does NOT
        fix this — the issue is word reconstruction, not window size.
        For custom grams with rare words, verify your tokenizer keeps
        them as single tokens.

    Args:
        token_ids: Flat 1D token IDs [total_tokens].
        tokenizer: HuggingFace-compatible tokenizer with
                   ``convert_ids_to_tokens()``.
        strategic_grams: List of multi-word phrases to search for.
        max_window: Maximum sliding window size in tokens.  Automatically
                    raised to at least the longest gram's word count so
                    that no gram is silently missed.  The default (5)
                    gives headroom for 3-word grams where some words
                    occupy 2 tokens (e.g. a word with a space prefix
                    token).

    Returns:
        Binary mask [total_tokens] as float32 mx.array (0.0 or 1.0).
    """
    n_tokens = token_ids.shape[0] if token_ids.ndim > 0 else 0
    if n_tokens == 0 or not strategic_grams:
        return mx.zeros((n_tokens,), dtype=mx.float32)

    # Ensure window is large enough for the longest gram.  Each word in
    # a gram may map to multiple subword tokens, so we use the word count
    # as a floor (the caller's explicit max_window adds headroom).
    max_gram_words = max(len(g.split()) for g in strategic_grams)
    max_window = max(max_window, max_gram_words)

    # Decode token IDs to string fragments
    ids_list = token_ids.tolist()
    tokens_str = tokenizer.convert_ids_to_tokens(ids_list)

    # Build a Python mask (faster than repeated mx.array updates)
    mask = [0] * n_tokens

    # Pre-compile word-boundary regex for each gram so that e.g.
    # "key insight" does not match inside "monkey insight".
    #
    # Escape-level note (why r"\\s+" is correct here):
    #   re.escape("let me") → 'let\\ me'   (backslash-space between words)
    #   re.sub(r"\\ ", r"\\s+", ...)        (replace backslash-space with \s+)
    #     - replacement r"\\s+" is the 4-char string: \, \, s, +
    #     - re.sub processes \\ as "emit literal backslash", then s+ as literal
    #     - output: \s+  (the regex whitespace-run quantifier)
    #   Final pattern: \blet\s+me\b
    #
    # A single-backslash replacement r"\s+" (3 chars) would crash with
    # re.error in Python ≥ 3.12, but that is NOT what this code does.
    gram_patterns = []
    for g in strategic_grams:
        escaped = re.escape(g.lower())
        escaped = re.sub(r"\\ ", r"\\s+", escaped)
        gram_patterns.append(re.compile(rf"\b{escaped}\b", re.IGNORECASE))

    # Sliding window: for each start position, expand the window up to
    # max_window tokens.  When a match is found, mark only the tokens in
    # the *current* window span and stop expanding (greedy shortest match
    # from this start position).
    for start in range(n_tokens):
        window_text = ""
        for end in range(start, min(start + max_window, n_tokens)):
            # Accumulate text; many tokenizers use '▁' or 'Ġ' for space
            fragment = tokens_str[end]
            if fragment is None:
                fragment = ""
            # Strip common subword prefixes and join with space
            cleaned = fragment.replace("▁", " ").replace("Ġ", " ").strip()
            if cleaned:
                if window_text:
                    window_text += " " + cleaned
                else:
                    window_text = cleaned

            matched = False
            for pattern in gram_patterns:
                if pattern.search(window_text):
                    # Mark only the tokens in this window span
                    for idx in range(start, end + 1):
                        mask[idx] = 1
                    matched = True
                    break
            if matched:
                break  # Stop expanding window from this start position

    return mx.array(mask, dtype=mx.float32)


# ---------------------------------------------------------------------------
# 2. Advantage amplification — pure MLX, mx.compile-safe
# ---------------------------------------------------------------------------

def apply_hicra_amplification(
    advantages: mx.array,
    planning_mask: mx.array,
    alpha: float = 0.2,
) -> mx.array:
    """
    Amplify advantages at planning tokens using the HICRA formula.

    Formula:
        A_HICRA(t) = A(t) + alpha * |A(t)| * planning_mask(t)

    This has deliberate asymmetric sign behavior:
        A > 0, mask=1 → A + alpha*A = A*(1+alpha)  — amplified
        A < 0, mask=1 → A + alpha*|A| = A + alpha*(-A) = A*(1-alpha) — dampened blame
        alpha=0 or mask=0 → A (unchanged)

    The planning mask is detached from the backward pass via
    ``mx.stop_gradient`` so the model cannot learn to game advantage
    signal through token content.

    Args:
        advantages: Flat 1D advantages [total_tokens].
        planning_mask: Binary mask [total_tokens], float32 (0.0 or 1.0).
                      Must be the same shape as *advantages*.
        alpha: Amplification factor (default 0.2). 0 disables amplification.

    Returns:
        Amplified advantages [total_tokens], same shape as input.

    Raises:
        ValueError: If shapes don't match.
    """
    if advantages.shape != planning_mask.shape:
        raise ValueError(
            f"Shape mismatch: advantages {advantages.shape} vs "
            f"planning_mask {planning_mask.shape}. Both must match."
        )

    if alpha == 0.0:
        return advantages

    # Detach mask from gradient to prevent gaming
    mask = mx.stop_gradient(planning_mask)

    # A_HICRA(t) = A(t) + alpha * |A(t)| * mask(t)
    return advantages + alpha * mx.abs(advantages) * mask


# ---------------------------------------------------------------------------
# 2b. Entropy boosting — inject planning signal into entropy for GTPO fusion
# ---------------------------------------------------------------------------

def boost_entropy_with_planning(
    token_entropies: mx.array,
    planning_mask: mx.array,
    gamma: float = 0.3,
) -> mx.array:
    """
    Boost token entropies at planning positions for GTPO fusion.

    Instead of applying HICRA as a post-processing step on advantages,
    this function injects the planning signal *upstream* into the entropy
    array.  When GTPO's O+/O- machinery (Eq. 3, 5) processes the boosted
    entropies, planning tokens naturally receive:

    - **More credit in O+** (Eq. 3): higher H → larger ``H_i,t / ΣH`` share
    - **Less blame in O-** (Eq. 5): higher H → smaller ``1/H`` penalty

    Formula:
        H_boosted(t) = H(t) + gamma * planning_mask(t) * mean(H)

    The mean(H) scaling makes the boost magnitude adapt to the entropy
    distribution of the current batch, avoiding fragile absolute thresholds.

    The planning mask is detached via ``mx.stop_gradient`` to prevent the
    model from gaming entropy through token content (same rationale as
    Remark 2.5 in the GTPO paper).

    All operations are pure MLX and ``mx.compile``-safe.

    Args:
        token_entropies: Flat 1D per-token entropy [total_tokens].
        planning_mask: Binary mask [total_tokens], float32 (0.0 or 1.0).
                      Must be the same shape as *token_entropies*.
        gamma: Boost factor (default 0.3).  0 disables boosting.

    Returns:
        Boosted entropies [total_tokens], same shape as input.

    Raises:
        ValueError: If shapes don't match.
    """
    if gamma < 0.0:
        raise ValueError(
            f"gamma must be >= 0, got {gamma}. Negative gamma reduces "
            f"entropy at planning tokens, which can produce negative "
            f"values that break inverse-entropy shaping (Eq. 5)."
        )

    if token_entropies.shape != planning_mask.shape:
        raise ValueError(
            f"Shape mismatch: token_entropies {token_entropies.shape} vs "
            f"planning_mask {planning_mask.shape}. Both must match."
        )

    if gamma == 0.0:
        return token_entropies

    # Detach mask from gradient to prevent gaming (Remark 2.5)
    mask = mx.stop_gradient(planning_mask)

    # H_boosted(t) = H(t) + gamma * mask(t) * mean(H)
    mean_entropy = mx.mean(token_entropies)
    return token_entropies + gamma * mask * mean_entropy


# ---------------------------------------------------------------------------
# 3. Convenience composition
# ---------------------------------------------------------------------------

def compute_advantages_hicra(
    rewards: Union[List[float], mx.array],
    token_ids: mx.array,
    tokenizer,
    strategic_grams: List[str],
    alpha: float = 0.2,
    episode_lengths: Optional[List[int]] = None,
    token_entropies: Optional[mx.array] = None,
    entropy_weight: float = 0.0,
) -> mx.array:
    """
    Compute token-level advantages with HICRA amplification.

    Optionally composes with GTPO (entropy-weighted credit assignment):
    when *token_entropies* is provided and *entropy_weight* > 0, GTPO
    weighting is applied **first**, then HICRA amplification on top.

    Order: base GRPO → (optional GTPO weighting) → HICRA amplification.

    Note: When used via the Trainer's ``advantage_transform_fn`` hook,
    advantage expansion is handled by the Trainer (using real
    ``episode_lengths`` from ``batch_data``).  This function's own
    expansion logic is for standalone use outside the Trainer.

    Args:
        rewards: Episode rewards for group-relative baseline.
        token_ids: Flat 1D token IDs [total_tokens] for planning detection.
        tokenizer: HF-compatible tokenizer.
        strategic_grams: List of strategic gram phrases.
        alpha: HICRA amplification factor (default 0.2).
        episode_lengths: Token count per episode for expanding episode-level
                        advantages to token-level.  Required when *rewards*
                        are episode-level (the typical case).  When None,
                        rewards are assumed to be already token-level —
                        a ``ValueError`` is raised if the count doesn't
                        match ``token_ids``.
        token_entropies: Per-token entropy [total_tokens] for GTPO composition.
                        If None, skips entropy weighting.
        entropy_weight: β parameter for GTPO (only used when token_entropies
                       is provided).

    Returns:
        Amplified token-level advantages [total_tokens].

    Raises:
        ValueError: If ``episode_lengths`` is provided but
            ``sum(episode_lengths) != len(token_ids)``, or if
            ``episode_lengths`` is None but ``len(rewards) != len(token_ids)``.
    """
    # Import grpo functions (avoid circular import at module level)
    from textpolicy.algorithms.grpo import (
        compute_advantages,
        apply_entropy_weighting,
    )

    # Step 1: Compute base group-relative advantages
    base_advantages = compute_advantages(rewards)

    # Step 2: Expand to token-level if needed
    if episode_lengths is not None:
        num_episodes = base_advantages.shape[0]
        if num_episodes != len(episode_lengths):
            raise ValueError(
                f"Number of episodes ({num_episodes}) does not match "
                f"episode_lengths ({len(episode_lengths)})"
            )
        expected_tokens = sum(episode_lengths)
        actual_tokens = token_ids.shape[0]
        if expected_tokens != actual_tokens:
            raise ValueError(
                f"sum(episode_lengths)={expected_tokens} does not match "
                f"token_ids length {actual_tokens}. "
                f"Episode boundaries and token_ids array must align."
            )
        if len(set(episode_lengths)) == 1:
            expanded = mx.repeat(base_advantages, episode_lengths[0])
        else:
            parts = []
            for i, length in enumerate(episode_lengths):
                parts.append(mx.repeat(base_advantages[i : i + 1], length))
            expanded = mx.concatenate(parts)
    else:
        # No episode_lengths → assume rewards are already token-level.
        # Validate that the counts actually line up.
        n_adv = base_advantages.shape[0]
        n_tok = token_ids.shape[0] if token_ids.ndim > 0 else 0
        if n_adv != n_tok:
            raise ValueError(
                f"Without episode_lengths, rewards must be token-level "
                f"aligned. Got {n_adv} advantages but {n_tok} token_ids. "
                f"Provide episode_lengths to expand episode-level "
                f"advantages to token-level."
            )
        expanded = base_advantages

    # Step 3: Optional GTPO entropy weighting (applied first)
    if token_entropies is not None and entropy_weight > 0.0:
        expanded = apply_entropy_weighting(expanded, token_entropies, entropy_weight)

    # Step 4: Identify planning tokens and apply HICRA amplification
    planning_mask = identify_planning_tokens(
        token_ids, tokenizer, strategic_grams
    )
    return apply_hicra_amplification(expanded, planning_mask, alpha=alpha)
