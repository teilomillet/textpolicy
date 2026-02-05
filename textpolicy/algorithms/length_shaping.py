# textpolicy/algorithms/length_shaping.py
"""
DAPO-style soft overlong penalties and length shaping utilities.

These utilities replace hard truncation with graduated penalties,
reducing training instability from length-based confusion.

References:
    DAPO: An Open-Source LLM Reinforcement Learning System at Scale
    https://arxiv.org/abs/2503.14476
"""

from __future__ import annotations

from typing import List, Dict, Union

try:
    import mlx.core as mx  # type: ignore
except ImportError:
    mx = None


def compute_length_penalty(
    sequence_length: int,
    max_length: int,
    cache_length: int = 100,
    max_penalty: float = 0.5
) -> float:
    """
    Compute soft penalty for sequences approaching max length.

    Instead of hard cutoffs for max sequence length (which cause truncation
    that looks like failure to the model), use graduated penalties within
    an interval before max_length.

    This reduces training instability from length-based confusion and helps
    the model learn to be concise without hard punishment.

    Args:
        sequence_length: Current sequence length
        max_length: Maximum allowed sequence length
        cache_length: Start penalizing this many tokens before max_length.
                     Must be positive.
        max_penalty: Maximum penalty at max_length (default 0.5)

    Returns:
        Penalty value (0.0 for normal lengths, up to -max_penalty at max_length)

    Example:
        With max_length=512, cache_length=100 (threshold=412):
        - length=400: penalty=0.0 (below threshold)
        - length=412: penalty=0.0 (at threshold, progress=0)
        - length=462: penalty=-0.25 (50/100 * 0.5)
        - length=512: penalty=-0.5 (at max)

    Raises:
        ValueError: If cache_length <= 0
    """
    if cache_length <= 0:
        raise ValueError(f"cache_length must be positive, got {cache_length}")

    threshold = max_length - cache_length

    if sequence_length < threshold:
        return 0.0

    # Linear penalty from 0 to max_penalty as we approach max
    progress = (sequence_length - threshold) / cache_length
    progress = min(1.0, progress)  # Clamp at 1.0

    return -max_penalty * progress


def apply_length_shaping(
    rewards: "mx.array",
    sequence_lengths: List[int],
    max_length: int,
    cache_length: int = 100,
    max_penalty: float = 0.5
) -> "mx.array":
    """
    Apply soft length penalties to rewards.

    Modifies rewards by adding graduated penalties for sequences that
    approach the maximum length. This provides a smoother learning signal
    than hard truncation.

    Args:
        rewards: Original rewards array [batch_size]
        sequence_lengths: List of sequence lengths for each episode
        max_length: Maximum allowed sequence length
        cache_length: Start penalizing this many tokens before max_length
        max_penalty: Maximum penalty at max_length

    Returns:
        Rewards with length penalties applied

    Example:
        >>> rewards = mx.array([1.0, 0.5, 0.0])
        >>> lengths = [400, 500, 520]  # max_length=512, cache_length=100
        >>> shaped = apply_length_shaping(rewards, lengths, 512)
        >>> # shaped ≈ [1.0, 0.06, -0.5]  # last one gets max penalty
    """
    penalties = mx.array([
        compute_length_penalty(length, max_length, cache_length, max_penalty)
        for length in sequence_lengths
    ], dtype=mx.float32)

    return rewards + penalties


def compute_length_shaping_stats(
    sequence_lengths: List[int],
    max_length: int,
    cache_length: int = 100
) -> Dict[str, Union[int, float]]:
    """
    Compute statistics about length penalties for monitoring.

    Args:
        sequence_lengths: List of sequence lengths
        max_length: Maximum allowed sequence length
        cache_length: Penalty threshold offset

    Returns:
        Dictionary with length penalty statistics:
        - mean_length: Average sequence length
        - max_length_observed: Maximum observed sequence length
        - truncation_rate: Fraction of sequences at or past max_length
        - penalty_zone_rate: Fraction of sequences in penalty zone
    """
    threshold = max_length - cache_length
    total = len(sequence_lengths)

    if total == 0:
        return {
            'mean_length': 0.0,
            'max_length_observed': 0,
            'truncation_rate': 0.0,
            'penalty_zone_rate': 0.0,
        }

    truncated = sum(1 for l in sequence_lengths if l >= max_length)
    in_penalty_zone = sum(1 for l in sequence_lengths if threshold <= l < max_length)

    return {
        'mean_length': sum(sequence_lengths) / total,
        'max_length_observed': max(sequence_lengths),
        'truncation_rate': truncated / total,
        'penalty_zone_rate': in_penalty_zone / total,
    }
