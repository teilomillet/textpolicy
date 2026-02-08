# textpolicy/algorithms/grpo.py
"""
Group Relative Policy Optimization (GRPO) - Pure Functions for MLX.

GRPO eliminates value function training by using group-relative advantages:
A(τ) = R(τ) - mean(R(group))

These pure functions are designed for:
- MLX compilation with @mx.compile
- Apple Silicon unified memory
- Low abstraction cost
- Composability
"""

from __future__ import annotations

try:
    import mlx.core as mx  # type: ignore
except ImportError:
    mx = None  # MLX is optional; compilation-decorated functions will error if MLX is missing

# Provide a no-op compile decorator when MLX is not available
if mx is None:
    class _DummyMx:
        def compile(self, fn):
            return fn

    mx = _DummyMx()
from typing import List, Union, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import length shaping utilities from dedicated module
from .length_shaping import (
    compute_length_penalty,
    apply_length_shaping,
    compute_length_shaping_stats,
)


# --- Clip Configuration Helper ---
@dataclass
class ClipConfig:
    """Configuration for PPO/DAPO clipping bounds."""
    low: float = 0.2
    high: float = 0.28


def resolve_clip_config(
    clip_ratio: Optional[float],
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
) -> ClipConfig:
    """
    Resolve clipping configuration with backward compatibility.

    Centralizes the logic for handling symmetric vs asymmetric clipping bounds.

    Args:
        clip_ratio: Symmetric clipping ratio (backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)

    Returns:
        ClipConfig with resolved low and high bounds
    """
    if clip_ratio is not None:
        return ClipConfig(low=clip_ratio, high=clip_ratio)
    return ClipConfig(low=clip_ratio_low, high=clip_ratio_high)


def compute_advantages(rewards: Union[List[float], mx.array]) -> mx.array:
    """
    Compute group-relative advantages for GRPO.
    
    Core GRPO innovation: Use group mean as baseline instead of value function.
    This eliminates 50% of neural network training while providing stable gradients.
    
    Formula: A(τ) = R(τ) - mean(R(group))
    
    Args:
        rewards: Episode rewards, either Python list or MLX array
        
    Returns:
        Group-relative advantages as MLX array
        
    Notes:
    - Single vectorized operation (no Python loops)
    - Minimal memory allocation
    - Suitable for @mx.compile decoration
    - Handles variable batch sizes
    """
    if isinstance(rewards, list):
        if not rewards:
            return mx.array([])
        rewards_tensor = mx.array(rewards, dtype=mx.float32)
    elif isinstance(rewards, mx.array):
        rewards_tensor = rewards.astype(mx.float32)
    else:
        raise TypeError(f"Expected list or mx.array, got {type(rewards)}")
    
    # Group-relative advantages: rewards relative to group mean
    # Broadcasting handles the subtraction efficiently
    group_mean = mx.mean(rewards_tensor)
    advantages = rewards_tensor - group_mean
    
    return advantages


def compute_advantages_dr_grpo(rewards: Union[List[float], mx.array]) -> mx.array:
    """
    Compute advantages using Dr. GRPO (GRPO Done Right) - bias-corrected version.
    
    Based on https://arxiv.org/html/2503.20783, this version fixes two key biases:
    1. Response-level length bias: Removes 1/|o_i| normalization 
    2. Question-level difficulty bias: Removes std normalization
    
    Dr. GRPO formula: A(τ) = R(τ) - mean(R(group))
    (Same as basic GRPO but ensures no hidden normalizations)
    
    Args:
        rewards: Episode rewards, either Python list or MLX array
        
    Returns:
        Unbiased group-relative advantages as MLX array
        
    Key improvements over standard GRPO:
    - No response length normalization (prevents length bias)
    - No standard deviation normalization (prevents difficulty bias) 
    - Recovers original unbiased policy gradient objective
    """
    if isinstance(rewards, list):
        if not rewards:
            return mx.array([])
        rewards_tensor = mx.array(rewards, dtype=mx.float32)
    elif isinstance(rewards, mx.array):
        rewards_tensor = rewards.astype(mx.float32)
    else:
        raise TypeError(f"Expected list or mx.array, got {type(rewards)}")
    
    # Dr. GRPO: Pure group-relative advantages without any normalization bias
    # Key insight: Keep advantages raw to avoid length/difficulty biases
    group_mean = mx.mean(rewards_tensor)
    advantages = rewards_tensor - group_mean
    
    # Do not apply extra normalizations that introduce bias:
    # - NO division by response length |o_i| (creates length bias)
    # - NO division by std(rewards) (creates difficulty bias)
    # - Keep raw advantage signal for unbiased learning
    
    return advantages


def policy_loss(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio: float = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    normalize_constant: int = None
) -> mx.array:
    """
    GRPO policy loss with PPO-style clipping, supporting DAPO asymmetric bounds.

    Uses clipped surrogate objective but with group-relative advantages
    instead of GAE advantages. Supports asymmetric clipping bounds (DAPO-style)
    to prevent entropy collapse while maintaining training stability.

    DAPO insight: Asymmetric bounds allow the model to increase probabilities
    of good actions more easily than decreasing probabilities of bad actions,
    promoting diversity and preventing entropy collapse.

    Dr. GRPO insight: Dividing by a fixed constant instead of token count
    eliminates length bias that artificially inflates incorrect (longer) responses.

    Args:
        old_logprobs: Log probabilities from rollout collection
        new_logprobs: Log probabilities from current policy evaluation
        advantages: Group-relative advantages from compute_advantages()
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2, gives lower bound of 0.8)
        clip_ratio_high: Upper bound offset (default 0.28, gives upper bound of 1.28)
        normalize_constant: Fixed constant divisor for loss normalization.
                           If None (default), uses mean (original behavior).
                           If provided, uses sum/constant to eliminate length bias.
                           Typical values: 1024, or batch_size.

    Returns:
        Policy loss scalar (to be minimized)

    Notes:
    - Fully vectorized (no Python loops over batch)
    - Uses in-place operations where possible
    - Suitable for MLX graph optimization
    - Single forward pass through computation
    - DAPO defaults: clip_ratio_low=0.2, clip_ratio_high=0.28
    - Length bias: When using mean, longer sequences have lower per-token
      contribution, creating implicit bias toward short responses.

    References:
        DAPO: An Open-Source LLM Reinforcement Learning System at Scale
        https://arxiv.org/abs/2503.14476

        Dr. GRPO: Understanding R1-Zero-Like Training
        https://arxiv.org/abs/2503.20783
    """
    # Resolve clipping configuration (handles backward compatibility)
    clip_cfg = resolve_clip_config(clip_ratio, clip_ratio_low, clip_ratio_high)

    # Importance ratio: π_new / π_old
    # MLX optimizes exp() for Apple Silicon
    ratio = mx.exp(new_logprobs - old_logprobs)

    # PPO clipped surrogate objective with asymmetric bounds (DAPO-style)
    # L = min(ratio * A, clip(ratio, 1-ε_low, 1+ε_high) * A)
    clipped_ratio = mx.clip(ratio, 1 - clip_cfg.low, 1 + clip_cfg.high)

    # Element-wise minimum
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    min_surr = mx.minimum(surr1, surr2)

    # Normalization: either mean (original) or sum/constant (Dr. GRPO)
    if normalize_constant is not None:
        if normalize_constant <= 0:
            raise ValueError(
                f"normalize_constant must be positive, got {normalize_constant}"
            )
        # Fixed constant normalization eliminates length bias
        # All sequences contribute equally regardless of length
        loss = -mx.sum(min_surr) / normalize_constant
    else:
        # Original mean behavior (for backward compatibility)
        loss = -mx.mean(min_surr)

    return loss


# Optional: Compiled versions for maximum performance
@mx.compile
def compute_advantages_compiled(rewards: mx.array) -> mx.array:
    """Compiled version of compute_advantages for maximum performance."""
    group_mean = mx.mean(rewards)
    return rewards - group_mean


# --- Compiled Policy Loss Variants ---
# Two internal compiled functions for different normalization strategies.
# Compiled functions require static control flow, so we keep them separate.

@mx.compile
def _policy_loss_compiled_mean(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio_low: float,
    clip_ratio_high: float
) -> mx.array:
    """Internal compiled function: mean normalization."""
    ratio = mx.exp(new_logprobs - old_logprobs)
    clipped_ratio = mx.clip(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -mx.mean(mx.minimum(surr1, surr2))


@mx.compile
def _policy_loss_compiled_constant(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio_low: float,
    clip_ratio_high: float,
    normalize_constant: float
) -> mx.array:
    """Internal compiled function: constant normalization."""
    ratio = mx.exp(new_logprobs - old_logprobs)
    clipped_ratio = mx.clip(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -mx.sum(mx.minimum(surr1, surr2)) / normalize_constant


def policy_loss_compiled(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio: float = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28
) -> mx.array:
    """
    Compiled version of policy_loss for maximum performance (mean normalization).

    Supports DAPO-style asymmetric clipping bounds with backward compatibility.
    Uses mean normalization (original behavior).

    Args:
        old_logprobs: Log probabilities from rollout collection
        new_logprobs: Log probabilities from current policy evaluation
        advantages: Group-relative advantages
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)
    """
    clip_cfg = resolve_clip_config(clip_ratio, clip_ratio_low, clip_ratio_high)
    return _policy_loss_compiled_mean(
        old_logprobs, new_logprobs, advantages,
        clip_cfg.low, clip_cfg.high
    )


def policy_loss_compiled_constant_norm(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio: float = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    normalize_constant: float = 1024.0
) -> mx.array:
    """
    Compiled version of policy_loss with fixed constant normalization (Dr. GRPO).

    Uses sum/constant instead of mean to eliminate length bias.

    Args:
        old_logprobs: Log probabilities from rollout collection
        new_logprobs: Log probabilities from current policy evaluation
        advantages: Group-relative advantages
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)
        normalize_constant: Fixed constant divisor (default 1024)

    References:
        Dr. GRPO: Understanding R1-Zero-Like Training
        https://arxiv.org/abs/2503.20783
    """
    if normalize_constant <= 0:
        raise ValueError(
            f"normalize_constant must be positive, got {normalize_constant}"
        )
    clip_cfg = resolve_clip_config(clip_ratio, clip_ratio_low, clip_ratio_high)
    return _policy_loss_compiled_constant(
        old_logprobs, new_logprobs, advantages,
        clip_cfg.low, clip_cfg.high, normalize_constant
    )


def entropy_bonus(logprobs: mx.array, coefficient: float = 0.01) -> mx.array:
    """
    Entropy bonus for exploration (optional GRPO component).

    Args:
        logprobs: Log probabilities from policy
        coefficient: Entropy coefficient (typically small, like 0.01)

    Returns:
        Entropy bonus (added to loss for exploration)
    """
    if coefficient <= 0:
        return mx.array(0.0)

    # Entropy = -sum(p * log(p))
    # For log probabilities: entropy = -sum(exp(logp) * logp)
    probs = mx.exp(logprobs)
    entropy = -mx.sum(probs * logprobs, axis=-1)

    # Return negative entropy (since we add to loss but want to maximize entropy)
    return -coefficient * mx.mean(entropy)


# --- GTPO: Entropy-Weighted Credit Assignment (Issue #10) ---

def compute_token_entropy(logits: mx.array) -> mx.array:
    """
    Compute per-token entropy from logits.

    Uses log-softmax for numerical stability (avoids separate softmax + log
    which can overflow/underflow).

    Args:
        logits: Raw model logits [..., vocab_size]

    Returns:
        Per-token entropy [...] (same shape as logits minus last dimension).
        Values are non-negative; zero means the model is perfectly confident.

    References:
        GTPO: Token and Sequence-Level Reward Shaping with Policy Entropy
        https://arxiv.org/abs/2508.04349
    """
    # log_softmax is more numerically stable than log(softmax(x))
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)
    # H(p) = -sum(p * log(p))
    entropy = -mx.sum(probs * log_probs, axis=-1)
    return entropy


def apply_entropy_weighting(
    advantages: mx.array,
    token_entropies: mx.array,
    entropy_weight: float = 0.1
) -> mx.array:
    """
    Apply GTPO-style entropy weighting to token-level advantages.

    High-entropy tokens (decision points) get amplified advantages;
    low-entropy tokens (routine execution) get dampened advantages.

    Formula:
        w(t) = max(0, 1 + β * (H_norm(t) - 1))
        A_GTPO(t) = A(t) * w(t)

    Where H_norm(t) = H(t) / mean(H).

    Deviation from the GTPO paper (arxiv 2508.04349):
        Paper (Eq. 3):  r̃⁺ᵢ,ₜ = α₁·rᵢ + α₂ · (Hᵢ,ₜ / Σₖ Hₖ,ₜ) · dₜ
                        Additive reward shaping, sum-normalized entropy,
                        bounded in [0, 1] by construction, separate treatment
                        of successful/unsuccessful sequences.
        Ours:           A_GTPO(t) = A(t) · max(0, 1 + β·(H(t)/mean(H) − 1))
                        Multiplicative advantage weighting, mean-normalized
                        entropy, unbounded (hence the max(0, ·) clamp), unified
                        treatment of all sequences.
    The clamp ensures our weights share the paper's structural property:
    non-negative, so advantage signs are never flipped.

    Args:
        advantages: Token-level advantages [total_tokens] (flat 1D).
                   Must already be expanded from episode-level to token-level
                   (compute_advantages_gtpo does this automatically).
        token_entropies: Per-token entropy [total_tokens] (flat 1D).
                        Must be the same shape as ``advantages``.
        entropy_weight: β parameter controlling weighting strength.
                       0.0 disables weighting (returns advantages unchanged).
                       Default 0.1 per GTPO paper (α₂ = 0.1).
                       NOTE: This is NOT ``entropy_coeff`` from entropy_bonus().
                       That parameter controls an additive entropy bonus in the
                       loss; this one controls multiplicative advantage re-weighting.

    Returns:
        Entropy-weighted advantages [total_tokens] (flat 1D, same shape as input).

    Notes:
        - Weights are clamped to [0, ∞) so advantage signs are never flipped.
          Tokens with very low entropy relative to the mean get weight → 0,
          meaning they contribute no gradient signal (fully suppressed).
        - Entropy weights are detached from gradient via mx.stop_gradient
          to prevent the model from gaming entropy (GTPO paper Remark 2.5).
        - When entropy_weight=0, returns advantages unchanged.
        - When all tokens have equal entropy, returns advantages unchanged.

    References:
        GTPO: Token and Sequence-Level Reward Shaping with Policy Entropy
        https://arxiv.org/abs/2508.04349
    """
    if entropy_weight == 0.0:
        return advantages

    # Fail fast on shape mismatches rather than silently broadcasting
    if advantages.shape != token_entropies.shape:
        raise ValueError(
            f"Shape mismatch: advantages {advantages.shape} vs "
            f"token_entropies {token_entropies.shape}. "
            f"Both must be the same shape (token-level)."
        )

    # Compile-safe mean normalization:
    # avoid Python branching on mx arrays (illegal under mx.compile).
    entropy_mean = mx.mean(token_entropies)
    mean_is_tiny = entropy_mean < 1e-7
    safe_mean = mx.where(
        mean_is_tiny,
        mx.array(1.0, dtype=token_entropies.dtype),
        entropy_mean,
    )
    entropy_normalized = token_entropies / (safe_mean + 1e-8)

    # GTPO weight: 1 + β * (H_norm - 1)
    # When H_norm = 1 (average entropy), weight = 1 (no change)
    # When H_norm > 1 (high entropy), weight > 1 (amplified)
    # When H_norm < 1 (low entropy), weight < 1 (dampened)
    #
    # Clamp to non-negative: the actual GTPO paper (Eq. 3) normalizes
    # entropy as H/ΣH which is always in [0,1]. Our H/mean(H) proxy
    # can produce negative weights with large β, which would flip
    # advantage signs — semantically wrong. Clamping to 0 means
    # very-low-entropy tokens are fully suppressed (no gradient signal).
    raw_weights = 1.0 + entropy_weight * (entropy_normalized - 1.0)
    # CRITICAL: stop_gradient detaches entropy weights from the backward pass.
    # Without this, the model could learn to game entropy (increase uncertainty
    # to amplify its own advantage signal). The GTPO paper (Remark 2.5) states:
    # "the entropy term is detached from the gradient computation."
    # Gradient w.r.t. advantages flows normally; gradient w.r.t. token_entropies
    # is exactly zero — verified by test_gradient_wrt_entropies_is_zero.
    entropy_weights = mx.stop_gradient(mx.maximum(raw_weights, 0.0))
    weighted = advantages * entropy_weights

    # Preserve the uniform/near-zero invariant in both eager and compiled modes.
    return mx.where(mean_is_tiny, advantages, weighted)


def compute_advantages_gtpo(
    rewards: Union[List[float], mx.array],
    token_entropies: mx.array,
    entropy_weight: float = 0.1,
    episode_lengths: Optional[List[int]] = None
) -> mx.array:
    """
    Compute GTPO advantages: group-relative base + entropy weighting.

    Convenience function that combines compute_advantages() with
    apply_entropy_weighting(). Handles expansion from episode-level
    to token-level when episode_lengths is provided.

    Args:
        rewards: Episode rewards for group-relative baseline.
        token_entropies: Per-token entropy [total_tokens].
        entropy_weight: β parameter (default 0.1).
        episode_lengths: Length of each episode in tokens.
                        Required when rewards has fewer elements than
                        token_entropies (episode-level vs token-level).
                        If None, assumes rewards are already token-level.

    Returns:
        Entropy-weighted token-level advantages [total_tokens].

    Example:
        >>> rewards = [1.0, 0.0]  # 2 episodes
        >>> entropies = mx.array([3.2, 1.1, 4.5, 2.0, 3.8])  # 5 tokens total
        >>> lengths = [3, 2]  # episode 0 has 3 tokens, episode 1 has 2
        >>> advantages = compute_advantages_gtpo(rewards, entropies, 0.1, lengths)
        >>> advantages.shape  # [5]

    References:
        GTPO: Token and Sequence-Level Reward Shaping with Policy Entropy
        https://arxiv.org/abs/2508.04349
    """
    # Compute episode-level base advantages
    base_advantages = compute_advantages(rewards)

    # Expand to token-level if needed
    if episode_lengths is not None:
        num_episodes = base_advantages.shape[0]
        if num_episodes != len(episode_lengths):
            raise ValueError(
                f"Number of episodes ({num_episodes}) does not match "
                f"episode_lengths ({len(episode_lengths)})"
            )
        expected_tokens = sum(episode_lengths)
        actual_tokens = token_entropies.shape[0]
        if expected_tokens != actual_tokens:
            raise ValueError(
                f"sum(episode_lengths)={expected_tokens} does not match "
                f"token_entropies length {actual_tokens}. "
                f"Episode boundaries and entropy array must align."
            )
        # Expand: repeat each episode's advantage for its token count
        if len(set(episode_lengths)) == 1:
            # All same length: efficient vectorized repeat
            expanded = mx.repeat(base_advantages, episode_lengths[0])
        else:
            # Variable lengths
            parts = []
            for i, length in enumerate(episode_lengths):
                parts.append(mx.repeat(base_advantages[i:i+1], length))
            expanded = mx.concatenate(parts)
    else:
        expanded = base_advantages

    # Apply entropy weighting
    return apply_entropy_weighting(expanded, token_entropies, entropy_weight)


# Note: compute_length_penalty, apply_length_shaping, and compute_length_shaping_stats
# are imported from .length_shaping module (see imports at top of file)


# ---------------------------------------------------------------------------
# GTPO: Paper-Exact Implementation (arXiv 2508.04349)
# ---------------------------------------------------------------------------
#
# The functions below implement GTPO as described in:
#   "GTPO and GRPO-S: Token and Sequence-Level Reward Shaping
#    with Policy Entropy" (Tan et al., 2025)
#
# Key differences from the simplified `apply_entropy_weighting` above:
#   1. Separate O+/O- treatment (Eq. 3 vs Eq. 5)
#   2. Inverse entropy for O- (confident mistakes penalized harder)
#   3. Position-dependent d_t / h_t (surviving sequence counts)
#   4. Additive reward shaping (not multiplicative advantage scaling)
#   5. Separate advantage normalization per group (Eq. 6)
# ---------------------------------------------------------------------------


def compute_gtpo_shaped_rewards(
    rewards: Union[List[float], mx.array],
    token_entropies: mx.array,
    episode_lengths: List[int],
    alpha_1: float = 1.0,
    alpha_2: float = 0.1,
    reward_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[mx.array, mx.array]:
    """
    Compute GTPO token-level shaped rewards per Eq. 3 (O+) and Eq. 5 (O-).

    Paper reference: arXiv 2508.04349, Section 2.1–2.2.

    For successful sequences (O+), Eq. 3:
        r̃⁺ᵢ,ₜ = α₁·rᵢ + α₂ · (Hᵢ,ₜ / Σₖ∈O⁺ₜ Hₖ,ₜ) · dₜ

    For unsuccessful sequences (O-), Eq. 5:
        r̃⁻ⱼ,ₜ = α₁·(-1) + α₂ · ((1/Hⱼ,ₜ) / Σₖ∈O⁻ₜ (1/Hₖ,ₜ)) · hₜ · (-1)

    Where:
        dₜ = |O⁺ₜ| = count of successful sequences with length ≥ t+1
        hₜ = |O⁻ₜ| = count of unsuccessful sequences with length ≥ t+1
        Sequences shorter than t+1 are inactive (entropy treated as 0).

    Args:
        rewards: Episode-level rewards [num_episodes]. Binary {0,1} expected
                 but continuous rewards work too (thresholded by reward_threshold).
        token_entropies: Per-token entropy [total_tokens] (flat 1D).
        episode_lengths: Token count per episode.
        alpha_1: Base reward weight (default 1.0, paper experimental value).
        alpha_2: Entropy-shaped reward weight (default 0.1, paper experimental value).
                 Proposition 2.2 requires α₁ + α₂ = 1 for reward conservation.
        reward_threshold: Threshold for O+/O- partition (default 0.5).
                         Episodes with reward > threshold are O+, else O-.
        eps: Small constant for numerical stability (Remark 2.1).

    Returns:
        Tuple of:
        - shaped_rewards: Flat 1D [total_tokens] shaped token rewards.
        - is_positive: Flat 1D bool [total_tokens], True for tokens from O+ episodes.

    Raises:
        ValueError: If sum(episode_lengths) != len(token_entropies).

    References:
        Eq. 3, Eq. 5, Remark 2.1 — arXiv 2508.04349
    """
    # --- Input validation ---
    if isinstance(rewards, list):
        rewards_arr = mx.array(rewards, dtype=mx.float32)
    else:
        rewards_arr = rewards.astype(mx.float32)

    num_episodes = rewards_arr.shape[0]
    if num_episodes != len(episode_lengths):
        raise ValueError(
            f"rewards length {num_episodes} != episode_lengths length "
            f"{len(episode_lengths)}"
        )

    total_tokens = sum(episode_lengths)
    if total_tokens != token_entropies.shape[0]:
        raise ValueError(
            f"sum(episode_lengths)={total_tokens} != "
            f"token_entropies length {token_entropies.shape[0]}"
        )

    if total_tokens == 0:
        return mx.array([], dtype=mx.float32), mx.array([], dtype=mx.bool_)

    max_len = max(episode_lengths)

    # --- Partition into O+ and O- ---
    # Paper uses binary rewards {0, 1}; we threshold for generality.
    is_positive_ep = rewards_arr > reward_threshold  # [num_episodes]

    # --- Build 2D internal layout: [num_episodes, max_len] ---
    # Flat 1D in/out, 2D internal for vectorized position-wise operations.
    entropies_2d = mx.zeros((num_episodes, max_len), dtype=mx.float32)
    starts = []
    offset = 0
    for i, length in enumerate(episode_lengths):
        starts.append(offset)
        if length > 0:
            entropies_2d = entropies_2d.at[i, :length].add(
                token_entropies[offset:offset + length]
            )
        offset += length

    # Active mask: True where episode i has a token at position t.
    # [num_episodes, max_len]
    active_mask = mx.zeros((num_episodes, max_len), dtype=mx.bool_)
    for i, length in enumerate(episode_lengths):
        if length > 0:
            active_mask = active_mask.at[i, :length].add(
                mx.ones(length, dtype=mx.bool_)
            )

    # --- Positive (O+) shaping: Eq. 3 ---
    # pos_active[i, t] = True if episode i is in O+ AND has token at position t
    pos_ep = is_positive_ep[:, None]  # [num_episodes, 1]
    pos_active = pos_ep & active_mask  # [num_episodes, max_len]

    # H values for active O+ tokens, 0 elsewhere
    H_pos = entropies_2d * pos_active  # [num_episodes, max_len]
    H_sum_pos = mx.sum(H_pos, axis=0)  # [max_len] — Σₖ∈O⁺ₜ Hₖ,ₜ

    # d_t: count of active O+ episodes at each position
    d_t = mx.sum(pos_active.astype(mx.float32), axis=0)  # [max_len]

    # Entropy weight for O+: Hᵢ,ₜ / Σₖ Hₖ,ₜ (Eq. 3 inner fraction)
    # Safe division: if H_sum_pos[t] = 0 (no active O+ or all zero entropy),
    # the weight is 0 (no entropy signal to distribute).
    safe_H_sum_pos = mx.maximum(H_sum_pos, mx.array(eps))
    entropy_weight_pos = H_pos / safe_H_sum_pos[None, :]  # [num_episodes, max_len]

    # Shaped positive rewards:
    # r̃⁺ᵢ,ₜ = α₁·rᵢ + α₂ · (Hᵢ,ₜ/ΣH) · dₜ
    # rewards_arr[:, None] broadcasts episode reward to all positions
    shaped_pos = (
        alpha_1 * rewards_arr[:, None]
        + alpha_2 * entropy_weight_pos * d_t[None, :]
    )
    # Zero out non-O+ and inactive tokens
    shaped_pos = shaped_pos * pos_active

    # --- Negative (O-) shaping: Eq. 5 ---
    neg_ep = ~is_positive_ep[:, None]  # [num_episodes, 1]
    neg_active = neg_ep & active_mask  # [num_episodes, max_len]

    # Inverse entropy for O-: 1/H (confident mistakes get stronger signal)
    # Remark 2.1: add eps to avoid 1/0
    inv_H = 1.0 / (entropies_2d + eps)
    inv_H_neg = inv_H * neg_active  # [num_episodes, max_len]
    inv_H_sum_neg = mx.sum(inv_H_neg, axis=0)  # [max_len]

    # h_t: count of active O- episodes at each position
    h_t = mx.sum(neg_active.astype(mx.float32), axis=0)  # [max_len]

    # Inverse entropy weight: (1/Hⱼ,ₜ) / Σₖ(1/Hₖ,ₜ) (Eq. 5 inner fraction)
    safe_inv_H_sum_neg = mx.maximum(inv_H_sum_neg, mx.array(eps))
    inv_entropy_weight_neg = inv_H_neg / safe_inv_H_sum_neg[None, :]

    # Shaped negative rewards:
    # r̃⁻ⱼ,ₜ = α₁·(-1) + α₂ · (inv_weight) · hₜ · (-1)
    shaped_neg = (
        alpha_1 * (-1.0)
        + alpha_2 * inv_entropy_weight_neg * h_t[None, :] * (-1.0)
    )
    # Zero out non-O- and inactive tokens
    shaped_neg = shaped_neg * neg_active

    # --- Combine and flatten back to 1D ---
    shaped_2d = shaped_pos + shaped_neg  # [num_episodes, max_len]

    # Detach entropy from gradient graph (Remark 2.5):
    # "the entropy term is detached from the gradient computation"
    shaped_2d = mx.stop_gradient(shaped_2d)

    # Flatten: extract non-padded tokens in episode order
    parts_shaped = []
    parts_mask = []
    for i, length in enumerate(episode_lengths):
        if length > 0:
            parts_shaped.append(shaped_2d[i, :length])
            # Compile-safe: repeat the boolean scalar without .item()
            parts_mask.append(
                mx.repeat(is_positive_ep[i:i + 1].astype(mx.bool_), length)
            )

    shaped_flat = mx.concatenate(parts_shaped) if parts_shaped else mx.array(
        [], dtype=mx.float32
    )
    is_positive_flat = mx.concatenate(parts_mask) if parts_mask else mx.array(
        [], dtype=mx.bool_
    )

    return shaped_flat, is_positive_flat


def normalize_gtpo_advantages(
    shaped_rewards: mx.array,
    is_positive: mx.array,
    eps: float = 1e-8,
) -> mx.array:
    """
    Normalize shaped rewards into advantages with separate O+/O- normalization.

    Paper reference: arXiv 2508.04349, Eq. 6.

        Ã⁺ᵢ,ₜ = (r̃⁺ᵢ,ₜ − mean(R̃⁺)) / std(R̃⁺)
        Ã⁻ⱼ,ₜ = (r̃⁻ⱼ,ₜ − mean(R̃⁻)) / std(R̃⁻)

    Where R̃⁺ is the set of ALL shaped positive token rewards and R̃⁻ is the
    set of ALL shaped negative token rewards. Normalization is separate per set.

    Args:
        shaped_rewards: Flat 1D [total_tokens] from compute_gtpo_shaped_rewards.
        is_positive: Flat 1D bool [total_tokens], True for O+ tokens.
        eps: Small constant for numerical stability (Remark 2.1).

    Returns:
        Flat 1D [total_tokens] normalized advantages.

    References:
        Eq. 6 — arXiv 2508.04349
    """
    if shaped_rewards.size == 0:
        return shaped_rewards

    pos_mask_f = is_positive.astype(mx.float32)
    neg_mask_f = (1.0 - pos_mask_f)  # ~is_positive as float

    # --- O+ normalization ---
    pos_count = mx.sum(pos_mask_f)
    # Masked rewards (O- tokens zeroed)
    pos_rewards = shaped_rewards * pos_mask_f
    pos_mean = mx.sum(pos_rewards) / mx.maximum(pos_count, mx.array(1.0))
    pos_diff = (shaped_rewards - pos_mean) * pos_mask_f
    pos_var = mx.sum(pos_diff * pos_diff) / mx.maximum(pos_count, mx.array(1.0))
    pos_std = mx.sqrt(pos_var + eps)

    # --- O- normalization ---
    neg_count = mx.sum(neg_mask_f)
    neg_rewards = shaped_rewards * neg_mask_f
    neg_mean = mx.sum(neg_rewards) / mx.maximum(neg_count, mx.array(1.0))
    neg_diff = (shaped_rewards - neg_mean) * neg_mask_f
    neg_var = mx.sum(neg_diff * neg_diff) / mx.maximum(neg_count, mx.array(1.0))
    neg_std = mx.sqrt(neg_var + eps)

    # Combine: each group's tokens carry their group's normalization
    advantages = (pos_diff / pos_std) + (neg_diff / neg_std)

    return advantages


def gtpo_loss(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    rewards: Union[List[float], mx.array],
    token_entropies: mx.array,
    episode_lengths: List[int],
    alpha_1: float = 1.0,
    alpha_2: float = 0.1,
    clip_epsilon: float = 0.2,
    reward_threshold: float = 0.5,
    eps: float = 1e-8,
) -> mx.array:
    """
    Full GTPO loss per Eq. 7 (arXiv 2508.04349).

    Combines:
        1. Reward shaping (Eq. 3 + Eq. 5) → token-level shaped rewards
        2. Separate O+/O- advantage normalization (Eq. 6)
        3. PPO-style clipped surrogate objective (Eq. 7)

    Eq. 7:
        J_GTPO(θ) = E[ 1/Σ|oₖ| · (
            Σᵢ∈O⁺ Σₜ min(wᵢ,ₜ Ã⁺ᵢ,ₜ , clip(wᵢ,ₜ) Ã⁺ᵢ,ₜ)
          + Σⱼ∈O⁻ Σₜ min(wⱼ,ₜ Ã⁻ⱼ,ₜ , clip(wⱼ,ₜ) Ã⁻ⱼ,ₜ) )]

    Data layout: all arrays are flat 1D (tokens from all episodes concatenated).

    Args:
        old_logprobs: Log probabilities from old policy [total_tokens].
        new_logprobs: Log probabilities from current policy [total_tokens].
        rewards: Episode-level rewards [num_episodes]. Binary {0,1} expected.
        token_entropies: Per-token entropy from old policy [total_tokens].
        episode_lengths: Token count per episode.
        alpha_1: Base reward weight (default 1.0).
        alpha_2: Entropy-shaped weight (default 0.1).
        clip_epsilon: PPO clipping epsilon (default 0.2).
        reward_threshold: Threshold for O+/O- partition (default 0.5).
        eps: Numerical stability constant (Remark 2.1).

    Returns:
        Scalar loss (to be minimized, so negated internally).

    References:
        Eq. 3, 5, 6, 7 — arXiv 2508.04349
    """
    if new_logprobs.shape != old_logprobs.shape:
        raise ValueError(
            f"new_logprobs shape {new_logprobs.shape} != "
            f"old_logprobs shape {old_logprobs.shape}. "
            f"Both must be flat 1D with the same number of tokens."
        )

    total_tokens = old_logprobs.shape[0]
    if total_tokens == 0:
        return mx.array(0.0)

    # Step 1: Compute shaped rewards (Eq. 3 + Eq. 5)
    shaped_rewards, is_positive = compute_gtpo_shaped_rewards(
        rewards, token_entropies, episode_lengths,
        alpha_1=alpha_1, alpha_2=alpha_2,
        reward_threshold=reward_threshold, eps=eps,
    )

    # Step 2: Normalize advantages separately for O+/O- (Eq. 6)
    advantages = normalize_gtpo_advantages(shaped_rewards, is_positive, eps=eps)

    # Validate alignment
    if advantages.shape[0] != total_tokens:
        raise ValueError(
            f"Advantages length {advantages.shape[0]} != "
            f"logprobs length {total_tokens}"
        )

    # Step 3: PPO clipped surrogate objective (Eq. 7)
    ratio = mx.exp(new_logprobs - old_logprobs)
    clipped_ratio = mx.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    min_surr = mx.minimum(surr1, surr2)

    # Eq. 7 normalization: 1/Σ|oₖ| (total tokens) = mean
    loss = -mx.mean(min_surr)

    return loss


# DAPO-style dynamic batch filtering (Issue #9)
def _get_episode_reward(episode) -> float:
    """Extract total reward from episode (handles both Episode objects and dicts)."""
    if hasattr(episode, 'rew'):
        # Episode object
        return float(mx.sum(mx.array(episode.rew)).item())
    else:
        # Serialized dictionary
        rew = episode.get('rew', episode.get('reward', [0.0]))
        if isinstance(rew, (int, float)):
            return float(rew)
        return float(mx.sum(mx.array(rew)).item())


def _get_prompt_key(episode) -> tuple:
    """
    Generate a hashable key for an episode's prompt.

    Handles both Episode objects and serialized dictionaries.
    Uses the observation (prompt) tokens to identify the prompt.
    """
    if hasattr(episode, 'obs'):
        obs = episode.obs
    else:
        obs = episode.get('obs', [])

    # Flatten nested structures to create consistent key
    flattened = []
    for item in obs:
        if hasattr(item, 'tolist'):  # MLX array
            flattened.extend(item.tolist())
        elif isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    return tuple(flattened)


def _precompute_episode_rewards(episodes: List[Any]) -> List[float]:
    """
    Pre-compute rewards for all episodes in a single pass.

    Uses batched MLX evaluation to avoid per-episode .item() sync barriers.
    All mx.sum() calls are built lazily, then evaluated in one mx.eval() call.

    Args:
        episodes: List of episodes

    Returns:
        List of rewards in the same order as episodes
    """
    if not episodes:
        return []

    rewards: List[Optional[float]] = [None] * len(episodes)
    pending: List[Tuple[int, mx.array]] = []  # (index, lazy_sum) pairs

    for i, ep in enumerate(episodes):
        if hasattr(ep, 'rew'):
            rew = ep.rew
        else:
            rew = ep.get('rew', ep.get('reward', [0.0]))

        if isinstance(rew, (int, float)):
            rewards[i] = float(rew)
        else:
            pending.append((i, mx.sum(mx.array(rew))))

    # Single sync barrier for all array rewards
    if pending:
        indices, lazy_sums = zip(*pending)
        stacked = mx.stack(list(lazy_sums))
        mx.eval(stacked)
        values = stacked.tolist()
        for idx, val in zip(indices, values):
            rewards[idx] = float(val)

    return rewards  # type: ignore[return-value]


def _compute_group_variance_and_mean(
    group_indices: List[int],
    all_rewards: List[float]
) -> Tuple[float, float]:
    """
    Compute variance and mean for a group of episodes using pre-computed rewards.

    Args:
        group_indices: Indices into all_rewards for this group
        all_rewards: Pre-computed rewards for all episodes

    Returns:
        Tuple of (variance, mean)
    """
    group_rewards = mx.array([all_rewards[i] for i in group_indices])
    return mx.var(group_rewards).item(), mx.mean(group_rewards).item()


def filter_informative_prompts(
    episodes: List[Any],
    min_variance: float = 0.01,
    keep_single_completion: bool = True
) -> Tuple[List[Any], Dict[str, Union[int, float]]]:
    """
    Filter episodes to keep only informative prompts (DAPO dynamic sampling).

    Removes prompts where all completions have same outcome:
    - All correct (reward ~1.0): no learning signal (nothing to improve)
    - All wrong (reward ~0.0): no positive signal (can't learn what works)

    GRPO uses group-relative advantages. If all completions have the same
    outcome, advantages are zero, producing no gradient and wasting compute.

    Note on single-completion prompts:
        The DAPO paper (Equation 11) defines informative prompts as having
        mixed outcomes: `0 < |correct| < G`. This assumes G > 1 completions
        per prompt. For single-completion prompts (G=1), variance is always 0
        by definition, but this doesn't mean "all outcomes are the same" -
        it means we have insufficient data to determine variance.

        By default (keep_single_completion=True), single-completion prompts
        are kept since they still provide valid gradient signal. Set to False
        to filter them out (stricter DAPO interpretation).

    Args:
        episodes: List of episodes (Episode objects or serialized dicts)
        min_variance: Minimum reward variance to keep a prompt group.
                     Groups with variance below this threshold are filtered out.
                     Default 0.01 filters prompts with essentially identical rewards.
                     Only applied to groups with 2+ completions.
        keep_single_completion: Whether to keep prompts with only one completion.
                               Default True (keep them). Set False to require
                               multiple completions for variance calculation.

    Returns:
        Tuple of:
        - filtered: List of episodes from informative prompts
        - stats: Dictionary with filtering statistics:
            - 'prompts_kept': Number of prompt groups kept
            - 'prompts_dropped_all_correct': Prompts where all completions succeeded
            - 'prompts_dropped_all_wrong': Prompts where all completions failed
            - 'prompts_dropped_single': Prompts dropped due to single completion
            - 'prompts_kept_single': Single-completion prompts that were kept
            - 'episodes_kept': Total episodes kept
            - 'episodes_dropped': Total episodes filtered out
            - 'filter_rate': Fraction of prompts filtered

    Example:
        >>> filtered, stats = filter_informative_prompts(episodes, min_variance=0.01)
        >>> print(f"Kept {stats['prompts_kept']} prompts, "
        ...       f"dropped {stats['prompts_dropped_all_correct']} all-correct, "
        ...       f"{stats['prompts_dropped_all_wrong']} all-wrong")

    References:
        DAPO: An Open-Source LLM Reinforcement Learning System at Scale
        https://arxiv.org/abs/2503.14476 (Equation 11: 0 < |correct| < G)

        GRPO++ Tricks
        https://cameronrwolfe.substack.com/p/grpo-tricks
    """
    if not episodes:
        return [], {
            'prompts_kept': 0,
            'prompts_dropped_all_correct': 0,
            'prompts_dropped_all_wrong': 0,
            'prompts_dropped_single': 0,
            'prompts_kept_single': 0,
            'episodes_kept': 0,
            'episodes_dropped': 0,
            'filter_rate': 0.0,
        }

    # Pre-compute all rewards once (avoids repeated _get_episode_reward calls)
    all_rewards = _precompute_episode_rewards(episodes)

    # Group episodes by prompt, storing indices instead of episodes
    prompt_groups: Dict[tuple, List[int]] = defaultdict(list)
    for idx, ep in enumerate(episodes):
        prompt_key = _get_prompt_key(ep)
        prompt_groups[prompt_key].append(idx)

    filtered = []
    stats = {
        'prompts_kept': 0,
        'prompts_dropped_all_correct': 0,
        'prompts_dropped_all_wrong': 0,
        'prompts_dropped_single': 0,
        'prompts_kept_single': 0,
        'episodes_kept': 0,
        'episodes_dropped': 0,
    }

    for prompt_key, group_indices in prompt_groups.items():
        group_size = len(group_indices)

        # Handle single-completion prompts separately
        if group_size == 1:
            if keep_single_completion:
                # Keep single-completion prompts (variance undefined, not "zero")
                filtered.append(episodes[group_indices[0]])
                stats['prompts_kept'] += 1
                stats['prompts_kept_single'] += 1
                stats['episodes_kept'] += 1
            else:
                # Filter out single-completion prompts (strict DAPO interpretation)
                stats['prompts_dropped_single'] += 1
                stats['episodes_dropped'] += 1
            continue

        # For groups with 2+ completions, use variance criterion
        variance, mean_reward = _compute_group_variance_and_mean(group_indices, all_rewards)

        if variance > min_variance:
            # Informative: mixed outcomes, keep all episodes from this prompt
            for idx in group_indices:
                filtered.append(episodes[idx])
            stats['prompts_kept'] += 1
            stats['episodes_kept'] += group_size
        else:
            # Uninformative: all completions have same outcome
            stats['episodes_dropped'] += group_size
            if mean_reward > 0.5:
                stats['prompts_dropped_all_correct'] += 1
            else:
                stats['prompts_dropped_all_wrong'] += 1

    # Compute filter rate
    total_prompts = len(prompt_groups)
    stats['filter_rate'] = 1.0 - (stats['prompts_kept'] / total_prompts) if total_prompts > 0 else 0.0

    return filtered, stats


def compute_prompt_group_stats(episodes: List[Any]) -> Dict[str, Any]:
    """
    Compute statistics about prompt groups for monitoring.

    Useful for understanding the distribution of prompts and completions
    before and after filtering.

    Args:
        episodes: List of episodes

    Returns:
        Dictionary with:
        - 'num_prompts': Total unique prompts
        - 'num_episodes': Total episodes
        - 'completions_per_prompt': Average completions per prompt
        - 'reward_variance_mean': Mean variance across prompt groups
        - 'reward_variance_std': Std of variance across prompt groups
    """
    if not episodes:
        return {
            'num_prompts': 0,
            'num_episodes': 0,
            'completions_per_prompt': 0.0,
            'reward_variance_mean': 0.0,
            'reward_variance_std': 0.0,
        }

    # Pre-compute all rewards once
    all_rewards = _precompute_episode_rewards(episodes)

    # Group by prompt, storing indices
    prompt_groups: Dict[tuple, List[int]] = defaultdict(list)
    for idx, ep in enumerate(episodes):
        prompt_key = _get_prompt_key(ep)
        prompt_groups[prompt_key].append(idx)

    # Compute variance for each group using pre-computed rewards
    variances = []
    for group_indices in prompt_groups.values():
        group_rewards = mx.array([all_rewards[i] for i in group_indices])
        variances.append(mx.var(group_rewards).item())

    variances_arr = mx.array(variances) if variances else mx.array([0.0])

    return {
        'num_prompts': len(prompt_groups),
        'num_episodes': len(episodes),
        'completions_per_prompt': len(episodes) / len(prompt_groups) if prompt_groups else 0.0,
        'reward_variance_mean': float(mx.mean(variances_arr).item()),
        'reward_variance_std': float(mx.std(variances_arr).item()),
    }


# Convenience function for complete GRPO computation
def grpo_loss(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    rewards: Union[List[float], mx.array],
    clip_ratio: float = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    entropy_coeff: float = 0.0,
    normalize_constant: int = None,
    token_entropies: Optional[mx.array] = None,
    entropy_weight: float = 0.1,
    episode_lengths: Optional[List[int]] = None
) -> mx.array:
    """
    Complete GRPO loss computation in one function.

    Combines advantage calculation and policy loss for convenience.
    Can be compiled as a single unit for maximum efficiency.
    Supports DAPO-style asymmetric clipping bounds, Dr. GRPO length-bias fix,
    and GTPO entropy-weighted credit assignment.

    Data layout:
        All logprob/advantage arrays are **flat 1D** — tokens from all episodes
        concatenated into a single vector.  Episode boundaries are tracked via
        ``episode_lengths``, NOT via array dimensions.  There is no 2D
        ``[num_episodes, seq_len]`` layout anywhere in this pipeline.

    Args:
        old_logprobs: Log probabilities from rollout [total_tokens] (flat 1D).
        new_logprobs: Log probabilities from current policy [total_tokens] (flat 1D).
        rewards: Episode rewards for group-relative advantages
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)
        entropy_coeff: Entropy bonus coefficient (0 disables)
        normalize_constant: Fixed constant divisor for loss normalization.
                           If None (default), uses mean. If provided, uses
                           sum/constant to eliminate length bias.
        token_entropies: Per-token entropy for GTPO weighting [total_tokens].
                        If None, uses standard GRPO (no entropy weighting).
                        If provided, advantages are weighted by normalized
                        entropy so decision-point tokens get amplified signal.
        entropy_weight: β parameter for GTPO (default 0.1). Only used when
                       token_entropies is provided.
        episode_lengths: Token count per episode, required for expanding
                        episode-level advantages to token-level when using
                        GTPO. If None with token_entropies, assumes rewards
                        are already token-level aligned.

    Returns:
        Total GRPO loss (policy + optional entropy bonus)

    References:
        DAPO: An Open-Source LLM Reinforcement Learning System at Scale
        https://arxiv.org/abs/2503.14476

        Dr. GRPO: Understanding R1-Zero-Like Training
        https://arxiv.org/abs/2503.20783

        GTPO: Token and Sequence-Level Reward Shaping with Policy Entropy
        https://arxiv.org/abs/2508.04349
    """
    if token_entropies is not None:
        # GTPO mode: entropy-weighted token-level advantages
        advantages = compute_advantages_gtpo(
            rewards, token_entropies, entropy_weight, episode_lengths
        )
        # Validate GTPO advantages align with logprobs (both are flat 1D)
        if advantages.shape[0] != old_logprobs.shape[0]:
            raise ValueError(
                f"GTPO advantages length {advantages.shape[0]} does not match "
                f"old_logprobs length {old_logprobs.shape[0]}. "
                f"Check episode_lengths sum matches total tokens."
            )
    else:
        # Standard GRPO: episode-level advantages.
        # Note: expansion from episode-level [num_episodes] to token-level
        # [total_tokens] is handled by the Trainer._loss_fn, not here.
        # When called directly, the caller must ensure advantages and
        # logprobs are already aligned (both flat 1D, same length).
        advantages = compute_advantages(rewards)

    # Compute policy loss with asymmetric clipping and optional length-bias fix
    policy_loss_val = policy_loss(
        old_logprobs, new_logprobs, advantages,
        clip_ratio=clip_ratio,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        normalize_constant=normalize_constant
    )

    # Add entropy bonus if specified
    if entropy_coeff > 0:
        entropy_bonus_val = entropy_bonus(new_logprobs, entropy_coeff)
        return policy_loss_val + entropy_bonus_val

    return policy_loss_val


# Performance monitoring utilities
def compute_metrics(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio: float = None,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28
) -> dict:
    """
    Compute GRPO training metrics for monitoring.

    Supports DAPO-style asymmetric clipping bounds and tracks clip fractions
    for upper vs lower bounds separately.

    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy
        advantages: Group-relative advantages
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)

    Returns:
        Dictionary of metrics for logging/monitoring, including:
        - clip_fraction_lower: Fraction of ratios clipped at lower bound
        - clip_fraction_upper: Fraction of ratios clipped at upper bound
        - clip_fraction: Total fraction of ratios clipped (either bound)
    """
    # Resolve clipping configuration (handles backward compatibility)
    clip_cfg = resolve_clip_config(clip_ratio, clip_ratio_low, clip_ratio_high)

    # Importance ratio statistics
    ratio = mx.exp(new_logprobs - old_logprobs)

    # Asymmetric clipping bounds
    clip_lower = 1 - clip_cfg.low
    clip_upper = 1 + clip_cfg.high

    # Track clip fractions separately for upper and lower bounds
    clipped_lower = ratio < clip_lower
    clipped_upper = ratio > clip_upper
    clipped = clipped_lower | clipped_upper

    clip_fraction_lower = mx.mean(clipped_lower.astype(mx.float32))
    clip_fraction_upper = mx.mean(clipped_upper.astype(mx.float32))
    clip_fraction = mx.mean(clipped.astype(mx.float32))

    # KL divergence approximation
    kl_div = mx.mean(old_logprobs - new_logprobs)

    return {
        'mean_advantage': mx.mean(advantages).item(),
        'std_advantage': mx.std(advantages).item(),
        'mean_ratio': mx.mean(ratio).item(),
        'clip_fraction': clip_fraction.item(),
        'clip_fraction_lower': clip_fraction_lower.item(),
        'clip_fraction_upper': clip_fraction_upper.item(),
        'kl_divergence': kl_div.item(),
        'min_advantage': mx.min(advantages).item(),
        'max_advantage': mx.max(advantages).item(),
        'clip_ratio_low': clip_cfg.low,
        'clip_ratio_high': clip_cfg.high
    }


# --- Episode Packing Helper ---
def _flatten_tokens(items: List[Any]) -> List:
    """Flatten nested token sequences into a flat list."""
    if items is None:
        return []

    # Allow passing a bare array/scalar as a single token container.
    if hasattr(items, 'tolist'):
        values = items.tolist()
        return values if isinstance(values, list) else [values]

    if not isinstance(items, list):
        return [items]

    flattened = []
    for item in items:
        if hasattr(item, 'tolist'):  # MLX array
            values = item.tolist()
            if isinstance(values, list):
                flattened.extend(values)
            else:
                flattened.append(values)
        elif isinstance(item, list):  # Python list
            flattened.extend(item)
        else:  # Single token
            flattened.append(item)
    return flattened


def _pack_episodes(episodes: List[Any], sort_by_length: bool = True) -> Dict[str, Any]:
    """
    Pack episodes into batch data for GRPO training.

    This is the shared helper for episode-to-batch conversion, used by all
    data selectors (select_all_data, select_informative_data, select_recent_data).

    Output layout:
        - 'obs': 2D ``[N, max_obs_len]`` right-padded full sequences (prompt + response)
        - 'act': 2D ``[N, max_act_len]`` right-padded response tokens
        - 'logprob': Flat 1D unpadded concatenated log probabilities
        - 'rewards': Episode rewards as MLX array
        - 'episode_lengths': List of per-episode response token counts
        - 'prompt_lengths': List of per-episode prompt token counts

    The 2D obs/act layout enables batched model forward passes in
    ``_extract_grpo_logprobs``, converting N sequential calls into one.
    Logprobs remain flat 1D (unpadded) to preserve the flat-token invariant
    used by ``policy_loss`` and ``_expand_advantages``.

    When ``sort_by_length=True`` (default), episodes are sorted by total
    sequence length (prompt + response) so that similarly-sized episodes are
    adjacent. Combined with micro-batch trimming in the Trainer, this reduces
    wasted padding compute — each chunk is trimmed to its local max instead
    of the global max.

    Args:
        episodes: List of episodes (Episode objects or serialized dicts)
        sort_by_length: Sort episodes by total sequence length (ascending)
            before padding. Default True. Pass False to preserve original
            episode ordering (useful for tests with frozen expected values).

    Returns:
        Batch dictionary (see layout above)
    """
    if not episodes:
        return {
            'obs': mx.array([], dtype=mx.int64),
            'act': mx.array([], dtype=mx.int64),
            'logprob': mx.array([], dtype=mx.float32),
            'rewards': mx.array([]),
            'episode_lengths': [],
            'prompt_lengths': [],
        }

    episode_lengths = []
    prompt_lengths = []
    all_obs = []
    all_acts = []
    all_logprobs = []
    pending_reward_sums: List[Tuple[int, mx.array]] = []
    scalar_rewards: Dict[int, float] = {}

    for i, episode in enumerate(episodes):
        if hasattr(episode, 'rew'):
            # Episode object with attributes
            pending_reward_sums.append((i, mx.sum(mx.array(episode.rew))))

            # Flatten observation and action tokens
            flattened_obs = _flatten_tokens(episode.obs)
            flattened_acts = _flatten_tokens(episode.act)

            # Use flattened token count for episode_lengths (used by _expand_advantages)
            # This ensures alignment between expanded advantages and actual token sequences
            episode_lengths.append(len(flattened_acts))
            prompt_lengths.append(len(flattened_obs))

            # Create full sequence: [prompt_tokens..., response_tokens...]
            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            # Keep logprobs token-aligned with flattened actions.
            all_logprobs.append(_flatten_tokens(episode.logprob) if episode.logprob is not None else [])
        else:
            # Serialized dictionary from multiprocessing
            rew = episode['rew']
            if isinstance(rew, (int, float)):
                scalar_rewards[i] = float(rew)
            else:
                pending_reward_sums.append((i, mx.sum(mx.array(rew))))

            # Flatten observation and action tokens
            flattened_obs = _flatten_tokens(episode['obs'])
            flattened_acts = _flatten_tokens(episode['act'])

            # Use flattened token count for episode_lengths
            episode_lengths.append(len(flattened_acts))
            prompt_lengths.append(len(flattened_obs))

            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            logprob = episode.get('logprob', [])
            all_logprobs.append(_flatten_tokens(logprob) if logprob is not None else [])

    # Batch evaluate all pending reward sums (single sync barrier instead of N)
    episode_rewards = [0.0] * len(episodes)
    for idx, val in scalar_rewards.items():
        episode_rewards[idx] = val
    if pending_reward_sums:
        indices, lazy_sums = zip(*pending_reward_sums)
        stacked = mx.stack(list(lazy_sums))
        mx.eval(stacked)
        values = stacked.tolist()
        for idx, val in zip(indices, values):
            episode_rewards[idx] = float(val)

    # Sort episodes by total sequence length so similarly-sized episodes are
    # adjacent.  When micro-batching, each chunk is trimmed to its local max
    # instead of the global max — reducing wasted padding compute.
    if sort_by_length and len(all_obs) > 1:
        sort_idx = sorted(range(len(all_obs)), key=lambda i: len(all_obs[i]))
        all_obs = [all_obs[i] for i in sort_idx]
        all_acts = [all_acts[i] for i in sort_idx]
        all_logprobs = [all_logprobs[i] for i in sort_idx]
        episode_lengths = [episode_lengths[i] for i in sort_idx]
        prompt_lengths = [prompt_lengths[i] for i in sort_idx]
        episode_rewards = [episode_rewards[i] for i in sort_idx]

    # Find maximum sequence lengths for padding
    max_obs_len = max(len(obs) for obs in all_obs) if all_obs else 0
    max_act_len = max(len(act) for act in all_acts) if all_acts else 0

    # Convert to MLX arrays — keep ALL episodes (including empty ones) to
    # maintain 1:1 alignment with episode_lengths and prompt_lengths.
    # Empty episodes get zero-padded rows; compute_logprobs_batched skips
    # them via the r_len == 0 check.
    all_obs_mx = [mx.array(obs, dtype=mx.int64) if obs else mx.zeros(0, dtype=mx.int64) for obs in all_obs]
    all_acts_mx = [mx.array(act, dtype=mx.int64) if act else mx.zeros(0, dtype=mx.int64) for act in all_acts]

    # Pad and stack to 2D: [N, max_len] — enables batched model forward passes.
    # Every episode gets a row, even empty ones (padded to max_len with zeros).
    if max_obs_len > 0:
        padded_obs = [mx.pad(obs, (0, max_obs_len - obs.shape[0]), constant_values=0)
                      if obs.shape[0] < max_obs_len else obs[:max_obs_len]
                      for obs in all_obs_mx]
        stacked_obs = mx.stack(padded_obs)  # [N, max_obs_len]
    else:
        # All episodes empty — produce (N, 0) 2D for consistency with act.
        stacked_obs = mx.zeros((len(all_obs_mx), 0), dtype=mx.int64)

    if max_act_len > 0:
        padded_acts = [mx.pad(act, (0, max_act_len - act.shape[0]), constant_values=0)
                       if act.shape[0] < max_act_len else act[:max_act_len]
                       for act in all_acts_mx]
        stacked_acts = mx.stack(padded_acts)  # [N, max_act_len]
    else:
        # All episodes have empty responses — produce (N, 0) 2D so the
        # trainer still enters the batched path (where r_len == 0 skips
        # each episode) instead of falling to the flat 1D path.
        stacked_acts = mx.zeros((len(all_acts_mx), 0), dtype=mx.int64)

    # Logprobs: concatenate WITHOUT padding to preserve flat 1D invariant.
    # Padding would introduce spurious zero-logprobs that break
    # shape == sum(episode_lengths) for variable-length episodes.
    all_logprobs_mx = [mx.array(logprob, dtype=mx.float32) if logprob else mx.array([], dtype=mx.float32) for logprob in all_logprobs]
    non_empty_logprobs = [logprob for logprob in all_logprobs_mx if logprob.size > 0]
    flat_logprobs = mx.concatenate(non_empty_logprobs) if non_empty_logprobs else mx.array([], dtype=mx.float32)

    return {
        'obs': stacked_obs,
        'act': stacked_acts,
        'logprob': flat_logprobs,
        'rewards': mx.array(episode_rewards),
        'episode_lengths': episode_lengths,
        'prompt_lengths': prompt_lengths,
    }


# Algorithm-specific data selection strategies
def select_all_data(buffer):
    """
    GRPO data selector: Use all available data.

    GRPO is on-policy but can benefit from using all collected episodes
    since group-relative advantages normalize across the entire group.

    Args:
        buffer: Buffer containing episodes

    Returns:
        All episode data prepared for training
    """
    from textpolicy.buffer import Buffer
    if not isinstance(buffer, Buffer):
        raise TypeError(f"Expected Buffer, got {type(buffer)}")

    episodes = buffer.episodes
    if not episodes:
        raise ValueError("Buffer is empty - no episodes to train on")

    return _pack_episodes(episodes)


def select_informative_data(buffer, min_variance: float = 0.01):
    """
    GRPO data selector with dynamic batch filtering (DAPO-style).

    Filters out uninformative prompts where all completions have the same
    outcome (all correct or all wrong), improving sample efficiency by
    maintaining meaningful gradient signals.

    This is the recommended selector for GRPO training when using multiple
    completions per prompt, as it eliminates wasted compute on prompts
    that provide no learning signal.

    Args:
        buffer: Buffer containing episodes (Episode objects or serialized dictionaries)
        min_variance: Minimum reward variance to keep a prompt group.
                     Prompts with variance below this are filtered out.

    Returns:
        Filtered episode data prepared for training, plus filtering stats.

    Example:
        >>> batch_data = select_informative_data(buffer, min_variance=0.01)
        >>> # batch_data includes 'filter_stats' with filtering information

    References:
        DAPO: An Open-Source LLM Reinforcement Learning System at Scale
        https://arxiv.org/abs/2503.14476
    """
    from textpolicy.buffer import Buffer
    if not isinstance(buffer, Buffer):
        raise TypeError(f"Expected Buffer, got {type(buffer)}")

    episodes = buffer.episodes
    if not episodes:
        raise ValueError("Buffer is empty - no episodes to train on")

    # Filter to keep only informative prompts
    filtered_episodes, filter_stats = filter_informative_prompts(episodes, min_variance)

    if not filtered_episodes:
        raise ValueError(
            f"All prompts filtered out (min_variance={min_variance}). "
            f"Stats: {filter_stats}. Consider lowering min_variance or "
            "ensuring diversity in completions."
        )

    # Pack filtered episodes using shared helper
    batch_data = _pack_episodes(filtered_episodes)
    batch_data['filter_stats'] = filter_stats
    return batch_data


def select_recent_data(buffer, max_episodes: int = 100):
    """
    GRPO data selector: Use only recent episodes.

    Alternative selector for GRPO that limits to recent episodes
    for faster training on large buffers.

    Args:
        buffer: Buffer containing episodes (Episode objects or serialized dictionaries)
        max_episodes: Maximum number of recent episodes to use

    Returns:
        Recent episode data prepared for training
    """
    from textpolicy.buffer import Buffer
    if not isinstance(buffer, Buffer):
        raise TypeError(f"Expected Buffer, got {type(buffer)}")

    episodes = buffer.episodes
    if not episodes:
        raise ValueError("Buffer is empty - no episodes to train on")

    # Select recent episodes
    recent_episodes = episodes[-max_episodes:] if len(episodes) > max_episodes else episodes

    return _pack_episodes(recent_episodes)
