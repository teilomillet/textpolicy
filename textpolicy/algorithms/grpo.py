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


# Note: compute_length_penalty, apply_length_shaping, and compute_length_shaping_stats
# are imported from .length_shaping module (see imports at top of file)


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


def filter_informative_prompts(
    episodes: List[Any],
    min_variance: float = 0.01,
    keep_single_completion: bool = True
) -> Tuple[List[Any], Dict[str, int]]:
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

    # Group episodes by prompt
    prompt_groups: Dict[tuple, List[Any]] = defaultdict(list)
    for ep in episodes:
        prompt_key = _get_prompt_key(ep)
        prompt_groups[prompt_key].append(ep)

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

    for prompt_key, group in prompt_groups.items():
        group_size = len(group)

        # Handle single-completion prompts separately
        if group_size == 1:
            if keep_single_completion:
                # Keep single-completion prompts (variance undefined, not "zero")
                filtered.extend(group)
                stats['prompts_kept'] += 1
                stats['prompts_kept_single'] += 1
                stats['episodes_kept'] += 1
            else:
                # Filter out single-completion prompts (strict DAPO interpretation)
                stats['prompts_dropped_single'] += 1
                stats['episodes_dropped'] += 1
            continue

        # For groups with 2+ completions, use variance criterion
        rewards = mx.array([_get_episode_reward(ep) for ep in group])
        variance = mx.var(rewards).item()
        mean_reward = mx.mean(rewards).item()

        if variance > min_variance:
            # Informative: mixed outcomes, keep all episodes from this prompt
            filtered.extend(group)
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

    # Group by prompt
    prompt_groups: Dict[tuple, List[Any]] = defaultdict(list)
    for ep in episodes:
        prompt_key = _get_prompt_key(ep)
        prompt_groups[prompt_key].append(ep)

    # Compute variance for each group
    variances = []
    for group in prompt_groups.values():
        rewards = mx.array([_get_episode_reward(ep) for ep in group])
        variances.append(mx.var(rewards).item())

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
    normalize_constant: int = None
) -> mx.array:
    """
    Complete GRPO loss computation in one function.

    Combines advantage calculation and policy loss for convenience.
    Can be compiled as a single unit for maximum efficiency.
    Supports DAPO-style asymmetric clipping bounds and Dr. GRPO length-bias fix.

    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy
        rewards: Episode rewards for group-relative advantages
        clip_ratio: Symmetric clipping ratio (for backward compatibility).
                   If provided, overrides clip_ratio_low and clip_ratio_high.
        clip_ratio_low: Lower bound offset (default 0.2)
        clip_ratio_high: Upper bound offset (default 0.28)
        entropy_coeff: Entropy bonus coefficient (0 disables)
        normalize_constant: Fixed constant divisor for loss normalization.
                           If None (default), uses mean. If provided, uses
                           sum/constant to eliminate length bias.

    Returns:
        Total GRPO loss (policy + optional entropy)

    References:
        DAPO: An Open-Source LLM Reinforcement Learning System at Scale
        https://arxiv.org/abs/2503.14476

        Dr. GRPO: Understanding R1-Zero-Like Training
        https://arxiv.org/abs/2503.20783
    """
    # Compute group-relative advantages
    advantages = compute_advantages(rewards)

    # Expand advantages to match logprob sequence length if needed
    if advantages.ndim == 1 and old_logprobs.ndim > 1:
        # Each episode contributes its advantage to all tokens in that episode
        # This requires knowing episode boundaries - simplified version assumes
        # advantages and logprobs are already aligned
        pass

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
    flattened = []
    for item in items:
        if hasattr(item, 'tolist'):  # MLX array
            flattened.extend(item.tolist())
        elif isinstance(item, list):  # Python list
            flattened.extend(item)
        else:  # Single token
            flattened.append(item)
    return flattened


def _pack_episodes(episodes: List[Any]) -> Dict[str, Any]:
    """
    Pack episodes into batch data for GRPO training.

    This is the shared helper for episode-to-batch conversion, used by all
    data selectors (select_all_data, select_informative_data, select_recent_data).

    Args:
        episodes: List of episodes (Episode objects or serialized dicts)

    Returns:
        Dictionary with:
        - 'obs': Flat concatenated full sequences (prompt + response)
        - 'act': Flat concatenated response tokens
        - 'logprob': Flat concatenated log probabilities
        - 'rewards': Episode rewards as MLX array
        - 'episode_lengths': List of episode lengths
    """
    if not episodes:
        return {
            'obs': mx.array([], dtype=mx.int64),
            'act': mx.array([], dtype=mx.int64),
            'logprob': mx.array([], dtype=mx.float32),
            'rewards': mx.array([]),
            'episode_lengths': [],
        }

    episode_rewards = []
    episode_lengths = []
    all_obs = []
    all_acts = []
    all_logprobs = []

    for episode in episodes:
        if hasattr(episode, 'rew'):
            # Episode object with attributes
            episode_reward = mx.sum(mx.array(episode.rew)).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode.obs))

            # Flatten observation and action tokens
            flattened_obs = _flatten_tokens(episode.obs)
            flattened_acts = _flatten_tokens(episode.act)

            # Create full sequence: [prompt_tokens..., response_tokens...]
            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            all_logprobs.append(episode.logprob if episode.logprob is not None else [])
        else:
            # Serialized dictionary from multiprocessing
            episode_reward = mx.sum(mx.array(episode['rew'])).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode['obs']))

            # Flatten observation and action tokens
            flattened_obs = _flatten_tokens(episode['obs'])
            flattened_acts = _flatten_tokens(episode['act'])

            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            all_logprobs.append(episode.get('logprob', []))

    # Find maximum sequence lengths for padding
    max_obs_len = max(len(obs) for obs in all_obs) if all_obs else 0
    max_act_len = max(len(act) for act in all_acts) if all_acts else 0
    max_logprob_len = max(len(logprob) for logprob in all_logprobs) if all_logprobs else 0

    # Convert to MLX arrays with padding
    # Always create an array for each episode to maintain alignment
    all_obs_mx = [mx.array(obs, dtype=mx.int64) if obs else mx.array([], dtype=mx.int64) for obs in all_obs]
    all_acts_mx = [mx.array(act, dtype=mx.int64) if act else mx.array([], dtype=mx.int64) for act in all_acts]
    all_logprobs_mx = [mx.array(logprob, dtype=mx.float32) if logprob else mx.array([], dtype=mx.float32) for logprob in all_logprobs]

    # Filter out empty arrays for padding/concatenation
    non_empty_obs = [obs for obs in all_obs_mx if obs.size > 0]
    non_empty_acts = [act for act in all_acts_mx if act.size > 0]
    non_empty_logprobs = [logprob for logprob in all_logprobs_mx if logprob.size > 0]

    # Pad using native MLX operations
    if non_empty_obs:
        padded_obs = [mx.pad(obs, (0, max_obs_len - obs.shape[0]), constant_values=0)
                      if obs.shape[0] < max_obs_len else obs[:max_obs_len]
                      for obs in non_empty_obs]
    else:
        padded_obs = []

    if non_empty_acts:
        padded_acts = [mx.pad(act, (0, max_act_len - act.shape[0]), constant_values=0)
                       if act.shape[0] < max_act_len else act[:max_act_len]
                       for act in non_empty_acts]
    else:
        padded_acts = []

    if non_empty_logprobs:
        padded_logprobs = [mx.pad(logprob, (0, max_logprob_len - logprob.shape[0]), constant_values=0.0)
                          if logprob.shape[0] < max_logprob_len else logprob[:max_logprob_len]
                          for logprob in non_empty_logprobs]
    else:
        padded_logprobs = []

    return {
        'obs': mx.concatenate(padded_obs) if padded_obs else mx.array([], dtype=mx.int64),
        'act': mx.concatenate(padded_acts) if padded_acts else mx.array([], dtype=mx.int64),
        'logprob': mx.concatenate([lp.flatten() for lp in padded_logprobs]) if padded_logprobs else mx.array([], dtype=mx.float32),
        'rewards': mx.array(episode_rewards),
        'episode_lengths': episode_lengths,
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
