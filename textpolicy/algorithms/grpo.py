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
    min_variance: float = 0.01
) -> Tuple[List[Any], Dict[str, int]]:
    """
    Filter episodes to keep only informative prompts (DAPO dynamic sampling).

    Removes prompts where all completions have same outcome:
    - All correct (reward ~1.0): no learning signal (nothing to improve)
    - All wrong (reward ~0.0): no positive signal (can't learn what works)

    GRPO uses group-relative advantages. If all completions have the same
    outcome, advantages are zero, producing no gradient and wasting compute.

    Args:
        episodes: List of episodes (Episode objects or serialized dicts)
        min_variance: Minimum reward variance to keep a prompt group.
                     Groups with variance below this threshold are filtered out.
                     Default 0.01 filters prompts with essentially identical rewards.

    Returns:
        Tuple of:
        - filtered: List of episodes from informative prompts
        - stats: Dictionary with filtering statistics:
            - 'prompts_kept': Number of prompt groups kept
            - 'prompts_dropped_all_correct': Prompts where all completions succeeded
            - 'prompts_dropped_all_wrong': Prompts where all completions failed
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
        https://arxiv.org/abs/2503.14476

        GRPO++ Tricks
        https://cameronrwolfe.substack.com/p/grpo-tricks
    """
    if not episodes:
        return [], {
            'prompts_kept': 0,
            'prompts_dropped_all_correct': 0,
            'prompts_dropped_all_wrong': 0,
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
        'episodes_kept': 0,
        'episodes_dropped': 0,
    }

    for prompt_key, group in prompt_groups.items():
        # Get rewards for all completions in this group
        rewards = mx.array([_get_episode_reward(ep) for ep in group])
        variance = mx.var(rewards).item()
        mean_reward = mx.mean(rewards).item()

        if variance > min_variance:
            # Informative: keep all episodes from this prompt
            filtered.extend(group)
            stats['prompts_kept'] += 1
            stats['episodes_kept'] += len(group)
        else:
            # Uninformative: all completions have same outcome
            stats['episodes_dropped'] += len(group)
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
    
    # Use all available data - GRPO benefits from larger groups
    episodes_data = buffer.sample()  # This returns concatenated transitions
    
    # We need to convert this back to episode structure for reward extraction
    episodes = buffer.episodes  # Access episodes directly from storage
    
    if not episodes:
        raise ValueError("Buffer is empty - no episodes to train on")
    
    # Extract episode rewards for advantage computation
    episode_rewards = []
    episode_lengths = []
    
    # Collect all transitions
    all_obs = []
    all_acts = []
    all_logprobs = []
    
    for episode in episodes:
        # Episode reward (sum of all rewards in episode)
        # Handle both Episode objects and serialized dictionaries
        if hasattr(episode, 'rew'):
            # Episode object with attributes
            episode_reward = mx.sum(mx.array(episode.rew)).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode.obs))
            
            # Collect transitions
            # For proper logprob extraction during training, we need the full context (prompt + response)
            # This matches how the model was called during rollout generation
            # Flatten nested token sequences to create uniform token arrays
            
            # Extract and flatten observation tokens (prompt)
            flattened_obs = []
            for obs in episode.obs:
                if hasattr(obs, 'tolist'):  # MLX array
                    flattened_obs.extend(obs.tolist())
                elif isinstance(obs, list):  # Python list
                    flattened_obs.extend(obs)
                else:  # Single token
                    flattened_obs.append(obs)
            
            # Extract and flatten action tokens (response)
            flattened_acts = []
            for act in episode.act:
                if hasattr(act, 'tolist'):  # MLX array
                    flattened_acts.extend(act.tolist())
                elif isinstance(act, list):  # Python list
                    flattened_acts.extend(act)
                else:  # Single token
                    flattened_acts.append(act)
            
            # Create full sequence: [prompt_tokens..., response_tokens...]
            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            all_logprobs.append(episode.logprob if episode.logprob else [])
        else:
            # Serialized dictionary from multiprocessing
            episode_reward = mx.sum(episode['rew']).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode['obs']))
            
            # Collect transitions
            # For proper logprob extraction during training, we need the full context (prompt + response)
            # This matches how the model was called during rollout generation
            full_sequence = episode['obs'] + episode['act']  # Concatenate prompt + response
            all_obs.append(full_sequence)
            all_acts.append(episode['act'])
            all_logprobs.append(episode.get('logprob', []))
    
    # Convert Python lists to MLX arrays before concatenation
    # This is required because Episode objects store data as Python lists for memory efficiency
    # For proper logprob extraction, we need uniform-length sequences, so we pad to the maximum length
    
    # Find maximum sequence length for padding
    max_obs_len = max(len(obs) for obs in all_obs) if all_obs else 0
    max_act_len = max(len(act) for act in all_acts) if all_acts else 0
    max_logprob_len = max(len(logprob) for logprob in all_logprobs) if all_logprobs else 0
    
    # MLX-native padding and array operations for optimal Apple Silicon performance
    # Convert all sequences to MLX arrays and pad directly in MLX space
    try:
        # Convert all sequences to MLX arrays first (staying in unified memory)
        all_obs_mx = [mx.array(obs, dtype=mx.int64) for obs in all_obs if obs]
        all_acts_mx = [mx.array(act, dtype=mx.int64) for act in all_acts if act]
        all_logprobs_mx = [mx.array(logprob, dtype=mx.float32) for logprob in all_logprobs if logprob]
        
        # Pad using native MLX operations (more efficient for Apple Silicon)
        if all_obs_mx:
            padded_obs_mx = [mx.pad(obs, (0, max_obs_len - obs.shape[0]), constant_values=0) 
                           if obs.shape[0] < max_obs_len else obs[:max_obs_len] 
                           for obs in all_obs_mx]
        else:
            padded_obs_mx = []
            
        if all_acts_mx:
            padded_acts_mx = [mx.pad(act, (0, max_act_len - act.shape[0]), constant_values=0) 
                            if act.shape[0] < max_act_len else act[:max_act_len] 
                            for act in all_acts_mx]
        else:
            padded_acts_mx = []
            
        if all_logprobs_mx:
            padded_logprobs_mx = [mx.pad(logprob, (0, max_logprob_len - logprob.shape[0]), constant_values=0.0) 
                                if logprob.shape[0] < max_logprob_len else logprob[:max_logprob_len] 
                                for logprob in all_logprobs_mx]
        else:
            padded_logprobs_mx = []
        
        # Use padded MLX arrays directly (no intermediate conversion needed)
        all_obs_mx = padded_obs_mx
        all_acts_mx = padded_acts_mx  
        all_logprobs_mx = padded_logprobs_mx
        
    except Exception as e:
        print(f"ERROR in MLX array conversion: {e}")
        print(f"DEBUG: all_obs types: {[type(obs) for obs in all_obs[:3]]}")  # Show first 3 for brevity
        print(f"DEBUG: all_logprobs types: {[type(logprob) for logprob in all_logprobs[:3]]}")
        raise
    
    # GRPO data structure: both observations and actions as flat concatenated sequences
    # This matches the expected format for GRPO logprob extraction function
    batch_data = {
        'obs': mx.concatenate(all_obs_mx) if all_obs_mx else mx.array([]),  # Flat concatenated full sequences
        'act': mx.concatenate(all_acts_mx) if all_acts_mx else mx.array([]),  # Flat concatenated response tokens
        'logprob': mx.concatenate([logprob.flatten() for logprob in all_logprobs_mx]) if all_logprobs_mx else mx.array([]),  # Flat sequence for training
        'rewards': mx.array(episode_rewards),
        'episode_lengths': episode_lengths
    }
    
    return batch_data


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

    # Process filtered episodes (same logic as select_all_data)
    episode_rewards = []
    episode_lengths = []
    all_obs = []
    all_acts = []
    all_logprobs = []

    for episode in filtered_episodes:
        if hasattr(episode, 'rew'):
            episode_reward = mx.sum(mx.array(episode.rew)).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode.obs))

            flattened_obs = []
            for obs in episode.obs:
                if hasattr(obs, 'tolist'):
                    flattened_obs.extend(obs.tolist())
                elif isinstance(obs, list):
                    flattened_obs.extend(obs)
                else:
                    flattened_obs.append(obs)

            flattened_acts = []
            for act in episode.act:
                if hasattr(act, 'tolist'):
                    flattened_acts.extend(act.tolist())
                elif isinstance(act, list):
                    flattened_acts.extend(act)
                else:
                    flattened_acts.append(act)

            full_sequence = flattened_obs + flattened_acts
            all_obs.append(full_sequence)
            all_acts.append(flattened_acts)
            all_logprobs.append(episode.logprob if episode.logprob else [])
        else:
            episode_reward = mx.sum(episode['rew']).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode['obs']))

            obs_as_lists = []
            for obs_item in episode['obs']:
                if hasattr(obs_item, 'tolist'):
                    obs_as_lists.extend(obs_item.tolist())
                elif isinstance(obs_item, list):
                    obs_as_lists.extend(obs_item)
                else:
                    obs_as_lists.append(obs_item)

            act_as_lists = []
            for act_item in episode['act']:
                if hasattr(act_item, 'tolist'):
                    act_as_lists.extend(act_item.tolist())
                elif isinstance(act_item, list):
                    act_as_lists.extend(act_item)
                else:
                    act_as_lists.append(act_item)

            full_sequence = obs_as_lists + act_as_lists
            all_obs.append(full_sequence)
            all_acts.append(act_as_lists)
            all_logprobs.append(episode.get('logprob', []))

    # Convert to MLX arrays with padding
    max_obs_len = max(len(obs) for obs in all_obs) if all_obs else 0
    max_act_len = max(len(act) for act in all_acts) if all_acts else 0
    max_logprob_len = max(len(logprob) for logprob in all_logprobs) if all_logprobs else 0

    try:
        all_obs_mx = [mx.array(obs, dtype=mx.int64) for obs in all_obs if obs]
        all_acts_mx = [mx.array(act, dtype=mx.int64) for act in all_acts if act]
        all_logprobs_mx = [mx.array(logprob, dtype=mx.float32) for logprob in all_logprobs if logprob]

        if all_obs_mx:
            padded_obs_mx = [mx.pad(obs, (0, max_obs_len - obs.shape[0]), constant_values=0)
                           if obs.shape[0] < max_obs_len else obs[:max_obs_len]
                           for obs in all_obs_mx]
        else:
            padded_obs_mx = []

        if all_acts_mx:
            padded_acts_mx = [mx.pad(act, (0, max_act_len - act.shape[0]), constant_values=0)
                            if act.shape[0] < max_act_len else act[:max_act_len]
                            for act in all_acts_mx]
        else:
            padded_acts_mx = []

        if all_logprobs_mx:
            padded_logprobs_mx = [mx.pad(logprob, (0, max_logprob_len - logprob.shape[0]), constant_values=0.0)
                                if logprob.shape[0] < max_logprob_len else logprob[:max_logprob_len]
                                for logprob in all_logprobs_mx]
        else:
            padded_logprobs_mx = []

        all_obs_mx = padded_obs_mx
        all_acts_mx = padded_acts_mx
        all_logprobs_mx = padded_logprobs_mx

    except Exception as e:
        print(f"ERROR in MLX array conversion: {e}")
        raise

    batch_data = {
        'obs': mx.concatenate(all_obs_mx) if all_obs_mx else mx.array([]),
        'act': mx.concatenate(all_acts_mx) if all_acts_mx else mx.array([]),
        'logprob': mx.concatenate([logprob.flatten() for logprob in all_logprobs_mx]) if all_logprobs_mx else mx.array([]),
        'rewards': mx.array(episode_rewards),
        'episode_lengths': episode_lengths,
        'filter_stats': filter_stats,  # Include filtering statistics
    }

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
    
    # Process recent episodes
    episode_rewards = []
    episode_lengths = []
    all_obs = []
    all_acts = []
    all_logprobs = []
    
    for episode in recent_episodes:
        # Handle both Episode objects and serialized dictionaries
        if hasattr(episode, 'rew'):
            # Episode object with attributes
            episode_reward = mx.sum(mx.array(episode.rew)).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode.obs))
            
            # For proper logprob extraction during training, we need the full context (prompt + response)
            # This matches how the model was called during rollout generation
            # Convert both obs and act to consistent Python list format before concatenation
            obs_as_lists = []
            for obs_item in episode.obs:
                if hasattr(obs_item, 'tolist'):  # MLX array
                    obs_as_lists.extend(obs_item.tolist())
                elif isinstance(obs_item, list):  # Already Python list
                    obs_as_lists.extend(obs_item)
                else:  # Single item
                    obs_as_lists.append(obs_item)
            
            act_as_lists = []
            for act_item in episode.act:
                if hasattr(act_item, 'tolist'):  # MLX array
                    act_as_lists.extend(act_item.tolist())
                elif isinstance(act_item, list):  # Already Python list
                    act_as_lists.extend(act_item)
                else:  # Single item
                    act_as_lists.append(act_item)
            
            # Now concatenate the normalized lists
            full_sequence = obs_as_lists + act_as_lists
            all_obs.append(full_sequence)
            
            # Extract actions as consistent Python lists
            episode_actions = []
            for act_item in episode.act:
                if hasattr(act_item, 'tolist'):  # MLX array
                    episode_actions.extend(act_item.tolist())
                elif isinstance(act_item, list):  # Already Python list
                    episode_actions.extend(act_item)
                else:  # Single item
                    episode_actions.append(act_item)
            all_acts.append(episode_actions)
            
            # Extract logprobs as consistent Python lists
            episode_logprobs = []
            if episode.logprob:
                for logprob_item in episode.logprob:
                    if hasattr(logprob_item, 'tolist'):  # MLX array
                        episode_logprobs.extend(logprob_item.tolist())
                    elif isinstance(logprob_item, list):  # Already Python list
                        episode_logprobs.extend(logprob_item)
                    else:  # Single item
                        episode_logprobs.append(logprob_item)
            all_logprobs.append(episode_logprobs)
        else:
            # Serialized dictionary from multiprocessing
            episode_reward = mx.sum(episode['rew']).item()
            episode_rewards.append(episode_reward)
            episode_lengths.append(len(episode['obs']))
            
            # For proper logprob extraction during training, we need the full context (prompt + response)
            # This matches how the model was called during rollout generation
            # Convert both obs and act to consistent Python list format before concatenation
            obs_as_lists = []
            for obs_item in episode['obs']:
                if hasattr(obs_item, 'tolist'):  # MLX array
                    obs_as_lists.extend(obs_item.tolist())
                elif isinstance(obs_item, list):  # Already Python list
                    obs_as_lists.extend(obs_item)
                else:  # Single item
                    obs_as_lists.append(obs_item)
            
            act_as_lists = []
            for act_item in episode['act']:
                if hasattr(act_item, 'tolist'):  # MLX array
                    act_as_lists.extend(act_item.tolist())
                elif isinstance(act_item, list):  # Already Python list
                    act_as_lists.extend(act_item)
                else:  # Single item
                    act_as_lists.append(act_item)
            
            # Now concatenate the normalized lists
            full_sequence = obs_as_lists + act_as_lists
            all_obs.append(full_sequence)
            
            # Extract actions as consistent Python lists
            episode_actions = []
            for act_item in episode['act']:
                if hasattr(act_item, 'tolist'):  # MLX array
                    episode_actions.extend(act_item.tolist())
                elif isinstance(act_item, list):  # Already Python list
                    episode_actions.extend(act_item)
                else:  # Single item
                    episode_actions.append(act_item)
            all_acts.append(episode_actions)
            
            # Extract logprobs as consistent Python lists
            episode_logprobs = []
            if episode.get('logprob'):
                for logprob_item in episode['logprob']:
                    if hasattr(logprob_item, 'tolist'):  # MLX array
                        episode_logprobs.extend(logprob_item.tolist())
                    elif isinstance(logprob_item, list):  # Already Python list
                        episode_logprobs.extend(logprob_item)
                    else:  # Single item
                        episode_logprobs.append(logprob_item)
            all_logprobs.append(episode_logprobs)
    
    # Convert Python lists to MLX arrays before concatenation
    # This is required because Episode objects store data as Python lists for memory efficiency
    # For proper logprob extraction, we need uniform-length sequences, so we pad to the maximum length
    
    # Find maximum sequence length for padding
    max_obs_len = max(len(obs) for obs in all_obs) if all_obs else 0
    max_act_len = max(len(act) for act in all_acts) if all_acts else 0
    max_logprob_len = max(len(logprob) for logprob in all_logprobs) if all_logprobs else 0
    
    # MLX-native padding and array operations for optimal Apple Silicon performance  
    # Convert all sequences to MLX arrays and pad directly in MLX space
    try:
        # Convert all sequences to MLX arrays first (staying in unified memory)
        all_obs_mx = [mx.array(obs, dtype=mx.int64) for obs in all_obs if obs]
        all_acts_mx = [mx.array(act, dtype=mx.int64) for act in all_acts if act]  
        all_logprobs_mx = [mx.array(logprob, dtype=mx.float32) for logprob in all_logprobs if logprob]
        
        # Pad using native MLX operations (more efficient for Apple Silicon)
        if all_obs_mx:
            padded_obs_mx = [mx.pad(obs, (0, max_obs_len - obs.shape[0]), constant_values=0) 
                           if obs.shape[0] < max_obs_len else obs[:max_obs_len] 
                           for obs in all_obs_mx]
        else:
            padded_obs_mx = []
            
        if all_acts_mx:
            padded_acts_mx = [mx.pad(act, (0, max_act_len - act.shape[0]), constant_values=0) 
                            if act.shape[0] < max_act_len else act[:max_act_len] 
                            for act in all_acts_mx]
        else:
            padded_acts_mx = []
            
        if all_logprobs_mx:
            padded_logprobs_mx = [mx.pad(logprob, (0, max_logprob_len - logprob.shape[0]), constant_values=0.0) 
                                if logprob.shape[0] < max_logprob_len else logprob[:max_logprob_len] 
                                for logprob in all_logprobs_mx]
        else:
            padded_logprobs_mx = []
        
        # Use padded MLX arrays directly (no intermediate conversion needed)
        all_obs_mx = padded_obs_mx
        all_acts_mx = padded_acts_mx
        all_logprobs_mx = padded_logprobs_mx
        
    except Exception as e:
        print(f"ERROR in MLX array conversion: {e}")
        print(f"DEBUG: all_obs types: {[type(obs) for obs in all_obs[:3]]}")  # Show first 3 for brevity
        print(f"DEBUG: all_logprobs types: {[type(logprob) for logprob in all_logprobs[:3]]}")
        raise
    
    batch_data = {
        'obs': mx.concatenate(all_obs_mx) if all_obs_mx else mx.array([]),  # Flat concatenated full sequences
        'act': mx.concatenate(all_acts_mx) if all_acts_mx else mx.array([]),  # Flat concatenated response tokens
        'logprob': mx.concatenate([logprob.flatten() for logprob in all_logprobs_mx]) if all_logprobs_mx else mx.array([]),  # Flat sequence for training
        'rewards': mx.array(episode_rewards),
        'episode_lengths': episode_lengths
    }
    
    return batch_data
