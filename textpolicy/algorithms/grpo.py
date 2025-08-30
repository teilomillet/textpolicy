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
from typing import List, Union


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
    clip_ratio: float = 0.2
) -> mx.array:
    """
    GRPO policy loss with PPO-style clipping.
    
    Uses clipped surrogate objective but with group-relative advantages
    instead of GAE advantages.
    
    Args:
        old_logprobs: Log probabilities from rollout collection
        new_logprobs: Log probabilities from current policy evaluation
        advantages: Group-relative advantages from compute_advantages()
        clip_ratio: Clipping ratio for surrogate objective
        
    Returns:
        Policy loss scalar (to be minimized)
        
    Notes:
    - Fully vectorized (no Python loops over batch)
    - Uses in-place operations where possible
    - Suitable for MLX graph optimization
    - Single forward pass through computation
    """
    # Importance ratio: π_new / π_old  
    # MLX optimizes exp() for Apple Silicon
    ratio = mx.exp(new_logprobs - old_logprobs)
    
    # PPO clipped surrogate objective
    # L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    clipped_ratio = mx.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    
    # Element-wise minimum and mean reduction
    # Negative because we minimize (original maximizes)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -mx.mean(mx.minimum(surr1, surr2))
    
    return loss


# Optional: Compiled versions for maximum performance
@mx.compile
def compute_advantages_compiled(rewards: mx.array) -> mx.array:
    """Compiled version of compute_advantages for maximum performance."""
    group_mean = mx.mean(rewards)
    return rewards - group_mean


@mx.compile  
def policy_loss_compiled(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    clip_ratio: float = 0.2
) -> mx.array:
    """Compiled version of policy_loss for maximum performance."""
    ratio = mx.exp(new_logprobs - old_logprobs)
    clipped_ratio = mx.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    return -mx.mean(mx.minimum(surr1, surr2))


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


# Convenience function for complete GRPO computation
def grpo_loss(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    rewards: Union[List[float], mx.array],
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.0
) -> mx.array:
    """
    Complete GRPO loss computation in one function.
    
    Combines advantage calculation and policy loss for convenience.
    Can be compiled as a single unit for maximum efficiency.
    
    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy
        rewards: Episode rewards for group-relative advantages
        clip_ratio: PPO clipping ratio
        entropy_coeff: Entropy bonus coefficient (0 disables)
        
    Returns:
        Total GRPO loss (policy + optional entropy)
    """
    # Compute group-relative advantages
    advantages = compute_advantages(rewards)
    
    # Expand advantages to match logprob sequence length if needed
    if advantages.ndim == 1 and old_logprobs.ndim > 1:
        # Each episode contributes its advantage to all tokens in that episode
        # This requires knowing episode boundaries - simplified version assumes
        # advantages and logprobs are already aligned
        pass
    
    # Compute policy loss
    policy_loss_val = policy_loss(old_logprobs, new_logprobs, advantages, clip_ratio)
    
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
    clip_ratio: float = 0.2
) -> dict:
    """
    Compute GRPO training metrics for monitoring.
    
    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy  
        advantages: Group-relative advantages
        clip_ratio: Clipping ratio used in loss
        
    Returns:
        Dictionary of metrics for logging/monitoring
    """
    # Importance ratio statistics
    ratio = mx.exp(new_logprobs - old_logprobs)
    
    # Clipping statistics
    clip_lower = 1 - clip_ratio
    clip_upper = 1 + clip_ratio
    clipped = (ratio < clip_lower) | (ratio > clip_upper)
    clip_fraction = mx.mean(clipped.astype(mx.float32))
    
    # KL divergence approximation
    kl_div = mx.mean(old_logprobs - new_logprobs)
    
    return {
        'mean_advantage': mx.mean(advantages).item(),
        'std_advantage': mx.std(advantages).item(),
        'mean_ratio': mx.mean(ratio).item(),
        'clip_fraction': clip_fraction.item(),
        'kl_divergence': kl_div.item(),
        'min_advantage': mx.min(advantages).item(),
        'max_advantage': mx.max(advantages).item()
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
