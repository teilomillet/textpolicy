# textpolicy/rewards/mlx_batch_processor.py
"""
MLX-optimized batch processing system following DESIGN_GUIDELINES.md principles.

This module implements pure function composition with zero abstraction cost,
designed for maximum efficiency on Apple Silicon using MLX compilation.

Features:
- Pure function composition (no classes, no dispatch)
- MLX compilation for optimal performance
- Vectorized batch processing
- Single interface for all reward/verifier combinations
- Integrates with retrain's philosophy
"""

import mlx.core as mx
from typing import List, Dict, Any, Callable, Coroutine, Union
import asyncio

from .registry import (
    RewardFunction, RewardConfig,
    get_reward_function,
    apply_verifiers_to_reward, REWARD_REGISTRY, VERIFIER_REGISTRY
)


# Pure function composition following DESIGN_GUIDELINES.md Option 3
def create_batch_reward_processor(
    reward_configs: List[RewardConfig],
    enable_mlx_compilation: bool = True
) -> Callable[[List[str], List[str], List[Dict[str, Any]]], mx.array]:
    """
    Pure function factory for creating MLX-optimized batch processors.
    
    Following DESIGN_GUIDELINES.md:
    - Pure function composition over class hierarchies
    - Zero abstraction cost
    - Single training loop for all algorithms
    - MLX compilation ready
    
    Args:
        reward_configs: List of reward configurations
        enable_mlx_compilation: Whether to enable MLX compilation
        
    Returns:
        Pure function: (prompts, completions, examples) -> rewards [batch_size]
    """
    # Pre-load all reward functions (zero cost at runtime)
    loaded_functions: List[RewardFunction] = []
    weights: List[float] = []
    
    for config in reward_configs:
        # Create configured reward function with verifiers
        reward_func = create_configured_reward_function(config)
        loaded_functions.append(reward_func)
        weights.append(config.weight)
    
    # Convert weights to MLX array for efficient computation
    weights_array = mx.array(weights)
    
    # Pure function implementation
    def batch_processor(
        prompts: List[str], 
        completions: List[str], 
        examples: List[Dict[str, Any]]
    ) -> mx.array:
        """
        Process batch of episodes with pure function composition.
        
        Args:
            prompts: List of input prompts
            completions: List of generated completions
            examples: List of example contexts
            
        Returns:
            Combined rewards [batch_size]
        """
        batch_size = len(prompts)
        
        # Compute rewards for each function
        all_rewards = []
        
        for func_idx, reward_func in enumerate(loaded_functions):
            batch_rewards = []
            
            # Process each sample
            for i in range(batch_size):
                try:
                    reward = reward_func(prompts[i], completions[i], examples[i])
                    batch_rewards.append(float(reward))
                except Exception:
                    batch_rewards.append(0.0)
            
            all_rewards.append(mx.array(batch_rewards))
        
        # Stack rewards and apply weights
        if all_rewards:
            reward_matrix = mx.stack(all_rewards, axis=1)  # [batch_size, num_functions]
            return mx.sum(reward_matrix * weights_array, axis=1)
        else:
            return mx.zeros(batch_size)
    
    # Optionally compile for maximum performance
    if enable_mlx_compilation:
        # Create compiled version of core computation
        @mx.compile
        def compiled_weighted_sum(reward_matrix: mx.array, weights: mx.array) -> mx.array:
            return mx.sum(reward_matrix * weights, axis=1)
        
        # Update function to use compiled computation
        def optimized_processor(prompts, completions, examples):
            # Use original function for reward computation
            reward_matrix = mx.zeros((len(prompts), len(loaded_functions)))
            
            for func_idx, reward_func in enumerate(loaded_functions):
                batch_rewards = []
                for i in range(len(prompts)):
                    try:
                        reward = reward_func(prompts[i], completions[i], examples[i])
                        batch_rewards.append(float(reward))
                    except Exception:
                        batch_rewards.append(0.0)
                
                # Calculate difference between new values and current values at the specified indices
                current_values = reward_matrix[:, func_idx]
                new_values = mx.array(batch_rewards)
                diff = new_values - current_values
                
                # Use add method to update values at specified indices
                reward_matrix = reward_matrix.at[:, func_idx].add(diff)
            
            # Use compiled function for final computation
            return compiled_weighted_sum(reward_matrix, weights_array)
        
        return optimized_processor
    
    return batch_processor


def create_configured_reward_function(config: RewardConfig) -> RewardFunction:
    """
    Create a configured reward function following retrain's patterns.
    
    This is a pure function that creates other pure functions,
    maintaining zero abstraction cost.
    """
    # Get base reward function
    base_reward_func = get_reward_function(config.name)
    if base_reward_func is None:
        raise ValueError(f"Reward function '{config.name}' not found in registry")
    
    # Create parameter-injected function
    def reward_with_params(prompt: str, completion: str, example: Dict[str, Any]) -> float:
        # Merge config params with example
        merged_example = {**example, **config.params}
        return base_reward_func(prompt, completion, merged_example)
    
    # Apply verifiers if specified (following retrain's pre-filtering)
    if config.verifiers:
        reward_with_params = apply_verifiers_to_reward(
            reward_with_params, 
            config.verifiers, 
            config.verifier_penalty
        )
    
    return reward_with_params


# MLX-compiled vectorized operations for maximum performance
@mx.compile
def compute_length_rewards_vectorized(
    completion_lengths: mx.array,
    target_length: float,
    tolerance: float
) -> mx.array:
    """
    MLX-compiled vectorized length reward computation.
    
    Args:
        completion_lengths: Word counts [batch_size]
        target_length: Target word count
        tolerance: Tolerance fraction
        
    Returns:
        Length rewards [batch_size]
    """
    deviations = mx.abs(completion_lengths - target_length) / target_length
    
    # Vectorized conditional computation
    within_tolerance = deviations <= tolerance
    beyond_tolerance = ~within_tolerance
    
    # Linear decay for beyond tolerance
    decay_rewards = mx.maximum(
        0.0, 
        1.0 - (deviations - tolerance) / (1.0 - tolerance)
    )
    
    # Combine results
    rewards = within_tolerance * 1.0 + beyond_tolerance * decay_rewards
    return rewards


@mx.compile  
def compute_keyword_rewards_vectorized(
    keyword_matches: mx.array,
    total_keywords: mx.array,
    bonus_matches: mx.array,
    total_bonus_keywords: mx.array,
    bonus_multiplier: float
) -> mx.array:
    """
    MLX-compiled vectorized keyword reward computation.
    
    Args:
        keyword_matches: Number of keyword matches [batch_size]
        total_keywords: Total keywords for each sample [batch_size]
        bonus_matches: Bonus keyword matches [batch_size]
        total_bonus_keywords: Total bonus keywords [batch_size]
        bonus_multiplier: Bonus multiplier
        
    Returns:
        Keyword rewards [batch_size]
    """
    # Avoid division by zero
    safe_total_keywords = mx.maximum(total_keywords, 1.0)
    safe_total_bonus = mx.maximum(total_bonus_keywords, 1.0)
    
    base_rewards = keyword_matches / safe_total_keywords
    bonus_rewards = (bonus_matches / safe_total_bonus) * bonus_multiplier
    
    # Clip to reasonable range
    total_rewards = mx.minimum(base_rewards + bonus_rewards, 2.0)
    return total_rewards


def create_mlx_optimized_batch_processor(
    reward_configs: List[RewardConfig]
) -> Callable[[List[str], List[str], List[Dict[str, Any]]], mx.array]:
    """
    Create fully MLX-optimized batch processor for maximum Apple Silicon performance.
    
    This implementation follows DESIGN_GUIDELINES.md by:
    1. Using pure function composition
    2. Maximizing MLX compilation opportunities
    3. Minimizing memory allocations
    4. Utilizing unified memory efficiently
    """
    
    # Pre-compile all possible reward components
    compiled_functions = {}
    
    # Check which reward types we need and pre-compile them
    for config in reward_configs:
        if config.name == 'length_reward':
            compiled_functions['length'] = compute_length_rewards_vectorized
        elif config.name == 'keyword_reward':
            compiled_functions['keyword'] = compute_keyword_rewards_vectorized
    
    weights = mx.array([config.weight for config in reward_configs])
    
    def optimized_batch_processor(
        prompts: List[str],
        completions: List[str], 
        examples: List[Dict[str, Any]]
    ) -> mx.array:
        """
        Fully optimized batch processor using MLX compilation.
        """
        batch_size = len(prompts)
        
        # Collect all reward arrays
        all_rewards = []
        
        # Process each reward type
        for config_idx, config in enumerate(reward_configs):
            if config.name == 'length_reward' and 'length' in compiled_functions:
                # Use vectorized length computation
                lengths = mx.array([len(comp.split()) for comp in completions])
                target = config.params.get('target_length', 50)
                tolerance = config.params.get('tolerance', 0.2)
                
                rewards = compiled_functions['length'](lengths, float(target), tolerance)
                all_rewards.append(rewards)
                
            elif config.name == 'keyword_reward' and 'keyword' in compiled_functions:
                # Use vectorized keyword computation
                keywords = config.params.get('keywords', [])
                if keywords:
                    # Preprocess keyword matches
                    keyword_matches = []
                    bonus_matches = []
                    
                    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                        comp_lower = completion.lower()
                        prompt_lower = prompt.lower()
                        
                        matches = sum(1 for kw in keywords if kw.lower() in comp_lower)
                        bonus_kws = [kw for kw in keywords if kw.lower() not in prompt_lower]
                        bonus = sum(1 for kw in bonus_kws if kw.lower() in comp_lower)
                        
                        keyword_matches.append(matches)
                        bonus_matches.append(bonus)
                    
                    # Vectorized computation
                    match_array = mx.array(keyword_matches)
                    bonus_array = mx.array(bonus_matches)
                    total_kw = mx.full((batch_size,), len(keywords))
                    total_bonus = mx.array([len([kw for kw in keywords if kw.lower() not in prompts[i].lower()]) for i in range(batch_size)])
                    multiplier = config.params.get('bonus_multiplier', 1.0)
                    
                    rewards = compiled_functions['keyword'](
                        match_array, total_kw, bonus_array, total_bonus, multiplier
                    )
                    all_rewards.append(rewards)
                else:
                    # No keywords specified
                    all_rewards.append(mx.zeros(batch_size))
            else:
                # Fallback to individual function calls
                reward_func = create_configured_reward_function(config)
                batch_rewards = []
                
                for i in range(batch_size):
                    try:
                        reward = reward_func(prompts[i], completions[i], examples[i])
                        batch_rewards.append(float(reward))
                    except Exception:
                        batch_rewards.append(0.0)
                
                all_rewards.append(mx.array(batch_rewards))
        
        # Combine all rewards
        if all_rewards:
            reward_matrix = mx.stack(all_rewards, axis=1)  # [batch_size, num_rewards]
            return mx.sum(reward_matrix * weights, axis=1)
        else:
            return mx.zeros(batch_size)
    
    return optimized_batch_processor


# Async processing for external reward models
async def create_async_batch_processor(
    reward_configs: List[RewardConfig],
    max_workers: int = 4
) -> Callable[[List[str], List[str], List[Dict[str, Any]]], Coroutine[Any, Any, mx.array]]:
    """
    Create async batch processor for external reward models.
    
    Maintains pure function composition while enabling async operations.
    """
    
    # Separate local and external reward configs
    local_configs = []
    external_configs = []
    
    for config in reward_configs:
        if config.params.get('external_url'):
            external_configs.append(config)
        else:
            local_configs.append(config)
    
    # Create local processor
    local_processor = create_mlx_optimized_batch_processor(local_configs) if local_configs else None
    
    async def async_batch_processor(
        prompts: List[str],
        completions: List[str],
        examples: List[Dict[str, Any]]
    ) -> mx.array:
        """Async batch processor combining local and external rewards."""
        batch_size = len(prompts)
        
        # Process local rewards synchronously 
        local_rewards = None
        if local_processor:
            local_rewards = local_processor(prompts, completions, examples)
        
        # Process external rewards asynchronously
        external_rewards = None
        if external_configs:
            # Placeholder for external API calls
            # In practice, this would make HTTP requests to external reward models
            external_rewards = mx.zeros((batch_size, len(external_configs)))
        
        # Combine results
        if local_rewards is not None and external_rewards is not None:
            all_rewards = mx.concatenate([local_rewards.reshape(-1, len(local_configs)), external_rewards], axis=1)
        elif local_rewards is not None:
            all_rewards = local_rewards.reshape(-1, len(local_configs))
        elif external_rewards is not None:
            all_rewards = external_rewards
        else:
            all_rewards = mx.zeros((batch_size, 1))
        
        # Apply final weights
        all_weights = mx.array([config.weight for config in reward_configs])
        return mx.sum(all_rewards * all_weights, axis=1)
    
    return async_batch_processor


# Utility functions for integration
def list_available_processors() -> Dict[str, List[str]]:
    """List all available reward and verifier functions."""
    return {
        "rewards": list(REWARD_REGISTRY.keys()),
        "verifiers": list(VERIFIER_REGISTRY.keys()),
        "compiled_optimizations": ["length_reward", "keyword_reward"]
    }


async def create_processor_from_config(
    config_dict: Dict[str, Any]
) -> Union[Callable[[List[str], List[str], List[Dict[str, Any]]], mx.array],
           Callable[[List[str], List[str], List[Dict[str, Any]]], Coroutine[Any, Any, mx.array]]]:
    """
    Create processor from configuration dictionary following retrain's patterns.
    
    Args:
        config_dict: Configuration with reward specifications
        
    Returns:
        Batch processor function
    """
    reward_configs = []
    
    for name, config in config_dict.items():
        reward_config = RewardConfig(
            name=name,
            weight=config.get('weight', 1.0),
            params=config.get('params', {}),
            verifiers=config.get('verifiers', []),
            verifier_penalty=config.get('verifier_penalty', 0.0)
        )
        reward_configs.append(reward_config)
    
    # Choose optimal processor based on configuration
    has_external = any(cfg.params.get('external_url') for cfg in reward_configs)
    
    if has_external:
        # Use async processor for external models
        return await create_async_batch_processor(reward_configs)
    else:
        # Use MLX-optimized processor for local computation
        return create_mlx_optimized_batch_processor(reward_configs)
