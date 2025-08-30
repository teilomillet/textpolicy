# textpolicy/rewards/registry.py
"""
Unified reward and verifier registry system following retrain's philosophy.

This module provides decorator-based registration for rewards and verifiers,
maintaining compatibility with MLX optimization and pure function composition.

Key principles:
- Decorator-based registration (@reward, @verifier)
- Function signature consistency: (prompt, completion, example, **kwargs)
- Pre-filtering verification approach
- Global registries for modularity
- MLX compilation support
"""

from typing import Callable, Dict, Any, List, Optional, Union
import inspect
import functools
import mlx.core as mx
from dataclasses import dataclass

# Type definitions following retrain's patterns
RewardFunction = Callable[[str, str, Dict[str, Any]], float]
VerifierFunction = Callable[[str, str, Dict[str, Any]], bool]

# Global registries following retrain's architecture
REWARD_REGISTRY: Dict[str, RewardFunction] = {}
VERIFIER_REGISTRY: Dict[str, VerifierFunction] = {}

# Simple logging
import logging
logger = logging.getLogger(__name__)


def reward(_func: Optional[RewardFunction] = None, *, name: Optional[str] = None) -> Union[Callable[[RewardFunction], RewardFunction], RewardFunction]:
    """
    Decorator to register reward functions following retrain's pattern.
    
    Usage:
        @reward
        def my_reward(prompt: str, completion: str, example: Dict[str, Any]) -> float:
            return 1.0
            
        @reward(name="custom_name")
        def another_reward(prompt: str, completion: str, example: Dict[str, Any]) -> float:
            return 0.5
    """
    def decorator_reward(func: RewardFunction) -> RewardFunction:
        if not callable(func):
            raise TypeError(f"Object {getattr(func, '__name__', '<unknown>')} must be callable to be registered as reward.")
        
        registration_name = name if name is not None else func.__name__
        
        # Validate function signature for consistency
        sig = inspect.signature(func)
        expected_params = ['prompt', 'completion', 'example']
        
        if len(sig.parameters) < 3:
            logger.warning(f"Reward function '{registration_name}' has fewer than 3 expected parameters. Ensure signature compatibility.")
        
        param_names = list(sig.parameters.keys())
        for i, expected in enumerate(expected_params):
            if i < len(param_names) and param_names[i] != expected:
                logger.warning(f"Reward function '{registration_name}' parameter {i} is '{param_names[i]}', expected '{expected}'.")
        
        if registration_name in REWARD_REGISTRY:
            logger.warning(f"Reward function '{registration_name}' already registered. Overwriting.")
        
        REWARD_REGISTRY[registration_name] = func
        logger.info(f"Registered reward function: '{registration_name}' -> {func.__name__}")
        
        return func
    
    if _func is None:
        # Called with parentheses: @reward() or @reward(name=...)
        return decorator_reward
    elif callable(_func):
        # Called without parentheses: @reward
        if name is not None:
            raise TypeError("Cannot specify 'name' when using @reward without parentheses. Use @reward(name='...') instead.")
        return decorator_reward(_func)
    else:
        raise TypeError("Invalid arguments supplied to @reward decorator.")


def verifier(_func: Optional[VerifierFunction] = None, *, name: Optional[str] = None) -> Union[Callable[[VerifierFunction], VerifierFunction], VerifierFunction]:
    """
    Decorator to register verifier functions following retrain's pattern.
    
    Usage:
        @verifier
        def has_greeting(prompt: str, completion: str, example: Dict[str, Any]) -> bool:
            return completion.lower().startswith("hello")
            
        @verifier(name="custom_check")
        def custom_verifier(prompt: str, completion: str, example: Dict[str, Any]) -> bool:
            return len(completion) > 10
    """
    def decorator_verifier(func: VerifierFunction) -> VerifierFunction:
        if not callable(func):
            raise TypeError(f"Object {getattr(func, '__name__', '<unknown>')} must be callable to be registered as verifier.")
        
        registration_name = name if name is not None else func.__name__
        
        # Validate function signature
        sig = inspect.signature(func)
        if len(sig.parameters) < 3:
            logger.warning(f"Verifier function '{registration_name}' has fewer than 3 expected parameters (prompt, completion, example).")
        
        if registration_name in VERIFIER_REGISTRY:
            logger.warning(f"Verifier function '{registration_name}' already registered. Overwriting.")
        
        VERIFIER_REGISTRY[registration_name] = func
        logger.info(f"Registered verifier function: '{registration_name}' -> {func.__name__}")
        
        return func
    
    if _func is None:
        return decorator_verifier
    elif callable(_func):
        if name is not None:
            raise TypeError("Cannot specify 'name' when using @verifier without parentheses. Use @verifier(name='...') instead.")
        return decorator_verifier(_func)
    else:
        raise TypeError("Invalid arguments supplied to @verifier decorator.")


def get_reward_function(name: str) -> Optional[RewardFunction]:
    """Retrieve a registered reward function by name."""
    func = REWARD_REGISTRY.get(name)
    if func is None:
        available = list(REWARD_REGISTRY.keys())
        logger.error(f"Reward function '{name}' not found. Available: {available}")
    return func


def get_verifier_function(name: str) -> Optional[VerifierFunction]:
    """Retrieve a registered verifier function by name."""
    func = VERIFIER_REGISTRY.get(name)
    if func is None:
        available = list(VERIFIER_REGISTRY.keys())
        logger.error(f"Verifier function '{name}' not found. Available: {available}")
    return func


def apply_verifiers_to_reward(
    original_reward_func: RewardFunction,
    verifier_names: List[str],
    penalty_on_failure: float = 0.0
) -> RewardFunction:
    """
    Apply verifiers to a reward function following retrain's pre-filtering approach.
    
    If any verifier fails, returns the penalty value without executing the reward function.
    This follows retrain's philosophy of efficient pre-filtering.
    
    Args:
        original_reward_func: The reward function to wrap
        verifier_names: List of verifier names to apply
        penalty_on_failure: Value to return if any verifier fails
        
    Returns:
        Wrapped reward function that applies verifiers first
    """
    # Load verifier functions
    loaded_verifiers: List[VerifierFunction] = []
    missing_verifiers = []
    
    for verifier_name in verifier_names:
        verifier_func = get_verifier_function(verifier_name)
        if verifier_func is None:
            missing_verifiers.append(verifier_name)
        else:
            loaded_verifiers.append(verifier_func)
    
    if missing_verifiers:
        available = list(VERIFIER_REGISTRY.keys())
        raise ValueError(f"Verifiers {missing_verifiers} not found for reward '{original_reward_func.__name__}'. Available: {available}")
    
    @functools.wraps(original_reward_func)
    def reward_with_verifiers(prompt: str, completion: str, example: Dict[str, Any], **kwargs) -> float:
        # Apply verifiers first (pre-filtering approach)
        for i, verifier_func in enumerate(loaded_verifiers):
            verifier_name = verifier_names[i]
            try:
                if not verifier_func(prompt, completion, example):
                    logger.debug(f"Verifier '{verifier_name}' failed for reward '{original_reward_func.__name__}'. Applying penalty: {penalty_on_failure}")
                    return penalty_on_failure
            except Exception as e:
                logger.error(f"Verifier '{verifier_name}' errored: {e}. Applying penalty: {penalty_on_failure}")
                return penalty_on_failure
        
        # All verifiers passed, execute reward function
        try:
            result = original_reward_func(prompt, completion, example, **kwargs)
            return float(result)
        except Exception as e:
            logger.error(f"Reward function '{original_reward_func.__name__}' errored after verifiers passed: {e}. Returning 0.0")
            return 0.0
    
    # Set descriptive name
    if verifier_names:
        verifier_suffix = '_and_'.join(verifier_names)
        reward_with_verifiers.__name__ = f"{original_reward_func.__name__}_verified_by_{verifier_suffix}"
    else:
        reward_with_verifiers.__name__ = original_reward_func.__name__
    
    return reward_with_verifiers


@dataclass
class RewardConfig:
    """Configuration for a reward function following retrain's patterns."""
    name: str  # Name in REWARD_REGISTRY
    weight: float = 1.0
    params: Optional[Dict[str, Any]] = None
    verifiers: Optional[List[str]] = None
    verifier_penalty: float = 0.0
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.verifiers is None:
            self.verifiers = []


def create_configured_reward_function(config: RewardConfig) -> RewardFunction:
    """
    Create a reward function from configuration following retrain's approach.
    
    This function:
    1. Loads the base reward function from registry
    2. Applies verifiers if specified
    3. Handles parameter passing
    4. Returns a configured function ready for use
    """
    # Get base reward function
    base_reward_func = get_reward_function(config.name)
    if base_reward_func is None:
        raise ValueError(f"Reward function '{config.name}' not found in registry")
    
    # Create wrapper that handles params
    def reward_with_params(prompt: str, completion: str, example: Dict[str, Any], **kwargs) -> float:
        # Merge config params with runtime kwargs
        merged_kwargs = {**config.params, **kwargs}
        return base_reward_func(prompt, completion, example, **merged_kwargs)
    
    reward_with_params.__name__ = f"{base_reward_func.__name__}_with_params"
    
    # Apply verifiers if specified
    if config.verifiers:
        reward_with_params = apply_verifiers_to_reward(
            reward_with_params, 
            config.verifiers, 
            config.verifier_penalty
        )
    
    return reward_with_params


# MLX optimization support
@mx.compile
def batch_reward_computation(
    base_rewards: mx.array,
    weights: mx.array
) -> mx.array:
    """
    MLX-compiled function for efficient batch reward computation.
    
    Args:
        base_rewards: Individual reward scores [batch_size, num_rewards]
        weights: Reward weights [num_rewards]
        
    Returns:
        Weighted combined rewards [batch_size]
    """
    return mx.sum(base_rewards * weights, axis=1)


def list_registered_functions() -> Dict[str, List[str]]:
    """List all registered reward and verifier functions."""
    return {
        "rewards": list(REWARD_REGISTRY.keys()),
        "verifiers": list(VERIFIER_REGISTRY.keys())
    }


def clear_registries():
    """Clear all registries (useful for testing)."""
    global REWARD_REGISTRY, VERIFIER_REGISTRY
    REWARD_REGISTRY.clear()
    VERIFIER_REGISTRY.clear()
    logger.info("Cleared all reward and verifier registries")