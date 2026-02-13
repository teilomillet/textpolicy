# textpolicy/rollout/strategy.py
"""
Algorithm-specific rollout strategies.
"""

from typing import Callable, Dict, Any, Tuple
import mlx.core as mx # type: ignore
from textpolicy.buffer import Buffer
from .base import RolloutStrategy, validate_transition_data


class PPOStrategy(RolloutStrategy):
    """
    Rollout strategy for Proximal Policy Optimization (PPO).

    PPO requires:
    - Action probabilities (logprob) for policy gradient
    - Value function estimates for advantage calculation
    - Policy and value function trained together

    Expected policy output:
        (action, {"logprob": mx.array, "value": mx.array, "entropy": mx.array})

    Stored in buffer: obs, act, rew, next_obs, done, timeout, logprob, value
    Filtered out: entropy (computed during training, not stored)
    """

    def select_action(self, policy: Callable, obs: mx.array) -> Tuple[mx.array, Dict[str, Any]]:
        """
        Select action using PPO policy.

        PPO TRAINING BEHAVIOR: Uses stochastic policy (deterministic=False) for exploration
        during training data collection. This provides the variety of experiences needed
        for robust policy learning. Evaluation uses deterministic=True for consistent
        performance measurement.

        Args:
            policy: Policy function returning (action, extras)
            obs: MLX array observation

        Returns:
            action: Selected action as MLX array (sampled stochastically)
            extras: Dict with logprob, value, entropy
        """
        # Use stochastic policy for training data collection (explore during training, evaluate deterministically)
        return policy(obs, deterministic=False)

    def store_transition(self, buffer: Buffer, **data) -> None:
        """
        Store PPO transition data in buffer.

        Filters data to include only what the buffer supports and PPO needs.
        Validates required fields are present.

        Args:
            buffer: Buffer instance to store data
            **data: Transition data including obs, act, rew, etc.
        """
        # Validate and filter transition data
        filtered_data = validate_transition_data(data)

        # Remove entropy if present (not stored, computed during training)
        filtered_data.pop('entropy', None)

        # Store in buffer
        buffer.add(**filtered_data) # type: ignore


class GRPOStrategy(RolloutStrategy):
    """
    Rollout strategy for Group Relative Policy Optimization (GRPO).

    GRPO characteristics:
    - No value function required (uses group-relative advantages)
    - Only needs action probabilities for policy gradient
    - Advantage computed relative to group performance

    Expected policy output:
        (action, {"logprob": mx.array, "entropy": mx.array})

    Stored in buffer: obs, act, rew, next_obs, done, timeout, logprob
    Filtered out: value (not used), entropy (computed during training)
    """

    def select_action(self, policy: Callable, obs: mx.array) -> Tuple[mx.array, Dict[str, Any]]:
        """
        Select action using GRPO policy.

        GRPO TRAINING BEHAVIOR: Uses stochastic policy (deterministic=False) for exploration
        during training data collection, consistent with PPO approach.

        Args:
            policy: Policy function returning (action, extras)
            obs: MLX array observation

        Returns:
            action: Selected action as MLX array (sampled stochastically)
            extras: Dict with logprob, entropy (no value function)
        """
        # Use stochastic policy for training data collection
        return policy(obs, deterministic=False)

    def store_transition(self, buffer: Buffer, **data) -> None:
        """
        Store GRPO transition data in buffer.

        GRPO doesn't use value functions, so value data is filtered out.
        Only stores what's needed for group-relative advantage computation.

        Args:
            buffer: Buffer instance to store data
            **data: Transition data including obs, act, rew, etc.
        """
        # Validate and filter transition data
        filtered_data = validate_transition_data(data)

        # Remove fields not used by GRPO
        filtered_data.pop('value', None)    # No value function in GRPO
        filtered_data.pop('entropy', None)  # Computed during training

        # Store in buffer
        buffer.add(**filtered_data) # type: ignore


# Strategy registry for factory pattern
STRATEGY_REGISTRY = {
    'ppo': PPOStrategy,
    'grpo': GRPOStrategy,
    'gspo': GRPOStrategy,  # Add alias for backwards compatibility
    'maxrl': GRPOStrategy,  # Same rollout as GRPO; differs only in advantage_fn
}


def create_strategy(algorithm: str) -> RolloutStrategy:
    """
    Factory function for creating rollout strategies.

    Args:
        algorithm: Algorithm name ('ppo', 'grpo', 'gspo', 'maxrl')

    Returns:
        RolloutStrategy instance

    Raises:
        ValueError: If algorithm is not supported
    """
    if algorithm not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

    strategy_class = STRATEGY_REGISTRY[algorithm]
    return strategy_class()
