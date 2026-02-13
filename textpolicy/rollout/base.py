# textpolicy/rollout/base.py
"""
Base classes and protocols for the rollout system.
"""

from typing import Callable, Dict, Any, Tuple, Protocol
import mlx.core as mx # type: ignore
from textpolicy.buffer import Buffer


class RolloutStrategy(Protocol):
    """
    Protocol for rollout strategies.
    
    Defines the interface used by RolloutRunner for algorithm-specific behavior.
    Each strategy encapsulates how to:
    - Select actions from policy outputs
    - Store transition data in buffers
    - Handle algorithm-specific requirements
    """
    
    def select_action(self, policy: Callable, obs: mx.array) -> Tuple[mx.array, Dict[str, Any]]:
        """
        Select an action using the policy.

        Args:
            policy: Function that takes obs and returns (action, extras)
            obs: MLX array observation

        Returns:
            action: mx.array (scalar or tensor)
            extras: Dict of additional data (e.g. logprob, value)
        """
        ...

    def store_transition(self, buffer: Buffer, **data) -> None:
        """
        Store a transition in the buffer.

        Args:
            buffer: Buffer instance
            **data: Transition data (obs, act, rew, next_obs, done, logprob, value, etc.)
        """
        ...


# Common constants and configurations
DEFAULT_MAX_STEPS = 1000
DEFAULT_MAX_EPISODES = 100
DEFAULT_WORKER_TIMEOUT = 1.0

# Supported transition data keys for validation
REQUIRED_TRANSITION_KEYS = {'obs', 'act', 'rew', 'next_obs', 'done'}
OPTIONAL_TRANSITION_KEYS = {'timeout', 'logprob', 'value', 'entropy', 'is_correct'}
ALL_TRANSITION_KEYS = REQUIRED_TRANSITION_KEYS | OPTIONAL_TRANSITION_KEYS


def validate_transition_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and filter transition data.
    
    Args:
        data: Dictionary containing transition data
        
    Returns:
        Filtered dictionary with only valid keys
        
    Raises:
        ValueError: If required keys are missing
    """
    # Check required keys
    missing_keys = REQUIRED_TRANSITION_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required transition keys: {missing_keys}")
    
    # Filter to only valid keys
    valid_data = {k: v for k, v in data.items() if k in ALL_TRANSITION_KEYS}
    
    return valid_data


def serialize_mx_array(arr: mx.array) -> Any:
    """
    Serialize MLX array for multiprocessing communication.
    
    Args:
        arr: MLX array to serialize
        
    Returns:
        Python scalar or list suitable for queue transmission
    """
    if arr.ndim == 0:
        return arr.item()  # Scalar
    else:
        return arr.tolist()  # List


def deserialize_to_mx_array(data: Any) -> mx.array:
    """
    Deserialize data back to MLX array.
    
    Args:
        data: Python scalar or list
        
    Returns:
        MLX array
    """
    return mx.array(data) 
