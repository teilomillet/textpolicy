# mlx_rl/environment/base.py
"""
Base environment interface and protocols for MLX-RL.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Protocol


class Environment(ABC):
    """
    Unified environment interface for all environment types.
    
    This abstract base class defines the contract that all environments
    must implement to work with MLX-RL agents and trainers.
    """

    @abstractmethod
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment and return initial observation.
        
        Returns:
            Tuple of (observation, info) following gymnasium API
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Dict[str, Any]:
        """
        Take action and return step result.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Dict with keys: observation, reward, terminated, truncated, info
        """
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        """
        Observation space specification.
        
        Returns:
            Space object describing valid observations
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        """
        Action space specification.
        
        Returns:
            Space object describing valid actions
        """
        pass

    def clone(self) -> 'Environment':
        """
        Create a clone of this environment for multiprocessing.
        
        Default implementation raises NotImplementedError.
        Subclasses should override if they support cloning.
        
        Returns:
            New instance of the same environment
        """
        # Default clone implementation: use deepcopy to support multiprocessing clones
        import copy
        return copy.deepcopy(self)

    def render(self, mode: str = "human") -> Any:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode (e.g., "human", "rgb_array")
            
        Returns:
            Rendered output (depends on mode)
        """
        # Default implementation does nothing
        pass

    def close(self):
        """Clean up environment resources."""
        # Default implementation does nothing
        pass


class EnvironmentAdapter(Protocol):
    """
    Protocol for environment adapters.
    
    Adapters convert external environment APIs (gym, dm_env, etc.)
    to the unified Environment interface.
    """
    
    def __init__(self, env_spec: Any, **kwargs):
        """Initialize adapter with environment specification."""
        ...
    
    def clone(self) -> Environment:
        """Create a clone for multiprocessing."""
        ... 
