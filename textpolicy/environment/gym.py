# mlx_rl/environment/gym.py
"""
Gymnasium environment adapter for MLX-RL.
"""

from typing import Any, Dict, Tuple
import gymnasium as gym
from .base import Environment


class GymAdapter(Environment):
    """
    Adapter for gymnasium environments.
    
    Converts gymnasium environments to the unified Environment interface.
    Handles both old and new gymnasium APIs for maximum compatibility.
    """

    def __init__(self, env_name: str, **kwargs):
        """
        Initialize gymnasium environment.
        
        Args:
            env_name: Name of the gymnasium environment (e.g., "CartPole-v1")
            **kwargs: Additional arguments passed to gym.make()
        """
        self.env_name = env_name
        self.env_kwargs = kwargs
        self.env = gym.make(env_name, **kwargs)

    def clone(self) -> 'GymAdapter':
        """
        Create a new instance of the same environment.
        
        This is essential for multiprocessing where each worker
        needs its own environment instance.
        
        Returns:
            New GymAdapter instance with same configuration
        """
        return GymAdapter(self.env_name, **self.env_kwargs)

    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment and return initial observation.
        
        Handles both old gymnasium API (returns tuple) and
        newer API (returns observation, info).
        
        Returns:
            Tuple of (observation, info)
        """
        result = self.env.reset()
        if isinstance(result, tuple):
            return result  # New API: (obs, info)
        else:
            return result, {}  # Old API: just obs

    def step(self, action: Any) -> Dict[str, Any]:
        """
        Take action and return step result in unified format.
        
        Args:
            action: Action to take (format depends on action space)
            
        Returns:
            Dictionary with observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }

    @property
    def observation_space(self):
        """Get observation space from underlying gymnasium environment."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Get action space from underlying gymnasium environment."""
        return self.env.action_space

    def render(self, mode: str = "human") -> Any:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ("human", "rgb_array", etc.)
            
        Returns:
            Rendered output (depends on mode and environment)
        """
        return self.env.render()

    def close(self):
        """Close the underlying gymnasium environment."""
        self.env.close()

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"GymAdapter(env_name='{self.env_name}', kwargs={self.env_kwargs})" 