"""
Vectorized environment wrapper for MLX-RL to improve training performance.
"""

# Import debug utilities for proper debug management 
from textpolicy.utils.debug import vectorization_debug

from typing import Any, Dict, Optional, Tuple, cast

import gymnasium as gym
try:
    import mlx.core as mx  # type: ignore
except ImportError:
    mx = None  # MLX is optional; vectorized data conversion will error if invoked without MLX installed
import numpy as np


class VectorizedEnvironment:
    """
    MLX-compatible vectorized environment wrapper.
    
    This wrapper provides parallel environment execution using Gymnasium's
    vectorized environments, with MLX-optimized data conversion.
    """
    
    def __init__(self, vec_env):
        """
        Initialize vectorized environment.
        
        Args:
            vec_env: Gymnasium vectorized environment
        """
        vectorization_debug("Initializing VectorizedEnvironment")
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
    
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset all environments.
        
        Fixed to match Environment base class contract: returns (observation, info) tuple
        instead of dict. This enables VectorizedEnvironment to be used as drop-in
        replacement for GymAdapter in the training system.
        
        Previously returned dict caused silent failures - training system would unpack
        dict keys ("observation", "info") instead of actual values, passing strings
        to policy instead of observation arrays.
        """
        observations, infos = self.vec_env.reset()
        
        # Convert to MLX-compatible format and return as tuple per Environment interface
        batched_obs = self._to_mlx_batch(observations)
        return batched_obs, infos
    
    def step(self, actions: mx.array) -> Dict[str, Any]:
        vectorization_debug(f"VectorizedEnvironment.step: actions shape {actions.shape}")
        # Convert MLX actions to numpy for Gymnasium with zero-copy view for Apple Silicon efficiency
        # Use direct conversion - MLX arrays are naturally numpy-compatible via __array__ protocol
        actions_np = np.array(actions, copy=False)
        vectorization_debug(f"Converted actions: shape {actions_np.shape}")

        # Ensure proper shape for vectorized env based on action space
        # Check if action space is discrete or continuous
        from gymnasium.spaces import Discrete, MultiDiscrete, Box
        
        if isinstance(self.vec_env.action_space, (Discrete, MultiDiscrete)):
            # Discrete action space
            if actions_np.ndim == 0:
                # Scalar action - expand to [num_envs]
                actions_np = np.full(self.num_envs, actions_np.item())
            elif actions_np.ndim == 1:
                # 1D actions are already in the correct shape [num_envs]
                pass
            elif actions_np.ndim == 2 and actions_np.shape[1] == 1:
                # 2D array with shape [num_envs, 1] - flatten to [num_envs]
                actions_np = actions_np.flatten()
        elif isinstance(self.vec_env.action_space, Box):
            # Continuous action space
            if actions_np.ndim == 0:
                # Scalar action - this shouldn't happen for continuous spaces, but let's handle it
                actions_np = np.full((self.num_envs, self.vec_env.action_space.shape[0]), actions_np.item())
            elif actions_np.ndim == 1:
                # Reshape 1D array to [num_envs, action_dim]
                actions_np = actions_np.reshape(self.num_envs, -1)
        else:
            # Unknown action space type - try to handle gracefully
            if hasattr(self.vec_env.action_space, 'shape') and len(self.vec_env.action_space.shape) > 0:
                # Assume continuous action space
                if actions_np.ndim == 0:
                    actions_np = np.full((self.num_envs, self.vec_env.action_space.shape[0]), actions_np.item())
                elif actions_np.ndim == 1:
                    actions_np = actions_np.reshape(self.num_envs, -1)
            else:
                # Assume discrete action space
                if actions_np.ndim == 0:
                    actions_np = np.full(self.num_envs, actions_np.item())
                elif actions_np.ndim == 1:
                    pass
                elif actions_np.ndim == 2 and actions_np.shape[1] == 1:
                    actions_np = actions_np.flatten()

        vectorization_debug(f"Calling vec_env.step with actions shape: {actions_np.shape}")
        observations, rewards, terminated, truncated, infos = self.vec_env.step(actions_np)
        
        return {
            "observation": self._to_mlx_batch(observations),
            "reward": mx.array(rewards, dtype=mx.float32),
            "terminated": mx.array(terminated, dtype=mx.bool_),
            "truncated": mx.array(truncated, dtype=mx.bool_),
            "info": infos
        }
    
    def close(self):
        """Close all environments."""
        self.vec_env.close()
    
    def _to_mlx_batch(self, observations: np.ndarray) -> mx.array:
        """
        Convert numpy observations to MLX array batch.
        
        Args:
            observations: Numpy array [num_envs, obs_dim]
            
        Returns:
            MLX array with proper contiguous memory layout
        """
        # Ensure contiguous array for MLX compatibility
        if not observations.flags.c_contiguous:
            observations = np.ascontiguousarray(observations)
        
        return mx.array(observations, dtype=mx.float32)
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.vec_env.observation_space
    
    @property
    def action_space(self):
        """Get action space."""
        return self.vec_env.action_space


def make_vectorized_env(env_id: str, num_envs: int = 1, use_async: bool = True, env_kwargs: Optional[Dict] = None) -> VectorizedEnvironment:
    """
    Create a vectorized environment.
    
    Args:
        env_id: Environment ID
        num_envs: Number of environments to create
        use_async: Whether to use async vectorized env (default: True)
        env_kwargs: Optional kwargs to pass to individual environments
        
    Returns:
        VectorizedEnvironment instance
    """
    vectorization_debug(f"Creating vectorized env: {env_id}, num_envs={num_envs}, async={use_async}")
    
    # Handle edge case: num_envs must be positive
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    
    # Prepare environment creation arguments
    vec_kwargs = {"vectorization_mode": "async" if use_async else "sync"}
    if env_kwargs:
        vec_kwargs["env_kwargs"] = env_kwargs
    
    # Create the vectorized environment using Gymnasium
    try:
        vec_env = gym.make_vec(env_id, num_envs=num_envs, **vec_kwargs)
        vectorization_debug(f"Created gym vec env: {type(vec_env)}")
        return VectorizedEnvironment(vec_env)
    except Exception as e:
        vectorization_debug(f"Failed to create vec env: {e}")
        raise


class VectorizedCollector:
    """
    Efficient data collection using vectorized environments.
    
    This collector can gather experience from multiple environments in parallel,
    significantly speeding up training for sample-hungry algorithms like PPO.
    """
    
    def __init__(self, vec_env: VectorizedEnvironment, policy):
        """
        Initialize vectorized collector.
        
        Args:
            vec_env: Vectorized environment
            policy: Policy to collect data with
        """
        self.vec_env = vec_env
        self.policy = policy
        self.num_envs = vec_env.num_envs
    
    def collect_batch(self, batch_size: int) -> Dict[str, mx.array]:
        """
        Collect a batch of experiences using vectorized environments.
        
        Args:
            batch_size: Total number of steps to collect
            
        Returns:
            Dictionary containing batched experiences
        """
        # Calculate steps per environment
        steps_per_env = batch_size // self.num_envs
        if batch_size % self.num_envs != 0:
            steps_per_env += 1
        
        # Storage for collected data
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        logprobs = []
        
        # Reset environments (now returns tuple per Environment interface)
        current_obs, reset_info = self.vec_env.reset()
        
        for step in range(steps_per_env):
            # Get actions from policy
            action_mx, info = self.policy(current_obs, deterministic=False)
            
            # Store current step data
            observations.append(current_obs)
            actions.append(action_mx)
            values.append(info.get("value", mx.zeros(self.num_envs)))
            logprobs.append(info.get("logprob", mx.zeros(self.num_envs)))
            
            # Step environments
            step_result = self.vec_env.step(action_mx)
            
            rewards.append(step_result["reward"])
            dones.append(step_result["terminated"] | step_result["truncated"])
            
            # Update observations
            current_obs = step_result["observation"]
            
            # Handle episode resets (vectorized environments handle this automatically)
        
        # Stack all collected data
        return {
            "observations": mx.stack(observations[:batch_size]),
            "actions": mx.stack(actions[:batch_size]),
            "rewards": mx.stack(rewards[:batch_size]),
            "dones": mx.stack(dones[:batch_size]),
            "values": mx.stack(values[:batch_size]),
            "logprobs": mx.stack(logprobs[:batch_size])
        }
