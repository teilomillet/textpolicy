# textpolicy/training/rollout_manager.py
"""
Lightweight rollout manager that integrates with existing rollout system.
"""

from typing import Callable, Any
from textpolicy.rollout import RolloutCoordinator
from textpolicy.buffer import Buffer
from .metrics import RolloutMetrics


class RolloutManager:
    """
    Simple manager that wraps the existing RolloutCoordinator 
    with metrics tracking and convenient interface.
    """
    
    def __init__(
        self,
        env_fn: Callable[[], Any],
        policy_fn: Callable[[], Any],
        algorithm: str = 'grpo',
        num_workers: int = 0,
        max_steps: int = 1000,
        max_episodes: int = 100
    ):
        """
        Initialize rollout manager.
        
        Args:
            env_fn: Function that creates environment instances
            policy_fn: Function that creates policy instances  
            algorithm: Algorithm name ('grpo', 'ppo', etc.)
            num_workers: Number of worker processes (0 = single-process)
            max_steps: Maximum steps per rollout
            max_episodes: Maximum episodes to buffer
        """
        self.coordinator = RolloutCoordinator(
            env_fn=env_fn,
            policy_fn=policy_fn,
            algorithm=algorithm,
            num_workers=num_workers,
            max_steps=max_steps,
            max_episodes=max_episodes
        )
        
        self.metrics = RolloutMetrics()
    
    def collect(self) -> Buffer:
        """
        Collect rollout data and update metrics.
        
        Returns:
            Buffer containing collected episodes
        """
        # Use existing rollout system
        buffer = self.coordinator.collect()
        
        # Update metrics
        for episode in buffer.episodes:
            episode_data = episode.to_tensor_dict()
            reward = episode_data['rew'].sum().item()
            length = len(episode_data['obs'])
            self.metrics.add_episode(reward, length)
        
        return buffer
    
    def get_metrics(self) -> dict:
        """Get rollout collection metrics."""
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Reset rollout metrics."""
        self.metrics.reset()
    
    def close(self):
        """Cleanup resources."""
        self.coordinator.close()