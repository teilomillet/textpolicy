# textpolicy/rollout/runner.py
"""
Core rollout collection engine.
"""

from typing import Callable, Dict, Optional, Tuple
import mlx.core as mx # type: ignore
from textpolicy.buffer import Buffer
from .base import RolloutStrategy, DEFAULT_MAX_STEPS


class RolloutRunner:
    """
    Collects rollouts from a single environment using a policy.
    
    Designed for MLX and Apple Silicon:
    - Minimized Python overhead in the collection loop
    - MLX array conversions performed once per step
    - Python lists for storage to reduce overhead
    - Policy inference on GPU/ANE when available
    """
    
    def __init__(
        self,
        env,
        policy: Optional[Callable[[mx.array], Tuple[mx.array, Dict[str, mx.array]]]] = None,
        strategy: Optional[RolloutStrategy] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        agent = None  # Alternative API: pass agent instead of policy/strategy
    ) -> None:
        """
        Initialize rollout runner.
        
        Args:
            env: Environment instance (must implement gym interface)
            policy: Policy function that takes obs and returns (action, extras)
            strategy: RolloutStrategy defining algorithm-specific behavior
            max_steps: Maximum steps per rollout collection
            agent: Alternative API - Agent object containing policy and rollout_strategy
        """
        self.env = env
        
        # Support both direct policy/strategy and agent-based initialization
        if agent is not None:
            # Extract policy and strategy from agent for backward compatibility with tests
            self.policy = agent.policy
            self.strategy = agent.rollout_strategy
        else:
            # Direct policy/strategy initialization (current API)
            if policy is None or strategy is None:
                raise ValueError("Must provide either agent or both policy and strategy")
            self.policy = policy
            self.strategy = strategy
            
        self.max_steps = max_steps
        self.buffer = Buffer(max_episodes=10)
        self.step_count = 0  # Track total steps collected for test compatibility

    def _normalize_step_result(self, step_result):
        """
        Normalize Environment.step results to a tuple
        (next_obs, reward, terminated, truncated, info).

        Enforces dict-shaped step results per Environment contract.
        Raises TypeError for tuple-based results.
        """
        if not isinstance(step_result, dict):
            raise TypeError(
                "Environment.step must return a dict with keys: observation, reward, "
                "terminated, truncated, info. Tuple returns are not supported."
            )
        return (
            step_result.get("observation"),
            step_result.get("reward"),
            step_result.get("terminated"),
            step_result.get("truncated"),
            step_result.get("info", {}),
        )

    def collect(self) -> Buffer:
        """
        Run one complete rollout collection.
        
        MLX efficiency guidelines:
        - Policy runs on GPU/ANE via MLX arrays
        - Buffer stores as Python lists
        - Batch array conversions reduce memory transfers
        - Strategy handles algorithm differences
        
        Returns:
            Buffer containing collected episodes
        """
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        for _ in range(self.max_steps):
            obs_single = obs
            obs_mx = mx.array(obs_single)

            # Policy forward pass - runs on GPU/ANE
            action_mx, extra = self.strategy.select_action(self.policy, obs_mx)

            # Convert action back to Python format (once per step)
            if action_mx.ndim == 0:
                action = action_mx.item()  # Scalar action (discrete environments)
            else:
                action = action_mx.tolist()  # Vector action (continuous environments)

            # Environment step (CPU-bound). Normalize to a standard tuple.
            step_result = self.env.step(action)
            next_obs, reward, done, trunc, info = self._normalize_step_result(step_result)

            # Store transition using strategy-specific logic
            # Strategy handles filtering and algorithm-specific data
            self.strategy.store_transition(
                buffer=self.buffer,
                obs=obs_single,
                act=action,
                rew=reward,
                next_obs=next_obs,
                done=done,
                timeout=trunc,
                **extra  # Algorithm-specific data (logprob, value, etc.)
            )

            # Handle episode boundaries
            if done or trunc:
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
            else:
                obs = next_obs

        return self.buffer
    
    def collect_episode(self, deterministic=False):
        """
        Collect a single episode and return as trajectory (backward compatibility).
        
        Args:
            deterministic: Whether to use deterministic policy actions
        
        Returns:
            List of transition dictionaries for test compatibility
        """
        # Reset buffer for clean episode collection
        self.buffer.clear()
        
        # Handle both old-style (obs only) and new-style (obs, info) reset
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        trajectory = []
        
        for step in range(self.max_steps):
            # Convert observation to MLX array
            obs_mx = mx.array(obs)
            
            # Policy forward pass - pass through deterministic flag to policy
            if hasattr(self.policy, '__call__'):
                # If policy supports deterministic parameter, use it
                try:
                    action_mx, extra = self.policy(obs_mx, deterministic=deterministic)
                except TypeError:
                    # Fallback: use strategy select_action which handles deterministic behavior
                    action_mx, extra = self.strategy.select_action(self.policy, obs_mx)
            else:
                action_mx, extra = self.strategy.select_action(self.policy, obs_mx)
            
            # Convert action back to Python format
            if action_mx.ndim == 0:
                action = action_mx.item()  # Scalar action (discrete environments)
            else:
                action = action_mx.tolist()  # Vector action (continuous environments)
            
            # Environment step: normalize dict/tuple to a standard tuple
            step_result = self.env.step(action)
            next_obs, reward, done, trunc, info = self._normalize_step_result(step_result)
            
            # Build transition dictionary for test compatibility
            transition = {
                'obs': obs,
                'act': action,
                'rew': reward,
                'next_obs': next_obs,
                'done': done or trunc,  # Combine terminated/truncated for backward compatibility
                **extra  # Include logprob, value, entropy from strategy
            }
            trajectory.append(transition)
            
            # Track step count for tests
            self.step_count += 1
            
            # Handle episode boundaries
            if done or trunc:
                break
            else:
                obs = next_obs
                
        return trajectory
    
    def collect_rollout(self, num_episodes: int, collect_stats=False):
        """
        Collect multiple episodes (backward compatibility).
        
        Args:
            num_episodes: Number of episodes to collect
            collect_stats: Whether to collect statistics (for test compatibility)
            
        Returns:
            List of episode trajectories
        """
        trajectories = []
        for _ in range(num_episodes):
            trajectory = self.collect_episode()
            trajectories.append(trajectory)
        
        # Store trajectories for statistics if requested
        if collect_stats:
            self._last_trajectories = trajectories
            
        return trajectories
    
    def get_statistics(self):
        """
        Get rollout statistics (for test compatibility).
        
        Returns:
            Dictionary of rollout statistics
        """
        if not hasattr(self, '_last_trajectories'):
            return {
                'total_episodes': 0,
                'total_steps': self.step_count,
                'avg_episode_length': 0,
                'avg_episode_reward': 0
            }
        
        total_episodes = len(self._last_trajectories)
        total_steps = sum(len(traj) for traj in self._last_trajectories)
        avg_episode_length = total_steps / max(total_episodes, 1)
        
        total_reward = sum(
            sum(transition['rew'] for transition in traj) 
            for traj in self._last_trajectories
        )
        avg_episode_reward = total_reward / max(total_episodes, 1)
        
        return {
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'avg_episode_length': avg_episode_length,
            'avg_episode_reward': avg_episode_reward
        } 
