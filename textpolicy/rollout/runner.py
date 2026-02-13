# textpolicy/rollout/runner.py
"""
Core rollout collection engine.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import mlx.core as mx # type: ignore
from textpolicy.buffer import Buffer
from textpolicy.utils.timing import Timer
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
        agent = None,  # Alternative API: pass agent instead of policy/strategy
        profile: bool = False,
    ) -> None:
        """
        Initialize rollout runner.

        Args:
            env: Environment instance (must implement gym interface)
            policy: Policy function that takes obs and returns (action, extras)
            strategy: RolloutStrategy defining algorithm-specific behavior
            max_steps: Maximum steps per rollout collection
            agent: Alternative API - Agent object containing policy and rollout_strategy
            profile: When True, collect per-phase timing via Timer (zero overhead when False)
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
        # Ensure a single collect() call can retain all completed episodes.
        self._buffer_capacity = max(10, max_steps)
        self.buffer = Buffer(max_episodes=self._buffer_capacity)
        self.step_count = 0  # Track total steps collected for test compatibility
        self._timer: Optional[Timer] = Timer() if profile else None
        self._generation_profiles: List[Dict[str, float]] = []

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

        required_keys = {"observation", "reward", "terminated", "truncated", "info"}
        missing = sorted(required_keys - set(step_result.keys()))
        if missing:
            raise KeyError(
                f"Environment.step result is missing required keys: {missing}"
            )

        # Allow numpy.bool_ when numpy is available, while enforcing boolean semantics.
        bool_types = (bool,)
        try:
            import numpy as np
            bool_types = (bool, np.bool_)
        except Exception:
            pass

        terminated = step_result["terminated"]
        truncated = step_result["truncated"]
        info = step_result["info"]

        if not isinstance(terminated, bool_types) or not isinstance(truncated, bool_types):
            raise TypeError(
                "Environment.step keys 'terminated' and 'truncated' must be booleans."
            )
        if not isinstance(info, dict):
            raise TypeError("Environment.step key 'info' must be a dict.")

        return (
            step_result["observation"],
            step_result["reward"],
            bool(terminated),
            bool(truncated),
            info,
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
        timer = self._timer
        if timer is not None:
            timer.reset()
            timer.start("total")
        self._generation_profiles = []

        # Use a fresh buffer per collect() so repeated calls do not leak or mutate
        # previously returned buffers.
        self.buffer = Buffer(max_episodes=self._buffer_capacity)

        if timer is not None:
            timer.start("env_reset")
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        if timer is not None:
            timer.stop("env_reset")

        for _ in range(self.max_steps):
            obs_single = obs
            obs_mx = mx.array(obs_single)

            # Policy forward pass - runs on GPU/ANE
            if timer is not None:
                timer.start("generation")
            action_mx, extra = self.strategy.select_action(self.policy, obs_mx)
            if timer is not None:
                # Force lazy MLX arrays to evaluate so generation cost isn't
                # leaked into later phases (env_step, buffer_store).
                mx.eval(action_mx, *[v for v in extra.values() if isinstance(v, mx.array)])
                timer.stop("generation")

            # Convert action back to Python format (once per step)
            if action_mx.ndim == 0:
                action = action_mx.item()  # Scalar action (discrete environments)
            else:
                action = action_mx.tolist()  # Vector action (continuous environments)

            # Environment step (CPU-bound). Normalize to a standard tuple.
            if timer is not None:
                timer.start("env_step")
            step_result = self.env.step(action)
            next_obs, reward, done, trunc, info = self._normalize_step_result(step_result)
            if timer is not None:
                timer.stop("env_step")

            # Store transition using strategy-specific logic
            if timer is not None:
                timer.start("buffer_store")
            transition_extra = dict(extra)
            if isinstance(info, dict) and "is_correct" in info:
                transition_extra.setdefault("is_correct", info["is_correct"])
            self.strategy.store_transition(
                buffer=self.buffer,
                obs=obs_single,
                act=action,
                rew=reward,
                next_obs=next_obs,
                done=done,
                timeout=trunc,
                **transition_extra,  # Algorithm-specific data + explicit correctness
            )
            if timer is not None:
                timer.stop("buffer_store")

            # Handle episode boundaries
            if done or trunc:
                if timer is not None:
                    timer.start("env_reset")
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
                if timer is not None:
                    timer.stop("env_reset")
            else:
                obs = next_obs

        if timer is not None:
            timer.stop("total")

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
            if isinstance(info, dict) and "is_correct" in info:
                transition["is_correct"] = info["is_correct"]
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

    def collect_batched(
        self,
        batched_policy_fn: Callable[[List[mx.array]], List[Tuple[mx.array, Dict[str, Any]]]],
        batch_size: int,
    ) -> Buffer:
        """
        Collect rollouts using batched generation across episodes.

        This path is designed for single-turn text environments where each
        ``env.step`` terminates the episode. It keeps the same buffer format as
        ``collect()`` while amortizing decode overhead across multiple prompts.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        timer = self._timer
        if timer is not None:
            timer.reset()
            timer.start("total")
        self._generation_profiles = []

        self.buffer = Buffer(max_episodes=self._buffer_capacity)
        steps_remaining = self.max_steps

        while steps_remaining > 0:
            this_batch = min(batch_size, steps_remaining)

            # Reset per episode to get prompt observations.
            if timer is not None:
                timer.start("env_reset")
            obs_batch: List[Any] = []
            for _ in range(this_batch):
                reset_result = self.env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
                obs_batch.append(obs)
            if timer is not None:
                timer.stop("env_reset")

            if timer is not None:
                timer.start("generation")
            prompt_obs_list = [mx.array(obs) for obs in obs_batch]
            results = batched_policy_fn(prompt_obs_list)
            last_decode_profile = getattr(
                batched_policy_fn, "_tp_last_decode_profile", None
            )
            if isinstance(last_decode_profile, dict):
                numeric_profile: Dict[str, float] = {}
                for key, value in last_decode_profile.items():
                    if isinstance(value, (int, float)):
                        numeric_profile[key] = float(value)
                if numeric_profile:
                    self._generation_profiles.append(numeric_profile)
            if timer is not None:
                # Force lazy MLX arrays to evaluate so generation cost isn't
                # leaked into env_step or buffer_store phases.
                arrays_to_eval = []
                for tokens, extra in results:
                    arrays_to_eval.append(tokens)
                    arrays_to_eval.extend(
                        v for v in extra.values() if isinstance(v, mx.array)
                    )
                mx.eval(*arrays_to_eval)
                timer.stop("generation")

            if len(results) != this_batch:
                raise ValueError(
                    "batched_policy_fn must return one output per prompt. "
                    f"Expected {this_batch}, got {len(results)}."
                )

            # Pre-compute actions from generation results.
            actions: List[Any] = []
            extras: List[Dict[str, Any]] = []
            for i in range(this_batch):
                response_tokens, extra = results[i]
                if response_tokens.ndim == 0:
                    actions.append(response_tokens.item())
                else:
                    actions.append(response_tokens.tolist())
                extras.append(extra)

            # Environment step: reward computation + decoding.
            if timer is not None:
                timer.start("env_step")
            step_outputs: List[Tuple[Any, float, bool, bool, Dict]] = []
            for i in range(this_batch):
                step_result = self.env.step(actions[i])
                next_obs, reward, done, trunc, info = self._normalize_step_result(step_result)

                if not (done or trunc):
                    raise ValueError(
                        "collect_batched requires single-turn episodes "
                        "(env.step must terminate or truncate each episode)."
                    )
                step_outputs.append((next_obs, reward, done, trunc, info))
            if timer is not None:
                timer.stop("env_step")

            # Buffer storage.
            if timer is not None:
                timer.start("buffer_store")
            for i in range(this_batch):
                next_obs, reward, done, trunc, info = step_outputs[i]
                transition_extra = dict(extras[i])
                if isinstance(info, dict) and "is_correct" in info:
                    transition_extra.setdefault("is_correct", info["is_correct"])
                self.strategy.store_transition(
                    buffer=self.buffer,
                    obs=obs_batch[i],
                    act=actions[i],
                    rew=reward,
                    next_obs=next_obs,
                    done=done,
                    timeout=trunc,
                    **transition_extra,
                )
            if timer is not None:
                timer.stop("buffer_store")

            steps_remaining -= this_batch

        if timer is not None:
            timer.stop("total")

        return self.buffer

    # ------------------------------------------------------------------
    # Profiling API
    # ------------------------------------------------------------------

    def get_timing(self) -> Dict[str, float]:
        """Return rollout sub-phase timing breakdown, or empty dict if not profiling.

        Returns a dict mapping phase name to *total* accumulated seconds for that
        phase across all invocations since the last ``reset_timing()`` call.
        The ``"total"`` key gives the wall-clock time for the entire collect call.
        """
        if self._timer is None:
            return {}
        result: Dict[str, float] = {}
        for phase in self._timer.times:
            result[phase] = self._timer.get_stats(phase)["total"]
        return result

    def reset_timing(self) -> None:
        """Reset timing accumulators (call per step to avoid unbounded growth)."""
        if self._timer is not None:
            self._timer.reset()
        self._generation_profiles = []

    def get_generation_profile(self) -> Dict[str, float]:
        """Return averaged decode-internal generation profile for last collect."""
        if not self._generation_profiles:
            return {}
        totals: Dict[str, float] = {}
        for profile in self._generation_profiles:
            for key, value in profile.items():
                totals[key] = totals.get(key, 0.0) + float(value)
        count = float(len(self._generation_profiles))
        return {key: val / count for key, val in totals.items()}
