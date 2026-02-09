# textpolicy/rollout/rollout.py
"""
Main rollout coordinator and public interface.
"""

from typing import List, Callable, Any, Optional
import queue
import time
from .worker import RolloutWorker
from .runner import RolloutRunner
from .aggregator import BufferAggregator
from .strategy import create_strategy
from textpolicy.buffer import Buffer
from textpolicy.generation.mlx_generation import create_batched_policy


class RolloutCoordinator:
    """
    Coordinator for rollout collection.
    
    Provides unified interface for:
    - Single-process rollouts (for debugging/small-scale)
    - Multi-process rollouts (for production/performance)
    - Strategy management and worker coordination
    """
    
    def __init__(
        self,
        env_fn: Callable[[], Any],
        policy_fn: Callable[[], Any],
        algorithm: str,
        num_workers: int = 0,
        max_steps: int = 1000,
        max_episodes: int = 100,
        batch_size: int = 1,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        generation_params: Optional[dict] = None,
        profile: bool = False,
    ):
        """
        Initialize rollout coordinator.

        Args:
            env_fn: Function that creates environment instances
            policy_fn: Function that creates policy instances
            algorithm: Algorithm name ('ppo', 'grpo', etc.)
            num_workers: Number of worker processes (0 = single-process mode)
            max_steps: Maximum steps per rollout
            max_episodes: Maximum episodes to buffer
            batch_size: Number of episodes to generate in parallel (1 = sequential).
            model: Optional explicit model for batched policy construction.
            tokenizer: Optional explicit tokenizer for batched policy construction.
            generation_params: Optional generation params override for batched policy.
            profile: When True, collect per-phase timing in the rollout runner.
        """
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.algorithm = algorithm
        self.num_workers = num_workers
        self.max_steps = max_steps
        self.batch_size = max(1, int(batch_size))
        self._explicit_model = model
        self._explicit_tokenizer = tokenizer
        self._explicit_generation_params = generation_params
        self.profile = profile

        if self.batch_size > 1 and self.num_workers > 0:
            raise ValueError(
                "batch_size > 1 is supported only in single-process mode "
                "(num_workers must be 0)."
            )

        # Create strategy for algorithm
        self.strategy = create_strategy(algorithm)

        # Build batched policy when explicit model/tokenizer are provided.
        self._batched_policy_fn = None
        if self.batch_size > 1 and model is not None and tokenizer is not None:
            self._batched_policy_fn = create_batched_policy(
                model, tokenizer, generation_params
            )

        # Setup for multi-process or single-process mode
        if num_workers > 0:
            self._setup_multiprocess(max_episodes)
        else:
            self._setup_singleprocess()
    
    def _setup_multiprocess(self, max_episodes: int):
        """Setup multi-process rollout collection."""
        self.aggregator = BufferAggregator(self.num_workers, max_episodes)
        self.workers: List[RolloutWorker] = []
        
        # Create and start worker processes
        for i in range(self.num_workers):
            worker = RolloutWorker(
                env_fn=self.env_fn,
                policy_fn=self.policy_fn,
                strategy=self.strategy,
                max_steps=self.max_steps
            )
            self.workers.append(worker)
            self.aggregator.add_worker(worker, i)
            worker.start()
    
    def _setup_singleprocess(self):
        """Setup single-process rollout collection."""
        self.aggregator = None
        self.workers = []
        
        # Create single runner for direct use
        env = self.env_fn()
        policy = self.policy_fn()
        self.runner = RolloutRunner(
            env, policy, self.strategy, self.max_steps, profile=self.profile,
        )

        if self.batch_size > 1 and self._batched_policy_fn is None:
            # Accept user-supplied batched policy directly.
            if getattr(policy, "_tp_is_batched", False):
                self._batched_policy_fn = policy
                return

            # Prefer explicit constructor args when provided.
            model = self._explicit_model
            tokenizer = self._explicit_tokenizer
            generation_params = self._explicit_generation_params

            # Fallback: derive from create_policy metadata attributes.
            if model is None:
                model = getattr(policy, "_tp_model", None)
            if tokenizer is None:
                tokenizer = getattr(policy, "_tp_tokenizer", None)
            if generation_params is None:
                generation_params = getattr(policy, "_tp_generation_params", None)

            if model is None or tokenizer is None:
                raise ValueError(
                    "batch_size > 1 requires a policy created by "
                    "textpolicy.generation.create_policy/create_batched_policy "
                    "or explicit model/tokenizer arguments."
                )

            self._batched_policy_fn = create_batched_policy(
                model, tokenizer, generation_params
            )
    
    def collect(self) -> Buffer:
        """
        Collect rollout data.
        
        Returns:
            Buffer containing collected episodes
        """
        if self.num_workers > 0:
            return self._collect_multiprocess()
        else:
            return self._collect_singleprocess()
    
    def _collect_multiprocess(self) -> Buffer:
        """Collect data using multiple worker processes."""
        # Drop stale queue items so this call returns only fresh rollout data.
        for worker in self.workers:
            try:
                while True:
                    worker.send_queue.get_nowait()
            except queue.Empty:
                pass

        # Fresh buffer so previously returned references are not mutated.
        self.aggregator.buffer = Buffer(max_episodes=self.aggregator._max_episodes)

        # Request rollouts from all workers
        for worker in self.workers:
            worker.collect_async()

        # Wait for and consume exactly one fresh result per worker.
        pending_workers = set(range(self.num_workers))
        while pending_workers:
            progressed = False
            for wid in tuple(pending_workers):
                queue_ref = self.aggregator._worker_queues[wid]
                if queue_ref is not None and not queue_ref.empty():
                    self.aggregator.consume_from_worker(wid)
                    pending_workers.remove(wid)
                    progressed = True
            if not progressed:
                time.sleep(0.001)
        
        # Return aggregated buffer
        # Note: In practice, trainer would manage this more carefully
        return self.aggregator.buffer
    
    def _collect_singleprocess(self) -> Buffer:
        """Collect data using single process (sequential or batched)."""
        if self._batched_policy_fn is not None and self.batch_size > 1:
            return self.runner.collect_batched(
                self._batched_policy_fn, self.batch_size
            )
        return self.runner.collect()
    
    def get_rollout_timing(self) -> dict:
        """Return rollout sub-phase timing breakdown from the runner.

        Delegates to ``RolloutRunner.get_timing()``.  Returns an empty dict
        when profiling is disabled or in multi-process mode.
        """
        if hasattr(self, "runner"):
            return self.runner.get_timing()
        return {}

    def reset_rollout_timing(self) -> None:
        """Reset rollout timing accumulators."""
        if hasattr(self, "runner"):
            self.runner.reset_timing()

    def get_rollout_generation_profile(self) -> dict:
        """Return decode-internal generation profile from the runner."""
        if hasattr(self, "runner"):
            return self.runner.get_generation_profile()
        return {}

    def close(self):
        """Cleanup resources."""
        if self.workers:
            for worker in self.workers:
                worker.close()


# Public API functions for external use
def create_rollout_coordinator(
    env_fn: Callable[[], Any],
    policy_fn: Callable[[], Any], 
    algorithm: str,
    **kwargs
) -> RolloutCoordinator:
    """
    Factory function for creating rollout coordinators.
    
    Args:
        env_fn: Environment factory function
        policy_fn: Policy factory function
        algorithm: Algorithm name
        **kwargs: Additional configuration options
        
    Returns:
        RolloutCoordinator instance
    """
    return RolloutCoordinator(env_fn, policy_fn, algorithm, **kwargs)
