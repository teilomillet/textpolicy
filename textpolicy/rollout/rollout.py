# textpolicy/rollout/rollout.py
"""
Main rollout coordinator and public interface.
"""

from typing import List, Callable, Any
import queue
import time
from .worker import RolloutWorker
from .runner import RolloutRunner
from .aggregator import BufferAggregator
from .strategy import create_strategy
from textpolicy.buffer import Buffer


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
        max_episodes: int = 100
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
        """
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.algorithm = algorithm
        self.num_workers = num_workers
        self.max_steps = max_steps
        
        # Create strategy for algorithm
        self.strategy = create_strategy(algorithm)
        
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
        self.runner = RolloutRunner(env, policy, self.strategy, self.max_steps)
    
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

        # Return only data collected for this request.
        self.aggregator.clear()

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
        """Collect data using single process."""
        return self.runner.collect()
    
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
