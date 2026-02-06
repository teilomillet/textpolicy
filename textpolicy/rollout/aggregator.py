# textpolicy/rollout/aggregator.py
"""
Multi-worker buffer aggregation and coordination.
"""

from typing import List, Optional
import multiprocessing as mp
from textpolicy.buffer import Buffer
from .worker import RolloutWorker
from .base import DEFAULT_MAX_EPISODES


class BufferAggregator:
    """
    Aggregates episodes from multiple RolloutWorkers.
    
    Coordinates data collection from multiple processes:
    - Maintains bounded buffer of complete episodes
    - Provides sampling interface for trainer
    - Handles queue management and data consumption
    - Thread-safe operation for async collection
    """

    def __init__(self, num_workers: int, max_episodes: int = DEFAULT_MAX_EPISODES):
        """
        Initialize aggregator for multi-worker coordination.
        
        Args:
            num_workers: Number of RolloutWorker processes to coordinate
            max_episodes: Maximum episodes to keep (oldest are dropped)
        """
        self.num_workers = num_workers
        self._max_episodes = max_episodes
        self.buffer = Buffer(max_episodes=max_episodes)
        self._worker_queues: List[Optional[mp.Queue]] = [None] * num_workers

    def add_worker(self, worker: RolloutWorker, worker_id: int):
        """
        Register a worker's send queue for data consumption.

        Args:
            worker: RolloutWorker instance
            worker_id: Unique worker ID (0 <= id < num_workers)
            
        Raises:
            ValueError: If worker_id is out of bounds
        """
        if not (0 <= worker_id < self.num_workers):
            raise ValueError(f"worker_id must be in [0, {self.num_workers - 1}]")
        self._worker_queues[worker_id] = worker.send_queue

    def consume_from_worker(self, worker_id: int) -> bool:
        """
        Try to consume new episodes from specific worker.

        Args:
            worker_id: ID of the worker to consume from

        Returns:
            True if data was consumed, False if no data available
            
        Raises:
            ValueError: If worker_id has no registered queue
        """
        queue = self._worker_queues[worker_id]
        if queue is None:
            raise ValueError(f"No queue registered for worker {worker_id}")

        if queue.empty():
            return False

        # Get serialized episodes from worker
        episodes_data = queue.get()
        
        # Add episodes to aggregated buffer
        for ep_dict in episodes_data:
            self.buffer.add_episode_from_dict(ep_dict)
            
        return True

    def consume_all(self) -> int:
        """
        Consume available data from all workers.
        
        Non-blocking operation that checks all worker queues.

        Returns:
            Number of workers that had data ready
        """
        count = 0
        for wid in range(self.num_workers):
            queue = self._worker_queues[wid]
            if queue is not None and not queue.empty():
                if self.consume_from_worker(wid):
                    count += 1
        return count

    def ready(self, min_episodes: int = 1) -> bool:
        """
        Check if enough episodes have been collected for training.

        Args:
            min_episodes: Minimum number of episodes required

        Returns:
            True if buffer has at least min_episodes
        """
        return self.buffer.ready(min_episodes)

    def sample_latest_steps(self, n: int) -> dict:
        """
        Sample the N most recent steps for training.

        Args:
            n: Number of steps to sample

        Returns:
            Dict of MLX arrays (obs, act, rew, done, etc.)
        """
        return self.buffer.sample_latest_steps(n)

    def sample_episodes(self, k: int, order: str = 'desc') -> dict:
        """
        Sample up to k episodes for training.

        Args:
            k: Number of episodes to sample
            order: 'asc' (oldest first) or 'desc' (newest first)

        Returns:
            Dict of MLX arrays containing episode data
        """
        return self.buffer.sample_episodes(k, order)

    def clear(self):
        """Clear all collected episodes from the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Total number of steps across all episodes."""
        return len(self.buffer)

    @property
    def episode_count(self) -> int:
        """Number of complete episodes currently stored."""
        return len(self.buffer.episodes) 