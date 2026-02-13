# textpolicy/buffer/buffer.py
"""
Coordinates storage and sampling for RL training.

The Buffer class provides a clean interface for episode-centric replay
buffer operations, optimized for on-policy RL algorithms.
"""

from typing import Optional, Any, Dict
from .storage import BufferStorage
from .sampling import BufferSampler


class Buffer:
    """
    Episode-centric replay buffer for on-policy RL (e.g., PPO).

    Stores full episodes and converts to tensors at sample time.
    Prevents silent corruption from circular overwrite.

    The buffer enforces clean rollouts:
    - Episodes are either complete or not stored
    - Optional fields (logprob, value) must be all-or-nothing
    - No partial episodes, no fragmented trajectories

    Designed for:
    - Apple Silicon (MLX, unified memory)
    - Multiprocessing (not threading) 
    - PPO, GAE, and other on-policy algorithms

    Example:
        buffer = Buffer(max_episodes=100)

        # Collect data
        buffer.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)

        # Sample data
        batch = buffer.sample_latest_steps(2048)  # Last 2k steps
        batch = buffer.sample_episodes(10, order='desc')  # Last 10 episodes
    """
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize the buffer.

        Args:
            max_episodes: Maximum number of complete episodes to store.
                         Oldest episodes are dropped when capacity is exceeded.
        """
        self.storage = BufferStorage(max_episodes)
        self.sampler = BufferSampler(self.storage.episodes)
    
    def add(
        self,
        obs: Any,
        act: Any,
        rew: Any,
        next_obs: Any,
        done: bool,
        timeout: bool = False,
        logprob: Optional[Any] = None,
        value: Optional[Any] = None,
        entropy: Optional[Any] = None
    ):
        """
        Add a transition to the current episode.

        Completes the episode and stores it if `done` or `timeout` is True.

        Args:
            obs: Observation
            act: Action taken
            rew: Reward received
            next_obs: Next observation
            done: Boolean indicating episode termination
            timeout: Boolean indicating truncation (e.g. time limit)
            logprob: Log probability of action (optional, all-or-nothing)
            value: Estimated state value (optional, all-or-nothing)
            entropy: Action entropy (optional, all-or-nothing)

        Example:
            buffer.add(
                obs=obs,
                act=action,
                rew=reward,
                next_obs=next_obs,
                done=done,
                timeout=timeout,
                logprob=logp.item(),
                value=value.item()
            )
        """
        self.storage.add_transition(
            obs=obs, act=act, rew=rew, next_obs=next_obs,
            done=done, timeout=timeout,
            logprob=logprob, value=value, entropy=entropy
        )
    
    def sample(self) -> Dict[str, Any]:
        """
        Sample all stored episodes as a single concatenated batch.

        Returns:
            Dict of MLX arrays with all transitions, in chronological order:
            - Oldest episode → Newest episode
            - Each episode: first step → last step

        Raises:
            ValueError: If buffer is empty
        """
        return self.sampler.sample_all()
    
    def sample_latest_steps(self, n: int) -> Dict[str, Any]:
        """
        Sample the N most recent transitions across episodes.

        Returns:
            Dict of MLX arrays with the latest `n` steps,
            in **chronological order** (oldest → newest).

        Args:
            n: Number of steps to sample (must be > 0)

        Raises:
            ValueError: If buffer is empty or n <= 0
        """
        return self.sampler.sample_latest_steps(n)
    
    def sample_episodes(self, k: int, order: str = 'asc') -> Dict[str, Any]:
        """
        Sample up to k complete episodes.

        Args:
            k: Number of episodes to sample (must be > 0)
            order: 'asc' for oldest first, 'desc' for newest first

        Returns:
            Dict of MLX arrays with concatenated transitions from selected episodes.

        Raises:
            ValueError: If buffer is empty, k <= 0, or invalid order
        """
        return self.sampler.sample_episodes(k, order)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        recent_first: bool = True,
        drop_incomplete: bool = True,
    ) -> Dict[str, Any]:
        """
        Sample contiguous sequences of length `seq_len`.

        Returns tensors shaped [batch, time, ...] and avoids crossing episode boundaries.

        This method is intentionally minimal and efficient to support Apple Silicon
        memory patterns and avoids padding logic; set `drop_incomplete=True` to skip
        short episodes.
        """
        return self.sampler.sample_sequences(
            batch_size=batch_size,
            seq_len=seq_len,
            recent_first=recent_first,
            drop_incomplete=drop_incomplete,
        )
    
    def add_episode_from_dict(self, data: Dict[str, Any]):
        """
        Reconstruct and add an episode from a serialized dictionary.

        This is used to deserialize episodes sent from RolloutWorker.

        Args:
            data: Dictionary containing episode data (e.g. from `episode.to_dict()`)
                  Must include: obs, act, rew, next_obs, done, timeout
                  Optional: logprob, value, entropy
        """
        self.storage.add_episode_from_dict(data)
    
    def clear(self):
        """
        Reset the buffer: clear all stored episodes and reset current episode.
        """
        self.storage.clear()
    
    def ready(self, min_episodes: int = 1) -> bool:
        """
        Check if buffer contains at least `min_episodes` complete episodes.

        Args:
            min_episodes: Minimum number of episodes required (default: 1)

        Returns:
            True if buffer has enough episodes, False otherwise
        """
        return self.storage.ready(min_episodes)
    
    def __len__(self) -> int:
        """
        Total number of steps in the buffer.

        Returns:
            Sum of steps across all stored episodes
        """
        return len(self.storage)
    
    @property
    def episodes(self):
        """Access to underlying episodes for backwards compatibility."""
        return self.storage.episodes
    
    @property
    def current_episode(self):
        """Access to current incomplete episode for backwards compatibility."""
        return self.storage.current_episode
    
    @property
    def episode_count(self) -> int:
        """Number of complete episodes currently stored."""
        return self.storage.episode_count
    
    def print_state(self, label: str = "Buffer State"):
        """
        Print current buffer state. Useful for debugging.

        Args:
            label: Label to display at the top
        """
        info = self.storage.get_storage_info()
        stats = self.sampler.get_episode_statistics()
        
        print("=" * 50)
        print(f"{label}")
        print(f"Episodes stored  : {info['episode_count']} (max={info['max_episodes']})")
        print(f"Total steps      : {info['total_steps']}")
        print(f"Capacity usage   : {info['capacity_usage']:.1%}")
        if info['episode_lengths']:
            print(f"Episode lengths  : {info['episode_lengths']}")
            print(f"Mean length      : {stats['mean_episode_length']:.1f}")
            print(f"Mean reward      : {stats['mean_episode_reward']:.2f}")
        print("=" * 50) 