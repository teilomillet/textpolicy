# textpolicy/buffer/storage.py
"""
Buffer storage and capacity management.

Handles episode storage, capacity limits, and episode lifecycle management.
Designed for efficient memory usage on Apple Silicon.
"""

from typing import List, Dict, Any
from .episode import Episode


class BufferStorage:
    """
    Manages episode storage with capacity limits and lifecycle.
    
    Features:
    - FIFO episode eviction when capacity exceeded
    - Episode validation before storage
    - Efficient storage for multiprocessing scenarios
    - Memory-conscious design for Apple Silicon
    """
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize buffer storage.
        
        Args:
            max_episodes: Maximum number of complete episodes to store.
                         Oldest episodes are dropped when capacity is exceeded.
        """
        self.max_episodes = max_episodes
        self.episodes: List[Episode] = []
        self.current_episode = Episode()
    
    def add_transition(
        self,
        obs: Any,
        act: Any, 
        rew: Any,
        next_obs: Any,
        done: bool,
        timeout: bool = False,
        **kwargs  # Additional fields like logprob, value, entropy
    ):
        """
        Add a transition to the current episode.
        
        Completes and stores the episode if done or timeout is True.
        
        Args:
            obs: Observation
            act: Action taken
            rew: Reward received
            next_obs: Next observation
            done: Boolean indicating episode termination
            timeout: Boolean indicating truncation (e.g. time limit)
            **kwargs: Optional fields (logprob, value, entropy)
        """
        # Add transition to current episode
        self.current_episode.append(
            obs=obs, act=act, rew=rew, next_obs=next_obs,
            done=done, timeout=timeout, **kwargs
        )
        
        # Complete episode if terminated
        if done or timeout:
            self._complete_current_episode()
    
    def _complete_current_episode(self):
        """
        Complete the current episode and start a new one.
        
        Validates episode before storage and enforces capacity limits.
        """
        # Validate episode before storage
        if len(self.current_episode) > 0:
            self.current_episode.validate_consistency()
            
            # Debug: episode data quality
            self._debug_episode_data(self.current_episode)
            
            # Add to storage
            self.episodes.append(self.current_episode)
            
            # Enforce capacity limit (FIFO eviction)
            if len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)  # Remove oldest episode
        
        # Start new episode
        self.current_episode = Episode()
    
    def _debug_episode_data(self, episode):
        """Debug what episodes look like - only show every 10th episode."""
        if not hasattr(self, 'episode_debug_count'):
            self.episode_debug_count = 0
        
        self.episode_debug_count += 1
        
        # Only debug every 50th episode to avoid spam
        if self.episode_debug_count % 50 == 1:
            try:
                # Type checking and safe conversion
                episode_count = int(self.episode_debug_count)
                
                rewards = episode.rew if hasattr(episode, 'rew') else []
                # Convert all rewards to float before summing to handle mixed types
                try:
                    numeric_rewards = [float(r) for r in rewards]
                    total_reward = sum(numeric_rewards)
                except (TypeError, ValueError):
                    # Fallback: filter out non-numeric rewards
                    numeric_rewards = []
                    for r in rewards:
                        try:
                            numeric_rewards.append(float(r))
                        except (TypeError, ValueError):
                            continue
                    total_reward = sum(numeric_rewards) if numeric_rewards else 0.0
                episode_length = len(episode)
                
                print(f"\nEPISODE DEBUG (Episode #{episode_count}):")
                print(f"  Episode length: {episode_length}")
                print(f"  Total reward: {total_reward:.2f}")
                
                if rewards:
                    # Ensure rewards are numeric before formatting
                    try:
                        first_rewards = [float(r) for r in rewards[:10]]
                        last_reward = float(rewards[-1])
                        print(f"  Reward sequence: {first_rewards}...")
                        print(f"  Last reward: {last_reward:.2f}")
                    except (TypeError, ValueError) as reward_error:
                        print(f"  Reward formatting error: {reward_error}")
                        print(f"  Raw rewards: {rewards[:10]}...")
                
                # Check termination
                if hasattr(episode, 'done') and episode.done:
                    final_done = episode.done[-1] if episode.done else False
                    print(f"  Final done: {final_done}")
                
                if hasattr(episode, 'timeout') and episode.timeout:
                    final_timeout = episode.timeout[-1] if episode.timeout else False
                    print(f"  Final timeout: {final_timeout}")
                    
            except Exception as e:
                print(f"  Episode debug error: {e}")
                # More detailed error info
                import traceback
                print(f"  Error details: {traceback.format_exc()}")
    
    def add_episode_from_dict(self, data: Dict[str, Any]):
        """
        Reconstruct and add an episode from serialized dictionary.
        
        Used for multiprocessing: worker serializes episode to dict,
        trainer deserializes and adds to buffer.
        
        Args:
            data: Dictionary containing episode data from episode.to_dict()
                  Must include: obs, act, rew, next_obs, done, timeout
                  Optional: logprob, value, entropy
        
        Raises:
            ValueError: If episode data is invalid or inconsistent
        """
        # Create new episode from dictionary
        episode = Episode()
        
        # Set required fields
        episode.obs = data['obs']
        episode.act = data['act'] 
        episode.rew = data['rew']
        episode.next_obs = data['next_obs']
        episode.done = data['done']
        episode.timeout = data['timeout']
        
        # Set optional fields if present
        episode.logprob = data.get('logprob', None)
        episode.value = data.get('value', None) 
        episode.entropy = data.get('entropy', None)
        
        # Validate consistency before adding
        episode.validate_consistency()
        
        # Add to storage
        self.episodes.append(episode)
        
        # Enforce capacity limit
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def clear(self):
        """
        Clear all stored episodes and reset current episode.
        
        Used to reset buffer state between training runs.
        """
        self.episodes.clear()
        self.current_episode = Episode()
    
    def ready(self, min_episodes: int = 1) -> bool:
        """
        Check if buffer has enough complete episodes for training.
        
        Args:
            min_episodes: Minimum number of episodes required
            
        Returns:
            True if buffer has at least min_episodes complete episodes
        """
        return len(self.episodes) >= min_episodes
    
    def total_steps(self) -> int:
        """
        Calculate total number of steps across all episodes.
        
        Returns:
            Sum of steps in all complete episodes
        """
        return sum(len(episode) for episode in self.episodes)
    
    def get_episodes(self) -> List[Episode]:
        """
        Get read-only access to stored episodes.
        
        Returns:
            List of complete episodes (does not include current incomplete episode)
        """
        return self.episodes.copy()
    
    def __len__(self) -> int:
        """Return total number of steps in buffer."""
        return self.total_steps()
    
    @property
    def episode_count(self) -> int:
        """Number of complete episodes currently stored."""
        return len(self.episodes)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get detailed storage information for debugging.
        
        Returns:
            Dictionary with storage statistics
        """
        return {
            'episode_count': len(self.episodes),
            'max_episodes': self.max_episodes,
            'total_steps': self.total_steps(),
            'current_episode_steps': len(self.current_episode),
            'episode_lengths': [len(ep) for ep in self.episodes],
            'capacity_usage': len(self.episodes) / self.max_episodes if self.max_episodes > 0 else 0.0
        } 
