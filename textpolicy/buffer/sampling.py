# textpolicy/buffer/sampling.py
"""
Buffer sampling methods for training data retrieval.

Designed for MLX and Apple Silicon with efficient tensor conversions.
"""

from typing import List, Dict
import random
import mlx.core as mx # type: ignore
from .episode import Episode


class BufferSampler:
    """
    Handles all data sampling operations for the buffer.
    
    Provides multiple sampling strategies:
    - Full buffer sampling
    - Latest N steps sampling
    - Episode-based sampling
    
    Uses MLX tensor operations and Apple Silicon-friendly patterns.
    """
    
    def __init__(self, episodes: List[Episode]):
        """
        Initialize sampler with episode storage reference.
        
        Args:
            episodes: Reference to list of complete episodes
        """
        self.episodes = episodes
    
    def sample_all(self) -> Dict[str, mx.array]:
        """
        Sample all stored episodes as a single concatenated batch.
        
        Returns all transitions in chronological order:
        - Oldest episode → Newest episode  
        - Each episode: first step → last step
        
        Returns:
            Dict of MLX arrays with all transitions
            
        Raises:
            ValueError: If buffer is empty
        """
        if not self.episodes:
            raise ValueError("Buffer empty. No episodes to sample.")
        
        # Concatenate all episodes by collecting data directly (bypass Episode validation)
        # Episodes can have different optional fields, so collect what exists across all episodes
        all_obs = []
        all_act = []
        all_rew = []
        all_next_obs = []
        all_done = []
        all_timeout = []
        all_logprob = []
        all_value = []
        all_entropy = []
        all_is_correct = []
        
        # Only include optional fields that exist in ALL episodes for consistent sampling
        # This matches the buffer's "all-or-nothing" design philosophy per episode
        has_logprob = all(episode.logprob is not None for episode in self.episodes)
        has_value = all(episode.value is not None for episode in self.episodes)
        has_entropy = all(episode.entropy is not None for episode in self.episodes)
        has_is_correct = all(episode.is_correct is not None for episode in self.episodes)
        
        # Collect transitions from all episodes
        for episode in self.episodes:
            for i in range(len(episode)):
                all_obs.append(episode.obs[i])
                all_act.append(episode.act[i])
                all_rew.append(episode.rew[i])
                all_next_obs.append(episode.next_obs[i])
                all_done.append(episode.done[i])
                all_timeout.append(episode.timeout[i])
                
                # Only collect optional fields that exist in ALL episodes
                if has_logprob:
                    all_logprob.append(episode.logprob[i])
                if has_value:
                    all_value.append(episode.value[i])
                if has_entropy:
                    all_entropy.append(episode.entropy[i])
                if has_is_correct:
                    all_is_correct.append(episode.is_correct[i])
        
        # Create Episode directly with collected data (bypassing validation during construction)
        all_transitions = Episode()
        all_transitions.obs = all_obs
        all_transitions.act = all_act
        all_transitions.rew = all_rew
        all_transitions.next_obs = all_next_obs
        all_transitions.done = all_done
        all_transitions.timeout = all_timeout
        
        # Only set optional fields if they exist in any episode
        if has_logprob:
            all_transitions.logprob = all_logprob
        if has_value:
            all_transitions.value = all_value
        if has_entropy:
            all_transitions.entropy = all_entropy
        if has_is_correct:
            all_transitions.is_correct = all_is_correct
        
        return all_transitions.to_tensor_dict()
    
    def sample_latest_steps(self, n: int) -> Dict[str, mx.array]:
        """
        Sample the N most recent transitions across episodes.
        
        Useful for on-policy RL algorithms that use recent experience for stable training.
        
        Args:
            n: Number of steps to sample (must be > 0)
            
        Returns:
            Dict of MLX arrays with the latest n steps in chronological order
            (oldest → newest, ensuring temporal consistency)
            
        Raises:
            ValueError: If buffer is empty, n <= 0, or not enough steps
        """
        if not self.episodes:
            raise ValueError("Buffer is empty. No recent episodes to sample.")
        if n <= 0:
            raise ValueError("Number of steps (n) to sample must be greater than 0.")
        
        # Collect steps in reverse chronological order (newest first)
        steps = []
        for episode in reversed(self.episodes):
            for i in reversed(range(len(episode))):
                step_dict = {
                    "obs": episode.obs[i],
                    "act": episode.act[i],
                    "rew": episode.rew[i], 
                    "next_obs": episode.next_obs[i],
                    "done": episode.done[i],
                    "timeout": episode.timeout[i],
                }
                
                # Add optional fields if present
                if episode.logprob is not None:
                    step_dict["logprob"] = episode.logprob[i]
                if episode.value is not None:
                    step_dict["value"] = episode.value[i]
                if episode.entropy is not None:
                    step_dict["entropy"] = episode.entropy[i]
                if episode.is_correct is not None:
                    step_dict["is_correct"] = episode.is_correct[i]
                
                steps.append(step_dict)
                
                # Stop when we have enough steps
                if len(steps) >= n:
                    break
            
            if len(steps) >= n:
                break
        
        if not steps:
            raise ValueError("No steps available to sample")
        
        # Reverse to get chronological order (oldest → newest)
        chronological_steps = list(reversed(steps[:n]))
        
        # Batch convert to MLX arrays
        # Collect values first, then convert in a single pass for memory efficiency
        batch = {}
        for key in chronological_steps[0].keys():
            # Extract all values for this key at once
            values = [step[key] for step in chronological_steps]
            # Single batch conversion is more efficient than individual mx.array() calls
            batch[key] = mx.stack([mx.array(v) for v in values])
        
        return batch
    
    def sample_episodes(self, k: int, order: str = 'asc') -> Dict[str, mx.array]:
        """
        Sample up to k complete episodes.
        
        Useful for episode-based training or evaluation analysis.
        
        Args:
            k: Number of episodes to sample (must be > 0)
            order: 'asc' for oldest first, 'desc' for newest first
            
        Returns:
            Dict of MLX arrays with concatenated transitions from selected episodes
            
        Raises:
            ValueError: If buffer is empty, k <= 0, or invalid order
        """
        if not self.episodes:
            raise ValueError("Buffer is empty. No episodes to sample.")
        if k <= 0:
            raise ValueError("k must be positive.")
        if order not in ('asc', 'desc'):
            raise ValueError("order must be 'asc' or 'desc'")
        
        # Select episodes based on order
        if order == 'asc':
            selected_episodes = self.episodes[:k]  # Oldest k episodes
        else:  # 'desc'
            selected_episodes = self.episodes[-k:]  # Newest k episodes
        
        if not selected_episodes:
            raise ValueError("No episodes matched the criteria.")
        
        # Concatenate all steps from selected episodes
        all_transitions = Episode()
        for episode in selected_episodes:
            for i in range(len(episode)):
                all_transitions.append(
                    obs=episode.obs[i],
                    act=episode.act[i],
                    rew=episode.rew[i],
                    next_obs=episode.next_obs[i], 
                    done=episode.done[i],
                    timeout=episode.timeout[i],
                    logprob=episode.logprob[i] if episode.logprob is not None else None,
                    value=episode.value[i] if episode.value is not None else None,
                    entropy=episode.entropy[i] if episode.entropy is not None else None,
                    is_correct=episode.is_correct[i] if episode.is_correct is not None else None,
                )
        
        return all_transitions.to_tensor_dict()

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        recent_first: bool = True,
        drop_incomplete: bool = True,
    ) -> Dict[str, mx.array]:
        """
        Sample contiguous sequences of length `seq_len`.

        Samples without crossing episode boundaries.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Length of each sequence (T). Must be > 0.
            recent_first: Prefer sampling from most recent episodes first.
            drop_incomplete: If True, skip episodes shorter than seq_len.

        Returns:
            Dict of MLX arrays with keys: obs, act, rew, next_obs, done, timeout
            and optional keys (logprob, value, entropy) included only if present.

        Raises:
            ValueError: If buffer is empty or inputs are invalid or not enough data.
        """
        if not self.episodes:
            raise ValueError("Buffer is empty. No episodes to sample.")
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive.")

        # Choose episode order per recency preference.
        episodes_iter: List[Episode]
        episodes_iter = list(reversed(self.episodes)) if recent_first else list(self.episodes)

        # Collect one latest contiguous window per episode until batch is filled.
        sequences = []
        episodes_used: List[Episode] = []

        for ep in episodes_iter:
            n = len(ep)
            if n < seq_len:
                if drop_incomplete:
                    continue
                # Padding/masking path intentionally not implemented for simplicity/perf.
                continue

            # Sample a random start index within the episode to avoid tail-bias
            # that would overrepresent terminal steps. This preserves recency at the
            # episode level via episodes_iter and reduces bias within each episode.
            max_start = n - seq_len
            start = random.randint(0, max_start) if max_start > 0 else 0
            end = start + seq_len  # ensure fixed-length window [start, start+seq_len)
            seq = {
                'obs': ep.obs[start:end],
                'act': ep.act[start:end],
                'rew': ep.rew[start:end],
                'next_obs': ep.next_obs[start:end],
                'done': ep.done[start:end],
                'timeout': ep.timeout[start:end],
            }

            # Optional fields: only include if present in the episode
            if ep.logprob is not None:
                seq['logprob'] = ep.logprob[start:end]
            if ep.value is not None:
                seq['value'] = ep.value[start:end]
            if ep.entropy is not None:
                seq['entropy'] = ep.entropy[start:end]
            if ep.is_correct is not None:
                seq['is_correct'] = ep.is_correct[start:end]

            sequences.append(seq)
            episodes_used.append(ep)
            if len(sequences) >= batch_size:
                break

        if not sequences:
            raise ValueError("No sequences available to sample.")

        # Determine optional keys that exist across all sampled sequences for consistent batching.
        all_keys = set(sequences[0].keys())
        for s in sequences[1:]:
            all_keys &= set(s.keys())

        # Convert to MLX arrays with shape [B, T, ...]
        batch: Dict[str, mx.array] = {}
        for key in all_keys:
            # First convert each sequence to [T, ...]
            per_seq = []
            for s in sequences:
                # Efficient batch conversion: one mx.array() per time step then stack along time
                # to minimize Python overhead while keeping explicit control of dimensions.
                per_seq.append(mx.stack([mx.array(v) for v in s[key]], axis=0))  # [T, ...]
            # Stack sequences along batch dimension
            batch[key] = mx.stack(per_seq, axis=0)  # [B, T, ...]

        return batch

    def get_episode_statistics(self) -> Dict[str, float]:
        """
        Get statistical information about stored episodes.
        
        Returns:
            Dictionary with episode statistics for analysis
        """
        if not self.episodes:
            return {
                'episode_count': 0,
                'total_steps': 0,
                'mean_episode_length': 0.0,
                'min_episode_length': 0,
                'max_episode_length': 0,
                'total_reward': 0.0,
                'mean_episode_reward': 0.0
            }
        
        episode_lengths = [len(ep) for ep in self.episodes]
        episode_rewards = [sum(ep.rew) for ep in self.episodes]
        
        return {
            'episode_count': len(self.episodes),
            'total_steps': sum(episode_lengths),
            'mean_episode_length': sum(episode_lengths) / len(episode_lengths),
            'min_episode_length': min(episode_lengths),
            'max_episode_length': max(episode_lengths),
            'total_reward': sum(episode_rewards),
            'mean_episode_reward': sum(episode_rewards) / len(episode_rewards)
        } 
