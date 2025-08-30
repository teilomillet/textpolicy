# textpolicy/rewards/integrated_system.py
"""
Integrated rollout reward system combining rewards and verifiers.

This system provides a unified interface for:
1. Computing rewards at the rollout level
2. Verifying text quality
3. Filtering episodes based on quality thresholds
4. MLX-optimized batch processing
"""

from typing import Dict, List, Any, Tuple
import mlx.core as mx
from dataclasses import dataclass
import numpy as np

from .rollout_rewards import RolloutRewardProcessor, RewardConfig
from .verifiers import create_default_verifier_pipeline


@dataclass
class IntegratedRewardConfig:
    """Configuration for integrated reward and verification system."""
    # Reward configuration
    reward_config: RewardConfig
    
    # Verification configuration
    enable_verification: bool = True
    verification_threshold: float = 0.7  # Minimum verification score
    strict_filtering: bool = False  # If True, reject episodes below threshold
    
    # Quality control
    min_reward_threshold: float = 0.3
    max_reward_threshold: float = 1.0
    
    # Batch processing
    batch_size: int = 32
    enable_mlx_compilation: bool = True


class IntegratedRolloutRewardSystem:
    """
    Integrated system for rollout-level reward computation and verification.
    
    Combines reward computation with quality verification for comprehensive
    episode evaluation and filtering.
    """
    
    def __init__(self, config: IntegratedRewardConfig):
        """
        Initialize integrated reward system.
        
        Args:
            config: Configuration for the integrated system
        """
        self.config = config
        
        # Initialize reward processor
        self.reward_processor = RolloutRewardProcessor(config.reward_config)
        
        # Initialize verification pipeline
        if config.enable_verification:
            self.verifier = create_default_verifier_pipeline()
        else:
            self.verifier = None
    
    def process_episodes(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> Tuple[mx.array, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process episodes with rewards and verification.
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            (rewards, accepted_episodes, rejected_episodes)
        """
        if not episodes:
            return mx.array([]), [], []
        
        # Compute rewards
        rewards = self.reward_processor.process_episode_rewards(episodes)
        
        # Verify quality if enabled
        if self.verifier:
            verification_results = self._verify_episodes(episodes)
            accepted, rejected = self._filter_episodes(
                episodes, rewards, verification_results
            )
        else:
            # No verification - accept all episodes above reward threshold
            accepted, rejected = self._filter_by_rewards_only(episodes, rewards)
        
        return rewards, accepted, rejected
    
    def _verify_episodes(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> List[Any]:
        """Verify quality of episodes."""
        prompts = [ep.get('prompt', '') for ep in episodes]
        responses = [ep.get('response', '') for ep in episodes]
        
        return self.verifier.verify_batch(prompts, responses)
    
    def _filter_episodes(
        self,
        episodes: List[Dict[str, Any]],
        rewards: mx.array,
        verification_results: List[Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter episodes based on rewards and verification."""
        accepted = []
        rejected = []
        
        for i, (episode, reward, verification) in enumerate(
            zip(episodes, rewards, verification_results)
        ):
            # Check reward threshold
            reward_ok = (
                self.config.min_reward_threshold <= reward <= self.config.max_reward_threshold
            )
            
            # Check verification threshold
            verification_ok = verification.score >= self.config.verification_threshold
            
            # Determine acceptance
            if self.config.strict_filtering:
                # Both must pass
                is_accepted = reward_ok and verification_ok
            else:
                # Either can pass (more lenient)
                is_accepted = reward_ok or verification_ok
            
            # Add metadata
            episode_with_metadata = episode.copy()
            episode_with_metadata.update({
                'reward': float(reward),
                'verification_score': verification.score,
                'verification_result': verification.result.value,
                'verification_message': verification.message,
                'verification_details': verification.details
            })
            
            if is_accepted:
                accepted.append(episode_with_metadata)
            else:
                rejected.append(episode_with_metadata)
        
        return accepted, rejected
    
    def _filter_by_rewards_only(
        self,
        episodes: List[Dict[str, Any]],
        rewards: mx.array
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter episodes based only on rewards."""
        accepted = []
        rejected = []
        
        for episode, reward in zip(episodes, rewards):
            episode_with_metadata = episode.copy()
            episode_with_metadata['reward'] = float(reward)
            
            if self.config.min_reward_threshold <= reward <= self.config.max_reward_threshold:
                accepted.append(episode_with_metadata)
            else:
                rejected.append(episode_with_metadata)
        
        return accepted, rejected
    
    def process_buffer(
        self, 
        buffer
    ) -> Tuple[mx.array, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Process all episodes in a buffer.
        
        Args:
            buffer: Buffer instance containing episodes
            
        Returns:
            (rewards, accepted_episodes, rejected_episodes)
        """
        # Extract episodes from buffer
        episodes = self._extract_episodes_from_buffer(buffer)
        
        return self.process_episodes(episodes)
    
    def _extract_episodes_from_buffer(self, buffer) -> List[Dict[str, Any]]:
        """Extract episodes from buffer in the expected format."""
        episodes = []
        
        for episode in buffer.storage.episodes:
            # Convert episode to dict format
            episode_dict = {
                'prompt': episode.obs[0] if episode.obs else '',
                'response': episode.act[-1] if episode.act else '',
                'metadata': {
                    'length': len(episode.obs),
                    'logprobs': episode.logprob,
                    'values': episode.value,
                    'episode_id': id(episode)
                }
            }
            episodes.append(episode_dict)
        
        return episodes
    
    def get_quality_metrics(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute quality metrics for a batch of episodes.
        
        Args:
            episodes: List of processed episodes with metadata
            
        Returns:
            Dictionary of quality metrics
        """
        if not episodes:
            return {}
        
        # Extract metrics
        rewards = [ep.get('reward', 0.0) for ep in episodes]
        verification_scores = [ep.get('verification_score', 1.0) for ep in episodes]
        
        # Compute statistics
        metrics = {
            'num_episodes': len(episodes),
            'reward_stats': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'median': float(np.median(rewards))
            }
        }
        
        if self.verifier:
            metrics['verification_stats'] = {
                'mean': float(np.mean(verification_scores)),
                'std': float(np.std(verification_scores)),
                'min': float(np.min(verification_scores)),
                'max': float(np.max(verification_scores)),
                'median': float(np.median(verification_scores))
            }
            
            # Count verification results
            verification_results = [ep.get('verification_result', 'pass') for ep in episodes]
            metrics['verification_counts'] = {
                'pass': verification_results.count('pass'),
                'warning': verification_results.count('warning'),
                'fail': verification_results.count('fail')
            }
        
        return metrics
    
    def close(self):
        """Cleanup resources."""
        if self.reward_processor:
            self.reward_processor.close()


# Pure function interface for easy integration
def create_integrated_reward_system(config: IntegratedRewardConfig) -> IntegratedRolloutRewardSystem:
    """
    Factory function for creating integrated reward systems.
    
    Args:
        config: Configuration for the integrated system
        
    Returns:
        IntegratedRolloutRewardSystem instance
    """
    return IntegratedRolloutRewardSystem(config)


def process_episodes_with_quality_control(
    episodes: List[Dict[str, Any]],
    config: IntegratedRewardConfig
) -> Tuple[mx.array, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Pure function for processing episodes with quality control.
    
    Args:
        episodes: List of episode dictionaries
        config: Configuration for the integrated system
        
    Returns:
        (rewards, accepted_episodes, rejected_episodes, quality_metrics)
    """
    system = IntegratedRolloutRewardSystem(config)
    try:
        rewards, accepted, rejected = system.process_episodes(episodes)
        quality_metrics = system.get_quality_metrics(accepted + rejected)
        return rewards, accepted, rejected, quality_metrics
    finally:
        system.close()


# MLX-optimized batch processing
@mx.compile
def compute_integrated_rewards(
    base_rewards: mx.array,
    verification_scores: mx.array,
    reward_weight: float = 0.7,
    verification_weight: float = 0.3
) -> mx.array:
    """
    MLX-compiled function for combining rewards and verification scores.
    
    Args:
        base_rewards: Base reward scores [batch_size]
        verification_scores: Verification scores [batch_size]
        reward_weight: Weight for base rewards
        verification_weight: Weight for verification scores
        
    Returns:
        Combined scores [batch_size]
    """
    # Normalize weights
    total_weight = reward_weight + verification_weight
    reward_weight = reward_weight / total_weight
    verification_weight = verification_weight / total_weight
    
    # Weighted combination
    combined_scores = (
        reward_weight * base_rewards +
        verification_weight * verification_scores
    )
    
    return combined_scores

