# textpolicy/rewards/rollout_rewards.py
"""
Rollout-level reward processing system for efficient MLX training.

This system processes rewards at the episode/rollout level rather than
per-transition, enabling vectorized operations and batch processing
for optimal MLX performance.

Key features:
- Batch reward computation for entire episodes
- Vectorized operations using MLX
- Integration with rollout buffer system
- Support for async external reward models
- Pure function composition
"""

from typing import Dict, List, Optional, Any
import mlx.core as mx
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optional dependency
try:
    import aiohttp # type: ignore
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

# Import reward functions used in this module
# These are registered via the @reward decorator and provide the standard (prompt, completion, example) signature
from .basic import length_reward, keyword_reward, perplexity_reward, accuracy_reward


@dataclass
class RewardConfig:
    """Configuration for rollout reward processing."""
    # Basic reward weights
    length_weight: float = 0.1
    keyword_weight: float = 0.2
    perplexity_weight: float = 0.3
    accuracy_weight: float = 0.4
    
    # Target parameters
    target_length: int = 50
    keywords: Optional[List[str]] = None
    
    # External reward model
    external_rm_url: Optional[str] = None
    external_rm_timeout: float = 30.0
    
    # Batch processing
    batch_size: int = 32
    max_workers: int = 4
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class RolloutRewardProcessor:
    """
    Efficient rollout-level reward processor for MLX training.
    
    Processes entire episodes in batches, enabling vectorized operations
    and optimal memory usage on Apple Silicon.
    """
    
    def __init__(self, config: RewardConfig):
        """
        Initialize reward processor.
        
        Args:
            config: Reward processing configuration
        """
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Pre-compile reward functions for MLX
        self._compile_reward_functions()
    
    def _compile_reward_functions(self):
        """Pre-compile reward functions for MLX optimization."""
        # These will be compiled when first called
        self._compiled_functions = {}
    
    def process_rollout(
        self, 
        prompt: str, 
        completion: str, 
        example: Dict[str, Any]
    ) -> mx.array:
        """
        Process a single rollout with prompt, completion, and example context.
        
        This method provides the standard interface expected by tests and other components.
        It follows the same signature as individual reward functions for consistency.
        
        Args:
            prompt: Input prompt text
            completion: Generated completion text
            example: Example context with target parameters and metadata
            
        Returns:
            MLX array with single reward value [1]
        """
        # Convert single rollout to episode format for consistency
        episode = {
            'prompt': prompt,
            'response': completion,
            'metadata': example
        }
        
        # Process as single episode and return scalar reward
        rewards = self.process_episode_rewards([episode])
        return rewards[0] if rewards.size > 0 else mx.array(0.0)
    
    def process_batch_rollouts(
        self, 
        prompts: List[str], 
        completions: List[str], 
        examples: List[Dict[str, Any]]
    ) -> mx.array:
        """
        Process multiple rollouts in batch for efficient processing.
        
        This method provides batch processing capability expected by tests and training loops.
        It maintains the same interface as individual reward functions but processes multiple
        examples simultaneously for better performance.
        
        Args:
            prompts: List of input prompt texts
            completions: List of generated completion texts
            examples: List of example contexts with target parameters
            
        Returns:
            MLX array of rewards [num_rollouts]
        """
        if len(prompts) != len(completions) or len(completions) != len(examples):
            raise ValueError(f"Mismatched input lengths: prompts={len(prompts)}, "
                           f"completions={len(completions)}, examples={len(examples)}")
        
        # Convert to episode format for consistency with existing implementation
        episodes = []
        for prompt, completion, example in zip(prompts, completions, examples):
            episode = {
                'prompt': prompt,
                'response': completion,
                'metadata': example
            }
            episodes.append(episode)
        
        # Process batch using existing episode processing logic
        return self.process_episode_rewards(episodes)
    
    def process_episode_rewards(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> mx.array:
        """
        Process rewards for a batch of episodes.
        
        Args:
            episodes: List of episode dictionaries with 'prompt' and 'response' fields
            
        Returns:
            MLX array of rewards [num_episodes]
        """
        if not episodes:
            return mx.array([])
        
        # Extract prompts and responses
        prompts = [ep.get('prompt', '') for ep in episodes]
        responses = [ep.get('response', '') for ep in episodes]
        
        # Compute basic rewards (vectorized where possible)
        rewards = self._compute_basic_rewards(prompts, responses)
        
        # Add external reward model scores if configured
        if self.config.external_rm_url:
            external_rewards = self._get_external_rewards(episodes)
            # Blend with basic rewards
            alpha = 0.7  # Weight for external rewards
            rewards = alpha * external_rewards + (1 - alpha) * rewards
        
        return rewards
    
    def _compute_basic_rewards(
        self, 
        prompts: List[str], 
        responses: List[str]
    ) -> mx.array:
        """
        Compute basic rewards using vectorized operations.
        
        Args:
            prompts: List of input prompts
            responses: List of generated responses
            
        Returns:
            MLX array of combined rewards
        """
        # Initialize reward array
        num_episodes = len(prompts)
        rewards = mx.zeros(num_episodes)
        
        # Length rewards - pass config as example dict for unified interface
        # The @reward decorator enforces (prompt, completion, example) signature
        if self.config.length_weight > 0:
            length_rewards = mx.array([
                length_reward(p, r, {"target_length": self.config.target_length}) # type: ignore
                for p, r in zip(prompts, responses)
            ])
            rewards += self.config.length_weight * length_rewards
        
        # Keyword rewards - pass config as example dict for unified interface  
        # The @reward decorator enforces (prompt, completion, example) signature
        if self.config.keyword_weight > 0 and self.config.keywords:
            keyword_rewards = mx.array([
                keyword_reward(p, r, {"keywords": self.config.keywords}) # type: ignore
                for p, r in zip(prompts, responses)
            ])
            rewards += self.config.keyword_weight * keyword_rewards
        
        # Perplexity rewards - pass empty example dict for unified interface
        # The @reward decorator enforces (prompt, completion, example) signature
        if self.config.perplexity_weight > 0:
            perplexity_rewards = mx.array([
                perplexity_reward(p, r, {}) # type: ignore
                for p, r in zip(prompts, responses)
            ])
            rewards += self.config.perplexity_weight * perplexity_rewards
        
        # Accuracy rewards - pass empty example dict for unified interface
        # The @reward decorator enforces (prompt, completion, example) signature
        if self.config.accuracy_weight > 0:
            accuracy_rewards = mx.array([
                accuracy_reward(p, r, {}) # type: ignore
                for p, r in zip(prompts, responses)
            ])
            rewards += self.config.accuracy_weight * accuracy_rewards
        
        return rewards
    
    def _get_external_rewards(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> mx.array:
        """
        Get rewards from external reward model.
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            MLX array of external rewards
        """
        # For now, return default rewards
        # In practice, this would call external API or model
        return mx.ones(len(episodes)) * 0.5
    
    async def _async_external_rewards(
        self, 
        episodes: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Async version for external reward model calls.
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            List of reward scores
        """
        if not self.config.external_rm_url:
            return [0.5] * len(episodes)
        
        if not HAS_AIOHTTP or aiohttp is None:
            return [0.5] * len(episodes)
        
        # Type guard: aiohttp is guaranteed to be available from this point
        # Assert for type checker that aiohttp is not None after the above checks
        assert aiohttp is not None, "aiohttp should be available when HAS_AIOHTTP is True"
        
        async def get_reward(session, episode):
            payload = {
                "prompt": episode.get("prompt", ""),
                "response": episode.get("response", ""),
                "metadata": episode.get("metadata", {})
            }
            
            try:
                async with session.post(
                    self.config.external_rm_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.external_rm_timeout)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("reward", 0.5)
                    else:
                        return 0.5
            except Exception:
                return 0.5
        
        async with aiohttp.ClientSession() as session:
            tasks = [get_reward(session, ep) for ep in episodes]
            results = await asyncio.gather(*tasks)
            return list(results)
    
    def process_buffer_rewards(self, buffer) -> mx.array:
        """
        Process rewards for all episodes in a buffer.
        
        Args:
            buffer: Buffer instance containing episodes
            
        Returns:
            MLX array of episode rewards
        """
        # Extract episodes from buffer
        episodes = []
        for episode in buffer.storage.episodes:
            # Convert episode to dict format
            episode_dict = {
                'prompt': episode.obs[0] if episode.obs else '',
                'response': episode.act[-1] if episode.act else '',
                'metadata': {
                    'length': len(episode.obs),
                    'logprobs': episode.logprob,
                    'values': episode.value
                }
            }
            episodes.append(episode_dict)
        
        return self.process_episode_rewards(episodes)
    
    def close(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


# Pure function interface for integration with rollout system
def create_rollout_reward_processor(config: RewardConfig) -> RolloutRewardProcessor:
    """
    Factory function for creating reward processors.
    
    Args:
        config: Reward processing configuration
        
    Returns:
        RolloutRewardProcessor instance
    """
    return RolloutRewardProcessor(config)


def process_episode_batch_rewards(
    episodes: List[Dict[str, Any]],
    config: RewardConfig
) -> mx.array:
    """
    Pure function for processing episode rewards.
    
    Args:
        episodes: List of episode dictionaries
        config: Reward configuration
        
    Returns:
        MLX array of rewards
    """
    processor = RolloutRewardProcessor(config)
    try:
        return processor.process_episode_rewards(episodes)
    finally:
        processor.close()


# MLX-compiled reward computation for high-performance training
@mx.compile
def compute_reward_vector(
    response_lengths: mx.array,
    keyword_matches: mx.array,
    fluency_scores: mx.array,
    accuracy_scores: mx.array,
    weights: mx.array
) -> mx.array:
    """
    MLX-compiled function for vectorized reward computation.
    
    Args:
        response_lengths: Normalized length scores [batch_size]
        keyword_matches: Keyword match scores [batch_size]
        fluency_scores: Fluency scores [batch_size]
        accuracy_scores: Accuracy scores [batch_size]
        weights: Reward weights [4] (length, keyword, fluency, accuracy)
        
    Returns:
        Combined rewards [batch_size]
    """
    # Weighted combination of reward components
    rewards = (
        weights[0] * response_lengths +
        weights[1] * keyword_matches +
        weights[2] * fluency_scores +
        weights[3] * accuracy_scores
    )
    
    return rewards

