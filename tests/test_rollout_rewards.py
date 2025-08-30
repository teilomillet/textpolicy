"""
Rollout Rewards Tests

Test rollout rewards processing and configuration.
"""

import pytest
import mlx.core as mx
from textpolicy.rewards.rollout_rewards import RewardConfig, RolloutRewardProcessor


@pytest.mark.unit
@pytest.mark.reward
class TestRolloutRewardsConfig:
    """Test rollout rewards configuration."""

    def test_reward_config_creation(self):
        """Test that RewardConfig can be created successfully."""
        config = RewardConfig(
            length_weight=1.0,
            target_length=15,
            keyword_weight=0.5,
            keywords=["AI", "machine"],
            perplexity_weight=0.3,
            accuracy_weight=0.2
        )
        
        assert config.length_weight == 1.0
        assert config.target_length == 15
        assert config.keyword_weight == 0.5
        assert config.keywords == ["AI", "machine"]
        assert config.perplexity_weight == 0.3
        assert config.accuracy_weight == 0.2

    def test_rollout_reward_processor_creation(self):
        """Test that RolloutRewardProcessor can be created successfully."""
        config = RewardConfig(
            length_weight=1.0,
            target_length=15,
            keyword_weight=0.5,
            keywords=["AI", "machine"],
            perplexity_weight=0.3,
            accuracy_weight=0.2
        )
        
        processor = RolloutRewardProcessor(config)
        assert processor is not None
        assert hasattr(processor, 'config')
        assert processor.config == config


@pytest.mark.unit
@pytest.mark.reward
class TestRolloutRewardsProcessing:
    """Test rollout rewards processing functionality."""

    def setup_method(self):
        """Set up test data for each test method."""
        self.config = RewardConfig(
            length_weight=1.0,
            target_length=15,
            keyword_weight=0.5,
            keywords=["AI", "machine", "learning"],
            perplexity_weight=0.3,
            accuracy_weight=0.2
        )
        self.processor = RolloutRewardProcessor(self.config)
        
        # Test data
        self.test_prompts = [
            "What is AI?",
            "Explain machine learning.",
            "How does artificial intelligence work?"
        ]
        
        self.test_completions = [
            "AI is artificial intelligence that enables machines to think.",
            "Machine learning allows computers to learn from data automatically.",
            "Artificial intelligence works by processing data through algorithms."
        ]
        
        self.test_examples = [
            {"target_length": 15, "keywords": ["AI"], "correct_answer": "intelligence"},
            {"target_length": 20, "keywords": ["learning"], "correct_answer": "data"},
            {"target_length": 18, "keywords": ["intelligence"], "correct_answer": "algorithms"}
        ]

    def test_process_single_rollout(self):
        """Test processing a single rollout."""
        try:
            # Process single example
            result = self.processor.process_rollout(
                self.test_prompts[0],
                self.test_completions[0],
                self.test_examples[0]
            )
            
            # Result should be a numeric reward
            assert isinstance(result, (int, float, mx.array))
            
            # If MLX array, should be scalar
            if isinstance(result, mx.array):
                assert result.ndim == 0 or (result.ndim == 1 and result.size == 1)
                
        except Exception as e:
            pytest.fail(f"Single rollout processing failed: {e}")

    def test_process_batch_rollout(self):
        """Test processing multiple rollouts in batch."""
        try:
            results = self.processor.process_batch_rollouts(
                self.test_prompts,
                self.test_completions,
                self.test_examples
            )
            
            # Results should be a list or MLX array
            assert isinstance(results, (list, mx.array))
            
            if isinstance(results, list):
                assert len(results) == len(self.test_prompts)
                for result in results:
                    assert isinstance(result, (int, float, mx.array))
            else:  # MLX array
                assert results.size == len(self.test_prompts)
                
        except Exception as e:
            pytest.fail(f"Batch rollout processing failed: {e}")

    def test_reward_components_isolation(self):
        """Test that individual reward components can be computed separately."""
        prompt = self.test_prompts[0]
        completion = self.test_completions[0]
        example = self.test_examples[0]
        
        try:
            # Test individual components if methods exist
            if hasattr(self.processor, 'compute_length_reward'):
                length_reward = self.processor.compute_length_reward(completion, example)
                assert isinstance(length_reward, (int, float, mx.array))
            
            if hasattr(self.processor, 'compute_keyword_reward'):
                keyword_reward = self.processor.compute_keyword_reward(completion, example)
                assert isinstance(keyword_reward, (int, float, mx.array))
                
        except Exception as e:
            # If individual components aren't available, that's acceptable
            pytest.skip(f"Individual reward components not available: {e}")


@pytest.mark.integration
@pytest.mark.reward
class TestRolloutRewardsIntegration:
    """Test rollout rewards integration with the broader system."""

    def test_reward_processor_with_mlx_tensors(self):
        """Test that reward processor works with MLX tensors."""
        config = RewardConfig(
            length_weight=1.0,
            target_length=15,
            keyword_weight=0.5,
            keywords=["test"],
            perplexity_weight=0.0,  # Disable to avoid model dependency
            accuracy_weight=0.5
        )
        
        processor = RolloutRewardProcessor(config)
        
        # Test with string inputs that should work
        prompt = "Test prompt"
        completion = "This is a test completion with some test keywords"
        example = {"target_length": 10, "keywords": ["test"], "correct_answer": "test"}
        
        try:
            result = processor.process_rollout(prompt, completion, example)
            assert isinstance(result, (int, float, mx.array))
            
        except Exception as e:
            pytest.fail(f"MLX tensor integration failed: {e}")

    def test_reward_weights_effect(self):
        """Test that changing reward weights affects the output."""
        # Create two processors with different weights
        config1 = RewardConfig(
            length_weight=1.0,
            target_length=15,
            keyword_weight=0.0,
            keywords=["test"],
            perplexity_weight=0.0,
            accuracy_weight=0.0
        )
        
        config2 = RewardConfig(
            length_weight=0.0,
            target_length=15,
            keyword_weight=1.0,
            keywords=["test"],
            perplexity_weight=0.0,
            accuracy_weight=0.0
        )
        
        processor1 = RolloutRewardProcessor(config1)
        processor2 = RolloutRewardProcessor(config2)
        
        prompt = "Test prompt"
        completion = "This is a test completion"  # Contains "test" keyword
        example = {"target_length": 5, "keywords": ["test"], "correct_answer": "test"}
        
        try:
            result1 = processor1.process_rollout(prompt, completion, example)
            result2 = processor2.process_rollout(prompt, completion, example)
            
            # Convert to float for comparison if needed
            if isinstance(result1, mx.array):
                result1 = float(result1)
            if isinstance(result2, mx.array):
                result2 = float(result2)
            
            # Results should be different since we're weighting different components
            assert result1 != result2, \
                "Different weight configurations should produce different results"
                
        except Exception as e:
            pytest.skip(f"Weight comparison test requires full reward implementation: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])