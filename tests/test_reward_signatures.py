"""
Reward Function Signature Tests

Test reward function signatures and compatibility to ensure proper integration.
"""

import pytest
from textpolicy.rewards import length_reward, keyword_reward, perplexity_reward, accuracy_reward


@pytest.mark.unit
@pytest.mark.reward
class TestRewardFunctionSignatures:
    """Test reward function signatures for compatibility."""

    def test_reward_functions_import(self):
        """Test that all reward functions can be imported successfully."""
        # Test that imports work
        assert callable(length_reward), "length_reward should be callable"
        assert callable(keyword_reward), "keyword_reward should be callable"
        assert callable(perplexity_reward), "perplexity_reward should be callable"
        assert callable(accuracy_reward), "accuracy_reward should be callable"

    def test_length_reward_signature(self):
        """Test length_reward function signature."""
        test_prompt = "What is AI?"
        test_completion = "AI is artificial intelligence technology that enables machines to simulate human thinking."
        test_example = {"target_length": 15}
        
        # Test basic call
        try:
            reward = length_reward(test_prompt, test_completion, test_example)
            assert isinstance(reward, (int, float)), "length_reward should return numeric value"
        except Exception as e:
            pytest.fail(f"length_reward failed with signature (prompt, completion, example): {e}")

    def test_keyword_reward_signature(self):
        """Test keyword_reward function signature."""
        test_prompt = "What is AI?"
        test_completion = "AI is artificial intelligence technology that enables machines to simulate human thinking."
        test_example = {"keywords": ["AI", "intelligence"]}
        
        try:
            reward = keyword_reward(test_prompt, test_completion, test_example)
            assert isinstance(reward, (int, float)), "keyword_reward should return numeric value"
        except Exception as e:
            pytest.fail(f"keyword_reward failed with signature (prompt, completion, example): {e}")

    def test_perplexity_reward_signature(self):
        """Test perplexity_reward function signature."""
        test_prompt = "What is AI?"
        test_completion = "AI is artificial intelligence technology."
        test_example = {"max_perplexity": 10.0}
        
        try:
            reward = perplexity_reward(test_prompt, test_completion, test_example)
            assert isinstance(reward, (int, float)), "perplexity_reward should return numeric value"
        except Exception as e:
            pytest.fail(f"perplexity_reward failed with signature (prompt, completion, example): {e}")

    def test_accuracy_reward_signature(self):
        """Test accuracy_reward function signature."""
        test_prompt = "What is 2+2?"
        test_completion = "4"
        test_example = {"correct_answer": "4"}
        
        try:
            reward = accuracy_reward(test_prompt, test_completion, test_example)
            assert isinstance(reward, (int, float)), "accuracy_reward should return numeric value"
        except Exception as e:
            pytest.fail(f"accuracy_reward failed with signature (prompt, completion, example): {e}")

    @pytest.mark.parametrize("reward_func,example_data", [
        (length_reward, {"target_length": 15}),
        (keyword_reward, {"keywords": ["test", "example"]}),
        (perplexity_reward, {"max_perplexity": 10.0}),
        (accuracy_reward, {"correct_answer": "test answer"}),
    ])
    def test_reward_function_consistency(self, reward_func, example_data):
        """Test that all reward functions follow consistent signature patterns."""
        test_prompt = "Test prompt"
        test_completion = "Test completion response"
        
        # All reward functions should accept (prompt, completion, example) signature
        try:
            result = reward_func(test_prompt, test_completion, example_data)
            assert isinstance(result, (int, float)), \
                f"{reward_func.__name__} should return numeric value"
            assert -1.0 <= result <= 1.0, \
                f"{reward_func.__name__} should return value in [-1, 1] range, got {result}"
        except Exception as e:
            pytest.fail(f"{reward_func.__name__} failed with standard signature: {e}")


@pytest.mark.integration
@pytest.mark.reward
class TestRewardIntegration:
    """Test reward function integration with the system."""

    def test_reward_functions_with_realistic_data(self):
        """Test reward functions with realistic data."""
        prompt = "Explain what machine learning is in simple terms."
        completion = "Machine learning is a type of artificial intelligence that allows computers to learn and improve from data without being explicitly programmed for every task."
        
        # Test length reward
        length_example = {"target_length": 20}
        length_result = length_reward(prompt, completion, length_example)
        assert isinstance(length_result, (int, float))
        
        # Test keyword reward
        keyword_example = {"keywords": ["machine", "learning", "artificial", "intelligence"]}
        keyword_result = keyword_reward(prompt, completion, keyword_example)
        assert isinstance(keyword_result, (int, float))
        
        # Test perplexity reward (if model available)
        perplexity_example = {"max_perplexity": 15.0}
        try:
            perplexity_result = perplexity_reward(prompt, completion, perplexity_example)
            assert isinstance(perplexity_result, (int, float))
        except Exception:
            # Perplexity might fail if model not available, which is acceptable
            pytest.skip("Perplexity reward requires model - skipping")
        
        # Test accuracy reward
        accuracy_example = {"correct_answer": "machine learning"}
        accuracy_result = accuracy_reward(prompt, completion, accuracy_example)
        assert isinstance(accuracy_result, (int, float))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])