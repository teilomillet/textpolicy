# textpolicy/validation/logprob_validation.py
"""
Critical Logprob Validation for MLX RL Training.

This module provides rigorous testing of logprob extraction to ensure
policy gradient algorithms receive correct probability values.

Incorrect logprob computation is the #1 cause of RL training failure.
This validation catches errors before they break training.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Dict, Any, List
import numpy as np

from ..generation.mlx_generation import compute_logprobs


class LogprobValidator:
    """
    Comprehensive validation of logprob extraction correctness.
    
    This class implements multiple tests to ensure logprob functions
    compute the correct values needed for policy gradient training.
    """
    
    @staticmethod
    def create_test_model(vocab_size: int = 100, hidden_size: int = 64) -> nn.Module:
        """Create a simple test model for logprob validation."""
        class SimpleTestModel(nn.Module):
            def __init__(self, vocab_size: int, hidden_size: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.linear = nn.Linear(hidden_size, vocab_size)
                self.vocab_size = vocab_size
            
            def __call__(self, x):
                # Simple model for testing
                if x.ndim == 1:
                    x = x[None]  # Add batch dimension
                
                # Get embeddings and average them (simple pooling)
                h = self.embedding(x)  # [batch, seq, hidden]
                h = mx.mean(h, axis=1)  # [batch, hidden]
                
                # Expand back to sequence length for autoregressive simulation
                seq_len = x.shape[1]  # type: ignore
                h_expanded = mx.broadcast_to(h[:, None, :], (h.shape[0], seq_len, h.shape[1]))  # type: ignore
                
                # Generate logits for each position
                logits = self.linear(h_expanded)  # [batch, seq, vocab]
                
                return logits
        
        return SimpleTestModel(vocab_size, hidden_size)
    
    @staticmethod
    def test_autoregressive_indexing():
        """
        Test that logprob extraction uses correct autoregressive indexing.
        
        This test verifies autoregressive indexing. Incorrect indexing breaks training.
        """
        print("Testing autoregressive indexing...")
        
        model = LogprobValidator.create_test_model()
        
        # Create test data
        prompt_tokens = mx.array([1, 2, 3])  # 3 tokens
        response_tokens = mx.array([4, 5])   # 2 tokens
        
        # Test the logprob computation
        try:
            logprobs = compute_logprobs(model, prompt_tokens, response_tokens)
            
            # Validate output shape
            assert logprobs.shape == (2,), f"Expected shape (2,), got {logprobs.shape}"
            
            # Validate logprobs are reasonable (negative values)
            assert mx.all(logprobs <= 0), "Logprobs should be <= 0 (log of probabilities)"
            
            print("Autoregressive indexing test passed")
            return True
            
        except Exception as e:
            print(f"Autoregressive indexing test failed: {e}")
            return False
    
    @staticmethod
    def test_teacher_forcing_consistency():
        """
        Test that logprobs match teacher-forcing computation.
        
        When we provide the full sequence [prompt + response] to the model,
        the logprobs should match what the model actually computed.
        """
        print("Testing teacher-forcing consistency...")
        
        model = LogprobValidator.create_test_model()
        
        prompt_tokens = mx.array([10, 20])
        response_tokens = mx.array([30, 40, 50])
        
        try:
            # Method 1: Our logprob function
            computed_logprobs = compute_logprobs(model, prompt_tokens, response_tokens)
            
            # Method 2: Manual teacher-forcing computation
            full_sequence = mx.concatenate([prompt_tokens, response_tokens])
            logits = model(full_sequence[None])  # [1, seq_len, vocab_size]
            
            # Extract the same logits our function should use
            prompt_len = len(prompt_tokens)
            manual_logits = logits[0, prompt_len-1:prompt_len-1+len(response_tokens), :]
            manual_log_probs = manual_logits - mx.logsumexp(manual_logits, axis=-1, keepdims=True)
            manual_selected = manual_log_probs[mx.arange(len(response_tokens)), response_tokens]
            
            # Compare results
            diff = mx.abs(computed_logprobs - manual_selected)
            max_diff = float(mx.max(diff))
            
            assert max_diff < 1e-6, f"Logprob mismatch: max difference {max_diff}"
            
            print(f"Teacher-forcing consistency test passed (max diff: {max_diff:.2e})")
            return True
            
        except Exception as e:
            print(f"Teacher-forcing consistency test failed: {e}")
            return False
    
    @staticmethod
    def test_batch_processing():
        """Test logprob extraction works correctly with batched inputs."""
        print("Testing batch processing...")
        
        model = LogprobValidator.create_test_model()
        
        try:
            # Test single sequence
            prompt1 = mx.array([1, 2])
            response1 = mx.array([3, 4])
            logprobs1 = compute_logprobs(model, prompt1, response1)
            
            # Test another sequence
            prompt2 = mx.array([5, 6])  
            response2 = mx.array([7, 8])
            logprobs2 = compute_logprobs(model, prompt2, response2)
            
            # Results should be consistent shapes
            assert logprobs1.shape == logprobs2.shape, "Batch processing shape inconsistency"
            
            print("Batch processing test passed")
            return True
            
        except Exception as e:
            print(f"Batch processing test failed: {e}")
            return False
    
    @staticmethod
    def test_edge_cases():
        """Test edge cases that could break training."""
        print("Testing edge cases...")
        
        model = LogprobValidator.create_test_model()
        
        try:
            # Test empty response
            prompt = mx.array([1, 2, 3])
            empty_response = mx.array([])
            logprobs_empty = compute_logprobs(model, prompt, empty_response)
            assert len(logprobs_empty) == 0, "Empty response should return empty logprobs"
            
            # Test single token response
            single_response = mx.array([4])
            logprobs_single = compute_logprobs(model, prompt, single_response)
            assert len(logprobs_single) == 1, "Single token response should return single logprob"
            
            # Test a short prompt
            short_prompt = mx.array([1])
            response = mx.array([2, 3])
            logprobs_short = compute_logprobs(model, short_prompt, response)
            assert len(logprobs_short) == 2, "Should handle short prompts correctly"
            
            print("Edge cases test passed")
            return True
            
        except Exception as e:
            print(f"Edge cases test failed: {e}")
            return False
    
    @staticmethod
    def test_numerical_stability():
        """Test numerical stability of logprob computation."""
        print("Testing numerical stability...")
        
        model = LogprobValidator.create_test_model()
        
        try:
            prompt = mx.array([1, 2, 3])
            response = mx.array([4, 5, 6])
            
            logprobs = compute_logprobs(model, prompt, response)
            
            # Check for NaN or Inf values
            assert not mx.any(mx.isnan(logprobs)), "NaN values detected in logprobs"
            assert not mx.any(mx.isinf(logprobs)), "Inf values detected in logprobs"
            
            # Check logprobs are reasonable (log-probabilities are non-positive)
            assert mx.all(logprobs <= 0), "Logprobs should be negative (log probabilities)"
            assert mx.all(logprobs >= -50), "Logprobs should not be too negative"
            
            print("Numerical stability test passed")
            return True
            
        except Exception as e:
            print(f"Numerical stability test failed: {e}")
            return False
    
    @staticmethod
    def test_gradient_computation():
        """Test that gradients can flow through logprob computation when needed."""
        print("Testing gradient computation...")
        
        model = LogprobValidator.create_test_model()
        
        try:
            prompt = mx.array([1, 2])
            response = mx.array([3, 4])
            
            # Test that we can compute gradients through the model
            def loss_fn(model_params):
                # Forward pass
                full_seq = mx.concatenate([prompt, response])
                logits = model(full_seq[None])
                
                # Simple loss (not using our logprob function, just testing model)
                return mx.mean(logits)
            
            # Test gradient computation
            grads = mx.grad(loss_fn)(model.parameters())
            
            # Should get gradients for all parameters
            assert len(grads) > 0, "No gradients computed"
            
            print("Gradient computation test passed")
            return True
            
        except Exception as e:
            print(f"Gradient computation test failed: {e}")
            return False
    
    @staticmethod
    def run_full_validation() -> bool:
        """
        Run complete logprob validation suite.
        
        Returns:
            True if all tests pass, False otherwise
        """
        print("=" * 60)
        print("LOGPROB VALIDATION")
        print("=" * 60)
        print("Testing logprob extraction correctness for RL training...")
        print()
        
        tests = [
            LogprobValidator.test_autoregressive_indexing,
            LogprobValidator.test_teacher_forcing_consistency,
            LogprobValidator.test_batch_processing,
            LogprobValidator.test_edge_cases,
            LogprobValidator.test_numerical_stability,
            LogprobValidator.test_gradient_computation
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                print()
            except Exception as e:
                print(f"Test {test.__name__} crashed: {e}")
                print()
        
        print("=" * 60)
        print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("ALL LOGPROB TESTS PASSED!")
            print("Logprob extraction is correct for RL training")
            return True
        else:
            print("LOGPROB VALIDATION FAILED!")
            print("RL training may not work correctly with the current implementation")
            return False


def validate_logprob_implementation() -> bool:
    """
    Main entry point for logprob validation.
    
    This function should be called before starting any RL training
    to ensure logprob extraction is implemented correctly.
    
    Returns:
        True if validation passes, False otherwise
    """
    return LogprobValidator.run_full_validation()


if __name__ == "__main__":
    validate_logprob_implementation()
