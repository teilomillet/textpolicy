"""
GSPO Verification Tests - Comprehensive Testing of GSPO Implementation

This test module verifies that GSPO is working correctly by testing:
1. Basic functionality of GSPO components
2. Comparison with GRPO behavior
3. Sequence-level vs token-level importance sampling
4. Mathematical correctness of importance weights
5. Training dynamics and convergence
"""

import pytest
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from textpolicy.algorithms import grpo, gspo
from textpolicy.generation.mlx_generation import load_model, create_policy
from textpolicy.rollout import RolloutCoordinator
from textpolicy.buffer import Buffer
from textpolicy.training import Trainer


@pytest.mark.unit
@pytest.mark.algorithm
class TestGSPOBasicFunctionality:
    """Test basic GSPO functions work correctly."""

    def test_sequence_importance_weights(self):
        """Test sequence-level importance weights computation."""
        # Create test data
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])  # 5 tokens
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])  # 5 tokens
        sequence_lengths = [2, 3]  # Two sequences: 2 tokens + 3 tokens
        
        # Test sequence-level importance weights
        seq_weights = gspo.compute_sequence_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, clip_ratio=0.2
        )
        
        assert len(seq_weights) == len(sequence_lengths), \
            f"Expected {len(sequence_lengths)} weights, got {len(seq_weights)}"
        assert all(not mx.isnan(w) and not mx.isinf(w) for w in seq_weights), \
            "All weights should be finite"

    def test_gspo_policy_loss(self):
        """Test GSPO policy loss computation."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        sequence_lengths = [2, 3]
        advantages = mx.array([0.5, -0.3])  # Advantages for each sequence
        
        # Test GSPO policy loss
        loss = gspo.gspo_policy_loss(
            old_logprobs, new_logprobs, advantages, sequence_lengths, variant="sequence"
        )
        
        assert not mx.isnan(loss) and not mx.isinf(loss), "Loss should be finite"
        assert isinstance(loss, mx.array), "Loss should be an MLX array"

    def test_hybrid_importance_weights(self):
        """Test hybrid importance weights computation."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        sequence_lengths = [2, 3]
        
        # Test hybrid variant
        hybrid_weights = gspo.compute_hybrid_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths
        )
        
        assert len(hybrid_weights) == len(old_logprobs), \
            f"Expected {len(old_logprobs)} hybrid weights, got {len(hybrid_weights)}"
        assert all(not mx.isnan(w) and not mx.isinf(w) for w in hybrid_weights), \
            "All hybrid weights should be finite"


@pytest.mark.unit
@pytest.mark.algorithm
class TestGSPOvsGRPO:
    """Test that GSPO produces different importance weights than GRPO."""

    def test_importance_weight_differences(self):
        """Test that GSPO produces different importance weights than GRPO."""
        # Create test data with clear differences between old and new policies
        old_logprobs = mx.array([-2.0, -2.0, -1.0, -1.0])  # 4 tokens
        new_logprobs = mx.array([-1.0, -1.0, -2.0, -2.0])  # Policy changed significantly
        sequence_lengths = [2, 2]  # Two sequences of equal length
        
        # Compute GSPO sequence-level importance weights
        gspo_weights = gspo.compute_sequence_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, clip_ratio=0.2
        )
        
        # Compute GRPO token-level importance ratios for comparison
        grpo_ratios = mx.exp(new_logprobs - old_logprobs)
        grpo_ratios_clipped = mx.clip(grpo_ratios, 0.8, 1.2)
        
        # GSPO should produce sequence-level weights (2 values)
        # GRPO produces token-level ratios (4 values)
        assert len(gspo_weights) == len(sequence_lengths), \
            f"GSPO should produce {len(sequence_lengths)} sequence weights"
        assert len(grpo_ratios) == len(old_logprobs), \
            f"GRPO should produce {len(old_logprobs)} token ratios"
        
        # The approaches should be fundamentally different
        # GSPO normalizes by sequence length, GRPO doesn't
        assert len(gspo_weights) != len(grpo_ratios), \
            "GSPO and GRPO should produce different numbers of weights"


@pytest.mark.unit
@pytest.mark.algorithm
class TestGSPOClipping:
    """Test GSPO clipping behavior."""

    def test_clipping_bounds_respected(self):
        """Test that importance weights respect clipping bounds."""
        # Test extreme case to verify clipping
        old_logprobs = mx.array([-10.0, -1.0])  # Extreme difference
        new_logprobs = mx.array([-1.0, -1.0])
        sequence_lengths = [2]
        clip_ratio = 0.2
        
        # Compute sequence weights
        weights = gspo.compute_sequence_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, clip_ratio=clip_ratio
        )
        
        # Weights should be clipped between (1-clip_ratio) and (1+clip_ratio)
        lower_bound = 1.0 - clip_ratio
        upper_bound = 1.0 + clip_ratio
        
        # Use tolerance for floating-point comparisons due to MLX float32 precision
        # MLX uses float32 by default, which has precision ~1.19e-7
        tolerance = 1e-6  # Conservative tolerance for float32 precision issues
        
        for weight in weights:
            weight_val = float(weight)  # Convert MLX scalar to Python float
            assert lower_bound - tolerance <= weight_val <= upper_bound + tolerance, \
                f"Weight {weight_val} outside clipping bounds [{lower_bound}, {upper_bound}] with tolerance {tolerance}"

    def test_length_normalization_effect(self):
        """Test that GSPO properly normalizes by sequence length."""
        # Identical sequences of different lengths should have similar weights
        old_logprobs_short = mx.array([-1.0, -1.0])  # 2 tokens
        new_logprobs_short = mx.array([-0.5, -0.5])  # Better by 0.5 per token
        
        old_logprobs_long = mx.array([-1.0, -1.0, -1.0, -1.0])  # 4 tokens
        new_logprobs_long = mx.array([-0.5, -0.5, -0.5, -0.5])  # Better by 0.5 per token
        
        weight_short = gspo.compute_sequence_importance_weights(
            old_logprobs_short, new_logprobs_short, [2], clip_ratio=1.0  # No clipping
        )
        weight_long = gspo.compute_sequence_importance_weights(
            old_logprobs_long, new_logprobs_long, [4], clip_ratio=1.0  # No clipping
        )
        
        # Both should be similar due to length normalization
        # Short: exp((sum(-0.5) - sum(-1.0)) / 2) = exp((−1.0 − (−2.0)) / 2) = exp(0.5)
        # Long: exp((sum(-0.5) - sum(-1.0)) / 4) = exp((−2.0 − (−4.0)) / 4) = exp(0.5)
        short_val = float(weight_short[0])
        long_val = float(weight_long[0])
        
        # They should be approximately equal due to length normalization
        assert abs(short_val - long_val) < 0.01, \
            f"Length normalization failed: short={short_val}, long={long_val}"


@pytest.mark.integration
@pytest.mark.algorithm
@pytest.mark.slow
class TestGSPOTraining:
    """Integration tests for GSPO training."""

    def test_gspo_training_step(self):
        """Test a complete GSPO training step."""
        # This is a minimal integration test
        # Create minimal test data
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.8, -0.8, -1.2, -1.2])
        advantages = mx.array([0.5, -0.3])
        sequence_lengths = [2, 2]
        
        # Test that we can compute a complete loss
        loss = gspo.gspo_policy_loss(
            old_logprobs, new_logprobs, advantages, sequence_lengths, variant="sequence"
        )
        
        assert not mx.isnan(loss) and not mx.isinf(loss), "Training loss should be finite"
        assert float(loss) != 0.0, "Loss should be non-zero for non-trivial inputs"

    def test_gspo_metrics_computation(self):
        """Test GSPO metrics computation."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.8, -0.8, -1.2, -1.2])
        advantages = mx.array([0.5, -0.3])
        
        # Test metrics computation
        metrics_fn = gspo.create_gspo_metrics(variant="sequence")
        metrics = metrics_fn(old_logprobs, new_logprobs, advantages)
        
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert len(metrics) > 0, "Metrics should not be empty"
        
        # Check for expected metric keys
        expected_keys = ['mean_advantage', 'std_advantage']
        for key in expected_keys:
            assert key in metrics, f"Missing expected metric: {key}"
            assert isinstance(metrics[key], (int, float)), \
                f"Metric {key} should be numeric, got {type(metrics[key])}"


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v"])