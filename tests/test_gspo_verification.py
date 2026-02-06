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


# ---------------------------------------------------------------------------
# Vectorization parity & regression tests (Issue #28)
# ---------------------------------------------------------------------------

def _reference_sequence_weights(old_logprobs, new_logprobs, sequence_lengths, clip_ratio):
    """
    Sequential reference implementation — mirrors the pre-vectorization code.

    Used exclusively as a test oracle to verify the vectorized version produces
    identical results.
    """
    import math
    weights = []
    idx = 0
    log_lower = math.log(1.0 - clip_ratio)
    log_upper = math.log(1.0 + clip_ratio)
    for seq_len in sequence_lengths:
        old_sum = sum(float(old_logprobs[idx + j]) for j in range(seq_len))
        new_sum = sum(float(new_logprobs[idx + j]) for j in range(seq_len))
        log_ratio = (new_sum - old_sum) / seq_len
        clipped_lr = max(log_lower, min(log_upper, log_ratio))
        w = max(1.0 - clip_ratio, min(1.0 + clip_ratio, math.exp(clipped_lr)))
        weights.append(w)
        idx += seq_len
    return mx.array(weights)


def _reference_expand(values, sequence_lengths):
    """Reference list-based expansion — the pre-vectorization pattern."""
    result = []
    for i, length in enumerate(sequence_lengths):
        result.extend([float(values[i])] * length)
    return mx.array(result)


@pytest.mark.unit
@pytest.mark.algorithm
class TestVectorizedSequenceWeightsParity:
    """
    H1-H5: Verify vectorized compute_sequence_importance_weights matches
    the sequential reference for a range of inputs.
    """

    def test_h1_basic_parity(self):
        """H1: Basic two-sequence case matches reference."""
        old = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]
        clip = 0.2

        actual = gspo.compute_sequence_importance_weights(old, new, seq_lens, clip)
        expected = _reference_sequence_weights(old, new, seq_lens, clip)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-6), (
            f"Mismatch: actual={actual}, expected={expected}"
        )

    def test_h2_single_episode(self):
        """H2: Degenerate case — single episode."""
        old = mx.array([-1.5, -2.0, -0.5])
        new = mx.array([-1.0, -1.0, -1.0])
        seq_lens = [3]
        clip = 0.2

        actual = gspo.compute_sequence_importance_weights(old, new, seq_lens, clip)
        expected = _reference_sequence_weights(old, new, seq_lens, clip)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-6)
        assert actual.shape == (1,), f"Expected shape (1,), got {actual.shape}"

    def test_h3_many_episodes(self):
        """H3: 64 episodes with variable lengths."""
        np.random.seed(42)
        seq_lens = [int(x) for x in np.random.randint(3, 20, size=64)]
        total = sum(seq_lens)
        old = mx.array(np.random.uniform(-3.0, -0.5, size=total).astype(np.float32))
        new = mx.array(np.random.uniform(-3.0, -0.5, size=total).astype(np.float32))
        clip = 0.2

        actual = gspo.compute_sequence_importance_weights(old, new, seq_lens, clip)
        expected = _reference_sequence_weights(old, new, seq_lens, clip)
        mx.eval(actual, expected)
        assert actual.shape == (64,), f"Expected shape (64,), got {actual.shape}"
        assert mx.allclose(actual, expected, atol=1e-5), (
            f"Max diff: {float(mx.max(mx.abs(actual - expected)))}"
        )

    def test_h4_uniform_lengths(self):
        """H4: All episodes have the same length (fast-path in _expand)."""
        old = mx.array([-1.0] * 12)
        new = mx.array([-0.9] * 12)
        seq_lens = [4, 4, 4]
        clip = 0.2

        actual = gspo.compute_sequence_importance_weights(old, new, seq_lens, clip)
        expected = _reference_sequence_weights(old, new, seq_lens, clip)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-6)

    def test_h5_extreme_logprob_difference(self):
        """H5: Extreme logprob differences — clipping must engage."""
        old = mx.array([-10.0, -10.0])
        new = mx.array([-0.1, -0.1])
        seq_lens = [2]
        clip = 0.2

        actual = gspo.compute_sequence_importance_weights(old, new, seq_lens, clip)
        mx.eval(actual)
        # Must be exactly at upper clip bound
        assert float(actual[0]) <= 1.0 + clip + 1e-6
        assert float(actual[0]) >= 1.0 + clip - 1e-6, (
            f"Expected upper bound ~{1.0 + clip}, got {float(actual[0])}"
        )


@pytest.mark.unit
@pytest.mark.algorithm
class TestExpandToTokenLevel:
    """
    H1-H3: Verify _expand_to_token_level matches reference list-based expansion.
    """

    def test_h1_variable_lengths(self):
        """H1: Variable-length expansion matches reference."""
        values = mx.array([0.5, -0.3, 1.0])
        seq_lens = [2, 3, 1]

        actual = gspo._expand_to_token_level(values, seq_lens)
        expected = _reference_expand(values, seq_lens)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-7)
        assert actual.shape == (6,)

    def test_h2_uniform_lengths(self):
        """H2: Uniform-length fast path."""
        values = mx.array([1.0, 2.0, 3.0])
        seq_lens = [4, 4, 4]

        actual = gspo._expand_to_token_level(values, seq_lens)
        expected = _reference_expand(values, seq_lens)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=1e-7)
        assert actual.shape == (12,)

    def test_h3_single_episode(self):
        """H3: Single episode expansion."""
        values = mx.array([42.0])
        seq_lens = [5]

        actual = gspo._expand_to_token_level(values, seq_lens)
        mx.eval(actual)
        assert actual.shape == (5,)
        assert mx.allclose(actual, mx.array([42.0] * 5), atol=1e-7)


@pytest.mark.unit
@pytest.mark.algorithm
class TestSegmentSums:
    """
    H1-H3: Verify _segment_sums correctness.
    """

    def test_h1_basic(self):
        """H1: Simple two-segment case."""
        values = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        seq_lens = [2, 3]

        result = gspo._segment_sums(values, seq_lens)
        mx.eval(result)
        assert mx.allclose(result, mx.array([3.0, 12.0]), atol=1e-6)

    def test_h2_single_segment(self):
        """H2: Single segment = full sum."""
        values = mx.array([1.0, 2.0, 3.0])
        result = gspo._segment_sums(values, [3])
        mx.eval(result)
        assert mx.allclose(result, mx.array([6.0]), atol=1e-6)

    def test_h3_many_segments(self):
        """H3: Many small segments."""
        values = mx.array([1.0] * 10)
        seq_lens = [1] * 10
        result = gspo._segment_sums(values, seq_lens)
        mx.eval(result)
        assert mx.allclose(result, mx.array([1.0] * 10), atol=1e-6)


@pytest.mark.unit
@pytest.mark.algorithm
class TestEdgeCases:
    """
    H1-H5: Empty sequence lists and zero-length sequences must not crash or NaN.
    """

    def test_h1_empty_segment_sums(self):
        """H1: _segment_sums with empty list returns shape-(0,)."""
        result = gspo._segment_sums(mx.array([]), [])
        mx.eval(result)
        assert result.shape == (0,)

    def test_h2_empty_expand(self):
        """H2: _expand_to_token_level with empty list returns shape-(0,)."""
        result = gspo._expand_to_token_level(mx.array([]), [])
        mx.eval(result)
        assert result.shape == (0,)

    def test_h3_empty_sequence_weights(self):
        """H3: compute_sequence_importance_weights with no sequences."""
        result = gspo.compute_sequence_importance_weights(
            mx.array([]), mx.array([]), [], 0.2
        )
        mx.eval(result)
        assert result.shape == (0,)

    def test_h4_zero_length_sequence_gives_unit_weight(self):
        """H4: A zero-length sequence gets weight 1.0, not NaN."""
        result = gspo.compute_sequence_importance_weights(
            mx.array([-1.0, -1.0]),
            mx.array([-0.9, -0.9]),
            [0, 2],
            clip_ratio=0.2
        )
        mx.eval(result)
        assert result.shape == (2,)
        assert not mx.isnan(result[0]), "Zero-length sequence weight should not be NaN"
        assert abs(float(result[0]) - 1.0) < 1e-6, (
            f"Zero-length sequence weight should be 1.0, got {float(result[0])}"
        )
        # Second weight should still be computed normally
        assert not mx.isnan(result[1])

    def test_h5_zero_length_hybrid_no_nan(self):
        """H5: Hybrid weights with a zero-length sequence produce no NaN."""
        result = gspo.compute_hybrid_importance_weights(
            mx.array([-1.0, -1.0]),
            mx.array([-0.9, -0.9]),
            [0, 2]
        )
        mx.eval(result)
        # Zero-length sequence contributes 0 tokens, so output has 2 tokens
        assert result.shape == (2,)
        assert not mx.any(mx.isnan(result)), f"Got NaN in hybrid weights: {result}"

    def test_h6_expand_skips_zero_length(self):
        """H6: _expand_to_token_level with a zero-length episode produces correct shape."""
        values = mx.array([10.0, 20.0, 30.0])
        result = gspo._expand_to_token_level(values, [0, 3, 2])
        mx.eval(result)
        assert result.shape == (5,), f"Expected (5,), got {result.shape}"
        # First episode (length 0) contributes nothing; second repeats 20.0 x3, third 30.0 x2
        expected = mx.array([20.0, 20.0, 20.0, 30.0, 30.0])
        assert mx.allclose(result, expected, atol=1e-7)

    def test_h7_all_zero_length_sequences(self):
        """H7: All sequences have length 0 — weights should all be 1.0."""
        result = gspo.compute_sequence_importance_weights(
            mx.array([]), mx.array([]), [0, 0], 0.2
        )
        mx.eval(result)
        assert result.shape == (2,)
        assert mx.allclose(result, mx.ones(2), atol=1e-6)

    def test_h8_empty_hybrid_weights(self):
        """H8: compute_hybrid_importance_weights with empty sequences."""
        result = gspo.compute_hybrid_importance_weights(
            mx.array([]), mx.array([]), []
        )
        mx.eval(result)
        assert result.shape == (0,)


@pytest.mark.unit
@pytest.mark.algorithm
class TestHybridWeightsVectorized:
    """
    H1-H2: Verify hybrid importance weights produce correct shapes and values.
    """

    def test_h1_output_shape(self):
        """H1: Hybrid weights have total_tokens shape."""
        old = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]

        result = gspo.compute_hybrid_importance_weights(old, new, seq_lens)
        mx.eval(result)
        assert result.shape == (5,), f"Expected (5,), got {result.shape}"

    def test_h2_identical_policies_give_unit_weights(self):
        """H2: When old == new, all hybrid weights should be ~1.0."""
        logprobs = mx.array([-1.0, -1.5, -2.0, -0.5])
        seq_lens = [2, 2]

        result = gspo.compute_hybrid_importance_weights(
            logprobs, logprobs, seq_lens, alpha=0.5, beta=0.5
        )
        mx.eval(result)
        assert mx.allclose(result, mx.ones(4), atol=1e-6), (
            f"Expected all ~1.0, got {result}"
        )


@pytest.mark.unit
@pytest.mark.algorithm
class TestMetricsBatching:
    """
    H1-H3: Verify compute_gspo_metrics produces correct values after
    batched mx.eval() optimization.
    """

    def test_h1_sequence_variant_keys(self):
        """H1: Sequence variant produces all expected metric keys."""
        old = mx.array([-1.0, -1.0, -1.0, -1.0])
        new = mx.array([-0.8, -0.8, -1.2, -1.2])
        advantages = mx.array([0.5, -0.3])
        seq_lens = [2, 2]

        metrics = gspo.compute_gspo_metrics(old, new, advantages, seq_lens, "sequence")

        expected_keys = {
            'mean_advantage', 'std_advantage', 'min_advantage', 'max_advantage',
            'mean_seq_weight', 'std_seq_weight', 'max_seq_weight', 'min_seq_weight',
            'seq_clip_fraction', 'mean_seq_length', 'std_seq_length',
            'min_seq_length', 'max_seq_length', 'kl_divergence'
        }
        assert expected_keys.issubset(metrics.keys()), (
            f"Missing keys: {expected_keys - metrics.keys()}"
        )
        for k, v in metrics.items():
            assert isinstance(v, float), f"Metric {k} should be float, got {type(v)}"

    def test_h2_hybrid_variant_keys(self):
        """H2: Hybrid variant produces hybrid-specific metric keys."""
        old = mx.array([-1.0, -1.0, -1.0, -1.0])
        new = mx.array([-0.8, -0.8, -1.2, -1.2])
        advantages = mx.array([0.5, -0.3])
        seq_lens = [2, 2]

        metrics = gspo.compute_gspo_metrics(old, new, advantages, seq_lens, "hybrid")
        assert 'mean_hybrid_weight' in metrics
        assert 'hybrid_clip_fraction' in metrics

    def test_h3_token_variant_keys(self):
        """H3: Token variant produces token-specific metric keys."""
        old = mx.array([-1.0, -1.0, -1.0, -1.0])
        new = mx.array([-0.8, -0.8, -1.2, -1.2])
        advantages = mx.array([0.5, -0.3])
        seq_lens = [2, 2]

        metrics = gspo.compute_gspo_metrics(old, new, advantages, seq_lens, "token")
        assert 'mean_token_ratio' in metrics
        assert 'token_clip_fraction' in metrics

    def test_h4_advantage_values_correct(self):
        """H4: Advantage statistics are numerically correct."""
        old = mx.array([-1.0, -1.0, -1.0, -1.0])
        new = mx.array([-1.0, -1.0, -1.0, -1.0])
        advantages = mx.array([1.0, 3.0])
        seq_lens = [2, 2]

        metrics = gspo.compute_gspo_metrics(old, new, advantages, seq_lens, "sequence")
        assert abs(metrics['mean_advantage'] - 2.0) < 1e-5
        assert abs(metrics['min_advantage'] - 1.0) < 1e-5
        assert abs(metrics['max_advantage'] - 3.0) < 1e-5


@pytest.mark.unit
@pytest.mark.algorithm
class TestPolicyLossVectorized:
    """
    H1-H3: Verify gspo_policy_loss works correctly with vectorized internals.
    """

    def test_h1_all_variants_produce_finite_loss(self):
        """H1: All three variants produce finite, non-zero loss."""
        old = mx.array([-1.0, -1.2, -0.8, -1.1])
        new = mx.array([-1.1, -1.0, -0.9, -1.0])
        advantages = mx.array([0.5, -0.3])
        seq_lens = [2, 2]

        for variant in ["sequence", "hybrid", "token"]:
            loss = gspo.gspo_policy_loss(old, new, advantages, seq_lens, variant=variant)
            mx.eval(loss)
            assert not mx.isnan(loss), f"{variant}: loss is NaN"
            assert not mx.isinf(loss), f"{variant}: loss is Inf"

    def test_h2_zero_advantage_gives_zero_loss(self):
        """H2: Zero advantages produce zero loss for sequence variant."""
        old = mx.array([-1.0, -1.0, -1.0, -1.0])
        new = mx.array([-0.9, -0.9, -0.9, -0.9])
        advantages = mx.array([0.0, 0.0])
        seq_lens = [2, 2]

        loss = gspo.gspo_policy_loss(old, new, advantages, seq_lens, variant="sequence")
        mx.eval(loss)
        assert abs(float(loss)) < 1e-7, f"Expected ~0 loss, got {float(loss)}"

    def test_h3_identical_policies_give_standard_ppo_loss(self):
        """H3: When old==new, importance weights are 1.0, loss = -mean(advantages)."""
        logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        advantages = mx.array([0.5, -0.3])
        seq_lens = [2, 2]

        loss = gspo.gspo_policy_loss(
            logprobs, logprobs, advantages, seq_lens, variant="sequence"
        )
        mx.eval(loss)
        expected = -float(mx.mean(advantages))
        assert abs(float(loss) - expected) < 1e-6, (
            f"Expected {expected}, got {float(loss)}"
        )


if __name__ == "__main__":
    # Allow running this file directly for debugging
    pytest.main([__file__, "-v"])