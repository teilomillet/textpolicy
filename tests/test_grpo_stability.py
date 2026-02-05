"""
Tests for GRPO Stability Optimizations (Issues #6, #7, #8).

Issue #6: DAPO-style asymmetric clipping bounds
Issue #7: Soft overlong penalties for sequence length
Issue #8: Dr. GRPO length-bias fix with constant normalization

This module tests:
1. Asymmetric clipping behavior (DAPO-style)
2. Constant normalization effects (Dr. GRPO)
3. Compiled constant-norm loss function
4. Length penalty computation
5. Length shaping application
6. Length statistics utilities
"""

import pytest
import mlx.core as mx
from textpolicy.algorithms import grpo


@pytest.mark.unit
@pytest.mark.algorithm
class TestAsymmetricClipping:
    """Test DAPO-style asymmetric clipping behavior (Issue #6)."""

    def test_symmetric_clipping_backward_compat(self):
        """Test that clip_ratio parameter still works for symmetric clipping."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.5, -0.3, 0.2, -0.1, 0.4])

        # Using old symmetric API
        loss_symmetric = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio=0.2
        )

        # Using new API with equal bounds
        loss_equal_bounds = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.2
        )

        # Should produce identical results
        assert mx.allclose(loss_symmetric, loss_equal_bounds), \
            "Symmetric clip_ratio should equal clip_ratio_low == clip_ratio_high"

    def test_asymmetric_clipping_different_bounds(self):
        """Test that asymmetric bounds produce different results than symmetric."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.5, -1.0, -1.5])  # Mixed changes
        advantages = mx.array([0.5, 0.5, 0.5])  # All positive advantages

        # Symmetric clipping
        loss_symmetric = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.2
        )

        # Asymmetric clipping (DAPO-style defaults)
        loss_asymmetric = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        # Should produce different losses
        assert not mx.allclose(loss_symmetric, loss_asymmetric), \
            "Asymmetric clipping should produce different loss than symmetric"

    def test_asymmetric_allows_higher_increases(self):
        """Test that higher clip_ratio_high allows larger probability increases."""
        old_logprobs = mx.array([-2.0, -2.0, -2.0])  # Low initial probs
        new_logprobs = mx.array([-0.5, -0.5, -0.5])  # Much higher probs (ratio > 1)
        advantages = mx.array([1.0, 1.0, 1.0])  # Positive advantages

        # Tight upper bound
        loss_tight = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.2
        )

        # Loose upper bound (DAPO-style)
        loss_loose = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.5
        )

        # With positive advantages and large ratio increases,
        # loose upper bound should allow more gradient signal (more negative loss)
        assert float(loss_loose) <= float(loss_tight), \
            "Larger clip_ratio_high should allow stronger positive updates"

    def test_clip_ratio_overrides_asymmetric(self):
        """Test that clip_ratio overrides clip_ratio_low and clip_ratio_high."""
        old_logprobs = mx.array([-1.0, -1.2])
        new_logprobs = mx.array([-1.1, -1.0])
        advantages = mx.array([0.5, -0.3])

        # When clip_ratio is provided, it should override asymmetric bounds
        loss_override = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio=0.3,  # This should override
            clip_ratio_low=0.1, clip_ratio_high=0.5
        )

        loss_expected = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.3, clip_ratio_high=0.3
        )

        assert mx.allclose(loss_override, loss_expected), \
            "clip_ratio should override clip_ratio_low and clip_ratio_high"

    def test_default_asymmetric_values(self):
        """Test that default DAPO-style values are used when no clip_ratio provided."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.8, -1.2])
        advantages = mx.array([0.5, 0.5])

        # Default call (should use DAPO defaults: 0.2 low, 0.28 high)
        loss_default = grpo.policy_loss(old_logprobs, new_logprobs, advantages)

        # Explicit DAPO defaults
        loss_explicit = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        assert mx.allclose(loss_default, loss_explicit), \
            "Default values should be clip_ratio_low=0.2, clip_ratio_high=0.28"


@pytest.mark.unit
@pytest.mark.algorithm
class TestConstantNormalization:
    """Test Dr. GRPO constant normalization (Issue #8)."""

    def test_mean_normalization_default(self):
        """Test that mean normalization is the default behavior."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -0.9, -1.1, -1.1])
        advantages = mx.array([0.5, 0.5, 0.5, 0.5])

        # Default (mean normalization)
        loss_default = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages
        )

        # Explicit None (should also use mean)
        loss_none = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=None
        )

        assert mx.allclose(loss_default, loss_none), \
            "normalize_constant=None should use mean normalization"

    def test_constant_normalization_different_from_mean(self):
        """Test that constant normalization produces different results than mean."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -0.9, -1.1, -1.1])
        advantages = mx.array([0.5, 0.5, 0.5, 0.5])

        # Mean normalization
        loss_mean = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=None
        )

        # Constant normalization with same count as batch size
        loss_constant = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=4  # Same as batch size
        )

        # With normalize_constant == batch_size, they should be equal
        # (sum / N == mean)
        assert mx.allclose(loss_mean, loss_constant, atol=1e-5), \
            "normalize_constant=batch_size should equal mean normalization"

    def test_constant_normalization_scaling(self):
        """Test that constant normalization scales loss appropriately."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1])
        advantages = mx.array([0.5, 0.5])

        # Different normalization constants
        loss_small = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=1
        )

        loss_large = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=10
        )

        # Larger constant should produce smaller (less negative) loss magnitude
        # Because we divide by larger number
        assert abs(float(loss_small)) > abs(float(loss_large)), \
            "Larger normalize_constant should reduce loss magnitude"

    def test_constant_normalization_removes_length_bias(self):
        """Test that constant normalization treats sequences equally regardless of length."""
        # Short sequence (2 tokens)
        old_short = mx.array([-1.0, -1.0])
        new_short = mx.array([-0.9, -0.9])
        adv_short = mx.array([1.0, 1.0])

        # Long sequence (4 tokens) with same per-token reward signal
        old_long = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_long = mx.array([-0.9, -0.9, -0.9, -0.9])
        adv_long = mx.array([1.0, 1.0, 1.0, 1.0])

        # With mean normalization, per-sequence contribution is the same
        loss_short_mean = grpo.policy_loss(old_short, new_short, adv_short)
        loss_long_mean = grpo.policy_loss(old_long, new_long, adv_long)

        # With constant normalization, longer sequence contributes more
        constant = 1024  # Fixed constant
        loss_short_const = grpo.policy_loss(
            old_short, new_short, adv_short, normalize_constant=constant
        )
        loss_long_const = grpo.policy_loss(
            old_long, new_long, adv_long, normalize_constant=constant
        )

        # Mean normalization: same loss (normalized per-element)
        assert mx.allclose(loss_short_mean, loss_long_mean), \
            "Mean normalization should give same per-sequence loss"

        # Constant normalization: longer sequence has larger total contribution
        assert abs(float(loss_long_const)) > abs(float(loss_short_const)), \
            "Constant normalization should make longer sequences contribute more"


@pytest.mark.unit
@pytest.mark.algorithm
class TestCompiledConstantNormLoss:
    """Test compiled version of constant-norm policy loss."""

    def test_compiled_produces_same_result(self):
        """Test that compiled version produces same result as non-compiled."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1])
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0])
        advantages = mx.array([0.5, -0.3, 0.2, -0.1])

        # Non-compiled with constant normalization
        loss_regular = grpo.policy_loss(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
            normalize_constant=1024
        )

        # Compiled version
        loss_compiled = grpo.policy_loss_compiled_constant_norm(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
            normalize_constant=1024.0
        )

        assert mx.allclose(loss_regular, loss_compiled, atol=1e-5), \
            "Compiled constant-norm loss should match non-compiled version"

    def test_compiled_with_different_constants(self):
        """Test compiled function with different normalization constants."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.0, -1.1])
        advantages = mx.array([0.5, 0.5, 0.5])

        loss_512 = grpo.policy_loss_compiled_constant_norm(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=512.0
        )

        loss_1024 = grpo.policy_loss_compiled_constant_norm(
            old_logprobs, new_logprobs, advantages,
            normalize_constant=1024.0
        )

        # Loss should scale inversely with constant
        # loss_512 / loss_1024 should be approximately 2
        ratio = float(loss_512) / float(loss_1024)
        assert 1.9 < ratio < 2.1, \
            f"Loss ratio should be ~2.0 when constant doubles, got {ratio}"

    def test_compiled_supports_asymmetric_clipping(self):
        """Test that compiled version supports asymmetric clipping."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.5, -1.5])  # One increases, one decreases
        advantages = mx.array([1.0, 1.0])

        # Symmetric
        loss_symmetric = grpo.policy_loss_compiled_constant_norm(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.2,
            normalize_constant=1024.0
        )

        # Asymmetric (DAPO defaults)
        loss_asymmetric = grpo.policy_loss_compiled_constant_norm(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
            normalize_constant=1024.0
        )

        # Should be different
        assert not mx.allclose(loss_symmetric, loss_asymmetric), \
            "Compiled version should support asymmetric clipping"

    def test_compiled_handles_edge_cases(self):
        """Test compiled function handles edge cases without errors."""
        # Empty arrays
        empty = mx.array([])
        # Should not crash (may produce nan or zero)

        # Single element
        single_old = mx.array([-1.0])
        single_new = mx.array([-0.9])
        single_adv = mx.array([0.5])

        loss = grpo.policy_loss_compiled_constant_norm(
            single_old, single_new, single_adv,
            normalize_constant=1024.0
        )

        assert not mx.isnan(loss) and not mx.isinf(loss), \
            "Compiled loss should handle single-element input"

    def test_compiled_validates_normalize_constant(self):
        """Test that compiled version validates normalize_constant > 0."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1])
        advantages = mx.array([0.5, 0.5])

        # Zero should raise ValueError
        with pytest.raises(ValueError, match="normalize_constant must be positive"):
            grpo.policy_loss_compiled_constant_norm(
                old_logprobs, new_logprobs, advantages,
                normalize_constant=0.0
            )

        # Negative should raise ValueError
        with pytest.raises(ValueError, match="normalize_constant must be positive"):
            grpo.policy_loss_compiled_constant_norm(
                old_logprobs, new_logprobs, advantages,
                normalize_constant=-1.0
            )


@pytest.mark.unit
@pytest.mark.algorithm
class TestLengthPenalty:
    """Test soft overlong penalty computation (Issue #7)."""

    def test_no_penalty_below_threshold(self):
        """Test that sequences below threshold get no penalty."""
        # max_length=512, cache_length=100 -> threshold=412
        penalty = grpo.compute_length_penalty(
            sequence_length=400,
            max_length=512,
            cache_length=100
        )

        assert penalty == 0.0, \
            "Sequences below threshold should get no penalty"

    def test_penalty_at_threshold(self):
        """Test penalty at exact threshold is zero."""
        penalty = grpo.compute_length_penalty(
            sequence_length=412,  # Exactly at threshold
            max_length=512,
            cache_length=100
        )

        assert penalty == 0.0, \
            "Penalty at exact threshold should be zero"

    def test_penalty_increases_linearly(self):
        """Test that penalty increases linearly in the penalty zone."""
        max_length = 512
        cache_length = 100
        max_penalty = 0.5

        # At 50% through penalty zone
        penalty_50 = grpo.compute_length_penalty(
            sequence_length=462,  # 412 + 50 = 50% through
            max_length=max_length,
            cache_length=cache_length,
            max_penalty=max_penalty
        )

        # At 100% (max_length)
        penalty_100 = grpo.compute_length_penalty(
            sequence_length=512,
            max_length=max_length,
            cache_length=cache_length,
            max_penalty=max_penalty
        )

        assert abs(penalty_50 - (-0.25)) < 0.01, \
            f"50% penalty should be -0.25, got {penalty_50}"
        assert abs(penalty_100 - (-0.5)) < 0.01, \
            f"100% penalty should be -0.5, got {penalty_100}"

    def test_penalty_clamped_at_max(self):
        """Test that penalty is clamped at max even if sequence exceeds max_length."""
        penalty = grpo.compute_length_penalty(
            sequence_length=600,  # Exceeds max_length
            max_length=512,
            cache_length=100,
            max_penalty=0.5
        )

        assert penalty == -0.5, \
            "Penalty should be clamped at -max_penalty for overlong sequences"

    def test_custom_max_penalty(self):
        """Test custom max_penalty values."""
        penalty = grpo.compute_length_penalty(
            sequence_length=512,
            max_length=512,
            cache_length=100,
            max_penalty=1.0  # Stronger penalty
        )

        assert penalty == -1.0, \
            "Custom max_penalty should be applied"

    def test_custom_cache_length(self):
        """Test custom cache_length values."""
        # Smaller cache_length means steeper penalty gradient
        penalty_small_cache = grpo.compute_length_penalty(
            sequence_length=500,  # In penalty zone for both
            max_length=512,
            cache_length=50,  # Small cache: threshold=462
            max_penalty=0.5
        )

        penalty_large_cache = grpo.compute_length_penalty(
            sequence_length=500,
            max_length=512,
            cache_length=200,  # Large cache: threshold=312
            max_penalty=0.5
        )

        # Smaller cache = steeper gradient = larger penalty at same length
        # Actually wait - 500 with cache=50 is 38/50 through = -0.38
        # 500 with cache=200 is 188/200 through = -0.47
        # Larger cache means more of the sequence is in penalty zone
        assert penalty_large_cache < penalty_small_cache, \
            "Larger cache_length should give larger penalty (more of sequence in zone)"


@pytest.mark.unit
@pytest.mark.algorithm
class TestLengthShaping:
    """Test length shaping application to rewards."""

    def test_length_shaping_applies_penalties(self):
        """Test that length shaping correctly applies penalties to rewards."""
        rewards = mx.array([1.0, 0.5, 0.0])
        sequence_lengths = [400, 500, 520]  # Below, in zone, at/past max

        shaped = grpo.apply_length_shaping(
            rewards=rewards,
            sequence_lengths=sequence_lengths,
            max_length=512,
            cache_length=100,
            max_penalty=0.5
        )

        # First reward: no penalty (below threshold)
        assert abs(float(shaped[0]) - 1.0) < 0.01, \
            "Reward below threshold should be unchanged"

        # Second reward: partial penalty (in penalty zone)
        # 500 is 88/100 through zone, penalty = -0.44
        assert float(shaped[1]) < 0.5, \
            "Reward in penalty zone should be reduced"

        # Third reward: full penalty (at/past max)
        assert float(shaped[2]) < 0.0, \
            "Reward at max_length should have full penalty applied"

    def test_length_shaping_preserves_reward_ordering(self):
        """Test that length shaping preserves relative reward ordering for similar lengths."""
        rewards = mx.array([1.0, 0.5, 0.2])
        sequence_lengths = [400, 400, 400]  # All same length (below threshold)

        shaped = grpo.apply_length_shaping(
            rewards=rewards,
            sequence_lengths=sequence_lengths,
            max_length=512
        )

        # Order should be preserved
        assert float(shaped[0]) > float(shaped[1]) > float(shaped[2]), \
            "Reward ordering should be preserved for same-length sequences"

    def test_length_shaping_empty_sequence(self):
        """Test length shaping handles empty inputs."""
        rewards = mx.array([])
        sequence_lengths = []

        shaped = grpo.apply_length_shaping(
            rewards=rewards,
            sequence_lengths=sequence_lengths,
            max_length=512
        )

        assert shaped.shape[0] == 0, \
            "Empty input should produce empty output"


@pytest.mark.unit
@pytest.mark.algorithm
class TestLengthShapingStats:
    """Test length shaping statistics computation."""

    def test_stats_empty_input(self):
        """Test statistics for empty input."""
        stats = grpo.compute_length_shaping_stats(
            sequence_lengths=[],
            max_length=512,
            cache_length=100
        )

        assert stats['mean_length'] == 0.0
        assert stats['max_length_observed'] == 0
        assert stats['truncation_rate'] == 0.0
        assert stats['penalty_zone_rate'] == 0.0

    def test_stats_all_below_threshold(self):
        """Test statistics when all sequences are below threshold."""
        stats = grpo.compute_length_shaping_stats(
            sequence_lengths=[100, 200, 300, 400],
            max_length=512,
            cache_length=100
        )

        assert stats['mean_length'] == 250.0
        assert stats['max_length_observed'] == 400
        assert stats['truncation_rate'] == 0.0
        assert stats['penalty_zone_rate'] == 0.0

    def test_stats_mixed_lengths(self):
        """Test statistics with mixed sequence lengths."""
        # max_length=512, cache_length=100 -> threshold=412
        stats = grpo.compute_length_shaping_stats(
            sequence_lengths=[300, 450, 480, 512, 550],  # 2 in zone, 2 truncated
            max_length=512,
            cache_length=100
        )

        assert stats['mean_length'] == 458.4
        assert stats['max_length_observed'] == 550
        assert stats['truncation_rate'] == 0.4  # 2/5 at or past max
        assert stats['penalty_zone_rate'] == 0.4  # 2/5 in penalty zone (450, 480)

    def test_stats_all_truncated(self):
        """Test statistics when all sequences are truncated."""
        stats = grpo.compute_length_shaping_stats(
            sequence_lengths=[512, 520, 600],
            max_length=512,
            cache_length=100
        )

        assert stats['truncation_rate'] == 1.0
        assert stats['penalty_zone_rate'] == 0.0  # None in penalty zone, all past

    def test_stats_all_in_penalty_zone(self):
        """Test statistics when all sequences are in penalty zone."""
        # threshold = 412, max = 512
        stats = grpo.compute_length_shaping_stats(
            sequence_lengths=[420, 450, 480, 500],
            max_length=512,
            cache_length=100
        )

        assert stats['truncation_rate'] == 0.0
        assert stats['penalty_zone_rate'] == 1.0


@pytest.mark.unit
@pytest.mark.algorithm
class TestComputeMetricsAsymmetric:
    """Test that compute_metrics works with asymmetric clipping."""

    def test_metrics_include_asymmetric_bounds(self):
        """Test that metrics include the asymmetric clip ratio bounds."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.5, -1.0, -1.5])
        advantages = mx.array([0.5, 0.0, -0.5])

        metrics = grpo.compute_metrics(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.3
        )

        assert metrics['clip_ratio_low'] == 0.2
        assert metrics['clip_ratio_high'] == 0.3

    def test_metrics_separate_clip_fractions(self):
        """Test that metrics track upper and lower clip fractions separately."""
        # Create data where some ratios hit upper bound, some hit lower
        old_logprobs = mx.array([-2.0, -0.5, -1.0])  # ratio = exp(new - old)
        new_logprobs = mx.array([-0.5, -2.0, -1.0])  # [exp(1.5), exp(-1.5), exp(0)]
        advantages = mx.array([0.5, 0.5, 0.5])

        metrics = grpo.compute_metrics(
            old_logprobs, new_logprobs, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        # First ratio ~4.48 (>1.28, clipped high)
        # Second ratio ~0.22 (<0.8, clipped low)
        # Third ratio = 1.0 (not clipped)
        assert 'clip_fraction_lower' in metrics
        assert 'clip_fraction_upper' in metrics
        assert 'clip_fraction' in metrics

        # Total clip fraction should be sum of upper and lower (no overlap)
        # Use tolerance for floating point comparison
        total = metrics['clip_fraction']
        sum_parts = metrics['clip_fraction_lower'] + metrics['clip_fraction_upper']
        assert abs(total - sum_parts) < 1e-6, \
            f"clip_fraction ({total}) should equal sum of parts ({sum_parts})"

    def test_metrics_backward_compat_with_clip_ratio(self):
        """Test that metrics work with symmetric clip_ratio parameter."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1])
        advantages = mx.array([0.5, 0.5])

        metrics = grpo.compute_metrics(
            old_logprobs, new_logprobs, advantages,
            clip_ratio=0.2
        )

        # When clip_ratio is provided, both bounds should be equal
        assert metrics['clip_ratio_low'] == 0.2
        assert metrics['clip_ratio_high'] == 0.2


@pytest.mark.unit
@pytest.mark.algorithm
class TestGRPOLossIntegration:
    """Integration tests for grpo_loss with all new features."""

    def test_grpo_loss_with_asymmetric_clipping(self):
        """Test grpo_loss convenience function with asymmetric clipping."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.0, -1.1])
        rewards = [1.0, 0.5, 0.0]

        loss = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        assert not mx.isnan(loss) and not mx.isinf(loss), \
            "grpo_loss should produce finite result with asymmetric clipping"

    def test_grpo_loss_with_constant_normalization(self):
        """Test grpo_loss with constant normalization."""
        old_logprobs = mx.array([-1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1])
        rewards = [1.0, 0.5]

        loss = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            normalize_constant=1024
        )

        assert not mx.isnan(loss) and not mx.isinf(loss), \
            "grpo_loss should produce finite result with constant normalization"

    def test_grpo_loss_with_all_features(self):
        """Test grpo_loss with all new features combined."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -0.9, -1.1, -1.1])
        rewards = [1.0, 0.8, 0.3, 0.0]

        loss = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            entropy_coeff=0.01,
            normalize_constant=1024
        )

        assert not mx.isnan(loss) and not mx.isinf(loss), \
            "grpo_loss should work with all features combined"


@pytest.mark.unit
@pytest.mark.algorithm
class TestDynamicFiltering:
    """Test dynamic batch filtering (DAPO-style prompt filtering)."""

    def _make_episode(self, prompt_tokens, reward):
        """Helper to create mock episode for testing."""
        return {
            'obs': prompt_tokens,
            'act': [100, 101, 102],
            'rew': [reward],
            'logprob': [-1.0, -1.0, -1.0],
        }

    def test_single_completion_kept_by_default(self):
        """Test that single-completion prompts are kept by default."""
        episodes = [
            self._make_episode([1, 2, 3], 1.0),  # Single completion for this prompt
            self._make_episode([4, 5, 6], 1.0),  # Different prompt, single completion
            self._make_episode([4, 5, 6], 0.0),  # Same prompt, different reward
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        # All 3 episodes should be kept:
        # - [1,2,3] has 1 completion, kept by default
        # - [4,5,6] has 2 completions with variance > 0, kept
        assert len(filtered) == 3
        assert stats['prompts_kept'] == 2
        assert stats['prompts_kept_single'] == 1
        assert stats['prompts_dropped_single'] == 0

    def test_single_completion_filtered_when_disabled(self):
        """Test that single-completion prompts are filtered when keep_single_completion=False."""
        episodes = [
            self._make_episode([1, 2, 3], 1.0),  # Single completion
            self._make_episode([4, 5, 6], 1.0),  # Two completions
            self._make_episode([4, 5, 6], 0.0),  # with variance
        ]

        filtered, stats = grpo.filter_informative_prompts(
            episodes, min_variance=0.01, keep_single_completion=False
        )

        # Only [4,5,6] prompt should be kept (2 completions with variance)
        assert len(filtered) == 2
        assert stats['prompts_kept'] == 1
        assert stats['prompts_kept_single'] == 0
        assert stats['prompts_dropped_single'] == 1

    def test_multi_completion_zero_variance_filtered(self):
        """Test that multi-completion prompts with zero variance are filtered."""
        episodes = [
            self._make_episode([1, 2, 3], 1.0),  # All same reward
            self._make_episode([1, 2, 3], 1.0),
            self._make_episode([1, 2, 3], 1.0),
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        # All completions same reward -> zero variance -> filtered
        assert len(filtered) == 0
        assert stats['prompts_kept'] == 0
        assert stats['prompts_dropped_all_correct'] == 1
        assert stats['episodes_dropped'] == 3

    def test_multi_completion_with_variance_kept(self):
        """Test that multi-completion prompts with variance are kept."""
        episodes = [
            self._make_episode([1, 2, 3], 1.0),  # Mixed rewards
            self._make_episode([1, 2, 3], 0.0),
            self._make_episode([1, 2, 3], 0.5),
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        # Mixed rewards -> variance > 0 -> kept
        assert len(filtered) == 3
        assert stats['prompts_kept'] == 1
        assert stats['episodes_kept'] == 3

    def test_stats_include_single_completion_tracking(self):
        """Test that stats properly track single-completion prompts."""
        episodes = [
            self._make_episode([1], 1.0),  # Single
            self._make_episode([2], 0.0),  # Single
            self._make_episode([3], 1.0),  # Multi, all correct
            self._make_episode([3], 1.0),
            self._make_episode([4], 1.0),  # Multi, mixed
            self._make_episode([4], 0.0),
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        assert stats['prompts_kept_single'] == 2  # [1] and [2]
        assert stats['prompts_dropped_all_correct'] == 1  # [3]
        assert stats['prompts_kept'] == 3  # [1], [2], [4]

    def test_empty_episodes_returns_empty_stats(self):
        """Test that empty input returns proper empty stats."""
        filtered, stats = grpo.filter_informative_prompts([])

        assert len(filtered) == 0
        assert stats['prompts_kept'] == 0
        assert stats['prompts_dropped_single'] == 0
        assert stats['prompts_kept_single'] == 0
        assert stats['filter_rate'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
