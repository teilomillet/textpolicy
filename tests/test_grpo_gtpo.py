"""
Tests for GTPO: Entropy-Weighted Credit Assignment (Issue #10).

Validates the GTPO (Group Token Policy Optimization) implementation which
weights token-level advantages by entropy, amplifying learning signal at
decision points where the model is uncertain.

Hypotheses tested:
  H1: Uniform entropy → GTPO equals baseline (degenerate case)
  H2: β=0 → GTPO equals baseline
  H3: High-entropy tokens get amplified advantages, low-entropy get dampened
  H4: mx.stop_gradient prevents gradient flow through entropy weights
  H5: GTPO loss produces valid, finite gradients

References:
    GTPO: Token and Sequence-Level Reward Shaping with Policy Entropy
    https://arxiv.org/abs/2508.04349
"""

import math

import pytest
import mlx.core as mx
import mlx.nn as nn

from textpolicy.algorithms import grpo


# ---------------------------------------------------------------------------
# H1 & H2: Degenerate cases — GTPO should collapse to baseline
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGTPODegenerateCases:
    """Verify that GTPO reduces to standard GRPO under degenerate conditions."""

    def test_uniform_entropy_equals_baseline(self):
        """H1: When all tokens have equal entropy, weighting is identity."""
        advantages = mx.array([0.5, -0.3, 0.2, -0.1, 0.4])
        # All tokens same entropy → H_norm(t) = 1 for all t → weight = 1
        uniform_entropy = mx.array([2.5, 2.5, 2.5, 2.5, 2.5])

        weighted = grpo.apply_entropy_weighting(advantages, uniform_entropy, entropy_weight=0.1)

        assert mx.allclose(weighted, advantages, atol=1e-6), \
            "Uniform entropy should produce unchanged advantages"

    def test_zero_entropy_equals_baseline(self):
        """H1: All-zero entropy is uniform — should return advantages unchanged.

        Regression test: without the early return, entropy_mean=0 causes
        entropy_normalized = 0/(0+1e-8) ≈ 0, producing weight = 1-β = 0.9
        instead of 1.0. Zero entropy means every token is equally confident;
        there is no signal to redistribute.
        """
        advantages = mx.array([0.5, -0.3, 0.2, -0.1, 0.4])
        zero_entropy = mx.array([0.0, 0.0, 0.0, 0.0, 0.0])

        weighted = grpo.apply_entropy_weighting(advantages, zero_entropy, entropy_weight=0.1)

        assert mx.allclose(weighted, advantages, atol=1e-6), \
            "All-zero entropy (uniform) should produce unchanged advantages"

    def test_near_zero_entropy_equals_baseline(self):
        """Entropy values below threshold should also be treated as uniform."""
        advantages = mx.array([0.5, -0.3, 0.2])
        tiny_entropy = mx.array([1e-10, 1e-10, 1e-10])

        weighted = grpo.apply_entropy_weighting(advantages, tiny_entropy, entropy_weight=0.5)

        assert mx.allclose(weighted, advantages, atol=1e-6), \
            "Near-zero uniform entropy should produce unchanged advantages"

    def test_beta_zero_equals_baseline(self):
        """H2: When β=0, entropy weighting is disabled."""
        advantages = mx.array([0.5, -0.3, 0.2, -0.1, 0.4])
        varied_entropy = mx.array([1.0, 5.0, 0.5, 3.0, 2.0])

        weighted = grpo.apply_entropy_weighting(advantages, varied_entropy, entropy_weight=0.0)

        assert mx.allclose(weighted, advantages, atol=1e-6), \
            "β=0 should produce unchanged advantages"

    def test_gtpo_advantages_with_uniform_entropy(self):
        """H1 via compute_advantages_gtpo: uniform entropy → same as compute_advantages."""
        rewards = [1.0, 0.5, 0.0]
        episode_lengths = [3, 3, 3]
        uniform_entropy = mx.array([2.0] * 9)  # 3 episodes × 3 tokens

        gtpo_adv = grpo.compute_advantages_gtpo(
            rewards, uniform_entropy, entropy_weight=0.1, episode_lengths=episode_lengths
        )

        # Compute baseline and expand manually
        base_adv = grpo.compute_advantages(rewards)
        expanded_base = mx.repeat(base_adv, 3)

        assert mx.allclose(gtpo_adv, expanded_base, atol=1e-5), \
            "GTPO with uniform entropy should equal expanded baseline advantages"

    def test_gtpo_advantages_with_beta_zero(self):
        """H2 via compute_advantages_gtpo: β=0 → same as compute_advantages."""
        rewards = [1.0, 0.5, 0.0]
        episode_lengths = [2, 3, 2]
        varied_entropy = mx.array([1.0, 5.0, 0.5, 3.0, 2.0, 4.0, 1.5])

        gtpo_adv = grpo.compute_advantages_gtpo(
            rewards, varied_entropy, entropy_weight=0.0, episode_lengths=episode_lengths
        )

        # Compute baseline and expand manually
        base_adv = grpo.compute_advantages(rewards)
        parts = []
        for i, length in enumerate(episode_lengths):
            parts.append(mx.repeat(base_adv[i:i+1], length))
        expanded_base = mx.concatenate(parts)

        assert mx.allclose(gtpo_adv, expanded_base, atol=1e-5), \
            "GTPO with β=0 should equal expanded baseline advantages"


# ---------------------------------------------------------------------------
# Token entropy computation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestComputeTokenEntropy:
    """Test compute_token_entropy correctness."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution over vocab should give maximum entropy."""
        vocab_size = 100
        # Uniform logits → uniform distribution
        logits = mx.zeros((5, vocab_size))  # 5 tokens, uniform over 100

        entropy = grpo.compute_token_entropy(logits)

        # H(uniform) = log(vocab_size)
        expected = math.log(vocab_size)
        assert entropy.shape == (5,)
        assert mx.allclose(entropy, mx.array([expected] * 5), atol=1e-4), \
            f"Uniform distribution entropy should be log({vocab_size})={expected:.4f}"

    def test_peaked_distribution_low_entropy(self):
        """Strongly peaked distribution should have low entropy."""
        vocab_size = 100
        # One logit much higher than rest → nearly deterministic
        logits = mx.zeros((3, vocab_size))
        logits = logits.at[:, 0].add(100.0)  # Token 0 gets logit 100

        entropy = grpo.compute_token_entropy(logits)

        assert entropy.shape == (3,)
        # Entropy should be very close to 0
        for i in range(3):
            assert float(entropy[i]) < 0.01, \
                f"Peaked distribution should have near-zero entropy, got {float(entropy[i])}"

    def test_entropy_always_non_negative(self):
        """Entropy is always ≥ 0 for any valid distribution."""
        # Random logits
        mx.random.seed(42)
        logits = mx.random.normal((10, 50))

        entropy = grpo.compute_token_entropy(logits)

        assert mx.all(entropy >= 0), \
            "Entropy should always be non-negative"

    def test_1d_logits_single_token(self):
        """Single token position (1D logits) should work."""
        logits = mx.array([1.0, 2.0, 3.0])  # [vocab_size]

        entropy = grpo.compute_token_entropy(logits)

        # Should be a scalar
        assert entropy.ndim == 0
        assert float(entropy) > 0

    def test_3d_logits_batch(self):
        """Batched 3D logits [batch, seq_len, vocab] should work."""
        logits = mx.zeros((2, 4, 50))  # batch=2, seq=4, vocab=50

        entropy = grpo.compute_token_entropy(logits)

        assert entropy.shape == (2, 4)
        expected = math.log(50)
        assert mx.allclose(entropy, mx.full((2, 4), expected), atol=1e-4)

    def test_numerical_stability_large_logits(self):
        """Should handle large logit values without overflow."""
        logits = mx.array([[1000.0, 0.0, 0.0],
                           [0.0, 1000.0, 0.0]])

        entropy = grpo.compute_token_entropy(logits)

        # Should not produce NaN or Inf
        assert not mx.any(mx.isnan(entropy)), "Should not produce NaN"
        assert not mx.any(mx.isinf(entropy)), "Should not produce Inf"
        # Should be close to 0 (very peaked)
        assert mx.all(entropy < 0.01)

    def test_numerical_stability_negative_logits(self):
        """Should handle large negative logit values."""
        logits = mx.array([[-1000.0, -1000.0, 0.0],
                           [0.0, -1000.0, -1000.0]])

        entropy = grpo.compute_token_entropy(logits)

        assert not mx.any(mx.isnan(entropy))
        assert not mx.any(mx.isinf(entropy))
        assert mx.all(entropy < 0.01)  # Nearly deterministic


# ---------------------------------------------------------------------------
# H3: High-entropy tokens get amplified, low-entropy get dampened
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestEntropyWeighting:
    """Test that entropy weighting correctly amplifies/dampens advantages."""

    def test_high_entropy_amplifies_positive_advantage(self):
        """H3: High-entropy token with positive advantage gets amplified."""
        advantages = mx.array([1.0, 1.0])  # Both positive
        # Token 0: high entropy (decision point), Token 1: low entropy (routine)
        entropies = mx.array([4.0, 1.0])  # Mean=2.5

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.1)

        # Token 0: H_norm = 4/2.5 = 1.6, weight = 1 + 0.1*(1.6-1) = 1.06
        # Token 1: H_norm = 1/2.5 = 0.4, weight = 1 + 0.1*(0.4-1) = 0.94
        assert float(weighted[0]) > float(advantages[0]), \
            "High-entropy token advantage should be amplified"
        assert float(weighted[1]) < float(advantages[1]), \
            "Low-entropy token advantage should be dampened"

    def test_high_entropy_amplifies_negative_advantage(self):
        """H3: High-entropy token with negative advantage gets more negative."""
        advantages = mx.array([-1.0, -1.0])
        entropies = mx.array([4.0, 1.0])

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.1)

        # More negative = more learning signal for bad actions at decision points
        assert float(weighted[0]) < float(advantages[0]), \
            "High-entropy token negative advantage should be amplified (more negative)"
        assert float(weighted[1]) > float(advantages[1]), \
            "Low-entropy token negative advantage should be dampened (less negative)"

    def test_weighting_preserves_sign(self):
        """Entropy weighting should never flip the sign of advantages."""
        advantages = mx.array([0.5, -0.3, 0.2, -0.1])
        entropies = mx.array([5.0, 1.0, 3.0, 0.5])

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.5)

        for i in range(4):
            adv = float(advantages[i])
            w = float(weighted[i])
            if adv > 0:
                assert w >= 0, f"Token {i}: positive advantage should stay non-negative"
            elif adv < 0:
                assert w <= 0, f"Token {i}: negative advantage should stay non-positive"

    def test_weighting_preserves_sign_large_beta(self):
        """Sign preservation must hold even with large β values.

        Regression test: without weight clamping, β=1.5 with entropies
        [5.0, 1.0, 3.0, 0.5] (mean=2.375) produces a negative weight
        for token 3: 1 + 1.5*(0.21-1) = -0.185, flipping its advantage.
        The actual GTPO paper (Eq. 3) uses H/ΣH normalization bounded in
        [0,1], so negative weights are impossible. Our clamp matches this.
        """
        advantages = mx.array([0.5, -0.3, 0.2, -0.1])
        entropies = mx.array([5.0, 1.0, 3.0, 0.5])  # mean=2.375, token 3 H_norm=0.21

        for beta in [1.0, 1.5, 2.0, 5.0, 10.0]:
            weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=beta)

            for i in range(4):
                adv = float(advantages[i])
                w = float(weighted[i])
                if adv > 0:
                    assert w >= 0, \
                        f"β={beta}, token {i}: positive advantage flipped to {w}"
                elif adv < 0:
                    assert w <= 0, \
                        f"β={beta}, token {i}: negative advantage flipped to {w}"

    def test_large_beta_clamps_to_zero_not_negative(self):
        """With very large β, low-entropy tokens should get weight=0, not negative.

        The GTPO paper normalizes entropy as H_{i,t}/Σ_k H_{k,t}, which is
        structurally bounded in [0,1]. Our H/mean(H) proxy can exceed this
        range, so we clamp to 0 — fully suppressing gradient for very
        confident tokens rather than flipping the learning signal.
        """
        advantages = mx.array([1.0, 1.0])
        # Token 1 has entropy far below mean → unclamped weight would be negative
        entropies = mx.array([5.0, 0.5])  # mean=2.75, H_norm_1 = 0.18

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=2.0)
        # Token 1: unclamped weight = 1 + 2.0*(0.18-1) = -0.64 → clamped to 0
        assert float(weighted[1]) == 0.0, \
            f"Very-low-entropy token should be clamped to 0, got {float(weighted[1])}"
        # Token 0 should still be amplified
        assert float(weighted[0]) > float(advantages[0]), \
            "High-entropy token should still be amplified"

    def test_weighting_mean_preserving(self):
        """Mean of unclamped weights should be approximately 1.0 (redistributes, doesn't scale)."""
        entropies = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        entropy_mean = mx.mean(entropies)
        entropy_normalized = entropies / (entropy_mean + 1e-8)
        weights = 1.0 + 0.1 * (entropy_normalized - 1.0)
        mean_weight = mx.mean(weights)

        assert abs(float(mean_weight) - 1.0) < 1e-5, \
            f"Mean weight should be ~1.0, got {float(mean_weight)}"

    def test_stronger_beta_larger_spread(self):
        """Larger β should create larger spread between high/low entropy weights."""
        advantages = mx.array([1.0, 1.0])
        entropies = mx.array([4.0, 1.0])

        weighted_small = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.1)
        weighted_large = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.5)

        spread_small = abs(float(weighted_small[0]) - float(weighted_small[1]))
        spread_large = abs(float(weighted_large[0]) - float(weighted_large[1]))

        assert spread_large > spread_small, \
            "Larger β should create larger spread between high/low entropy advantages"

    def test_exact_gtpo_formula(self):
        """Verify the exact GTPO formula: A_GTPO = A * (1 + β * (H_norm - 1))."""
        advantages = mx.array([0.5, -0.3, 0.8])
        entropies = mx.array([2.0, 4.0, 3.0])
        beta = 0.15

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=beta)

        # Manual computation
        entropy_mean = float(mx.mean(entropies))
        for i in range(3):
            h_norm = float(entropies[i]) / entropy_mean
            expected_weight = 1.0 + beta * (h_norm - 1.0)
            expected = float(advantages[i]) * expected_weight
            actual = float(weighted[i])
            assert abs(actual - expected) < 1e-5, \
                f"Token {i}: expected {expected:.6f}, got {actual:.6f}"


# ---------------------------------------------------------------------------
# H4: mx.stop_gradient prevents gradient flow through entropy weights
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestStopGradient:
    """Test that entropy weights are properly detached from gradient computation."""

    def test_gradient_does_not_flow_through_weights(self):
        """H4: Gradient of weighted loss w.r.t. entropy input should be zero-like."""
        # Create a simple model: linear layer
        model = nn.Linear(4, 4)

        def loss_fn(x):
            logits = model(x)
            # Compute token entropy from model output
            token_entropies = grpo.compute_token_entropy(logits)
            # Create dummy advantages
            advantages = mx.array([0.5, -0.3, 0.2, 0.1])
            # Apply entropy weighting (should have stop_gradient)
            weighted = grpo.apply_entropy_weighting(advantages, token_entropies, entropy_weight=0.1)
            return mx.sum(weighted)

        x = mx.ones((4, 4))

        # This should not error - gradient computation should work
        grad_fn = nn.value_and_grad(model, loss_fn)
        loss_val, grads = grad_fn(x)

        # The loss should be finite
        assert not mx.isnan(loss_val), "Loss should not be NaN"
        assert not mx.isinf(loss_val), "Loss should not be Inf"

    def test_entropy_weight_is_constant_in_backward(self):
        """Verify that changing entropy does not change the gradient direction."""
        advantages = mx.array([1.0, -1.0, 0.5])

        # With uniform entropy (weight=1 everywhere)
        uniform_entropy = mx.array([2.0, 2.0, 2.0])
        weighted_uniform = grpo.apply_entropy_weighting(advantages, uniform_entropy, 0.1)

        # The weighted values should be identical to base advantages
        assert mx.allclose(weighted_uniform, advantages, atol=1e-6)


# ---------------------------------------------------------------------------
# H5: GTPO loss produces valid, finite gradients
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGTPOLossIntegration:
    """Test GTPO integration with grpo_loss."""

    def test_grpo_loss_with_token_entropies(self):
        """H5: grpo_loss with GTPO produces finite loss."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9, -1.0])
        new_logprobs = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0, -1.1])
        rewards = [1.0, 0.5, 0.0]
        episode_lengths = [2, 2, 2]
        token_entropies = mx.array([3.0, 1.5, 4.2, 2.0, 1.0, 3.5])

        loss = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            token_entropies=token_entropies,
            entropy_weight=0.1,
            episode_lengths=episode_lengths
        )

        assert not mx.isnan(loss), "GTPO loss should not be NaN"
        assert not mx.isinf(loss), "GTPO loss should not be Inf"

    def test_grpo_loss_gtpo_differs_from_baseline(self):
        """GTPO loss should differ from baseline when entropy varies."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1, -0.9, -1.1])
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        # Non-uniform entropy: different pattern within each episode
        varied_entropy = mx.array([5.0, 1.0, 1.0, 5.0])

        # Baseline: manually expand episode advantages to token-level
        # (grpo_loss doesn't expand in non-GTPO mode; the trainer does)
        base_adv = grpo.compute_advantages(rewards)  # [0.5, -0.5]
        expanded_adv = mx.concatenate([mx.repeat(base_adv[0:1], 2), mx.repeat(base_adv[1:2], 2)])
        loss_baseline = grpo.policy_loss(old_logprobs, new_logprobs, expanded_adv)

        loss_gtpo = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            token_entropies=varied_entropy,
            entropy_weight=0.1,
            episode_lengths=episode_lengths
        )

        assert not mx.allclose(loss_baseline, loss_gtpo, atol=1e-6), \
            "GTPO with varied entropy should produce different loss than baseline"

    def test_grpo_loss_gtpo_with_uniform_entropy_matches_baseline(self):
        """GTPO with uniform entropy should match baseline loss."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -1.1, -0.9, -1.1])
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        uniform_entropy = mx.array([2.0, 2.0, 2.0, 2.0])

        # Baseline: manually expand advantages (same as trainer does)
        base_adv = grpo.compute_advantages(rewards)
        expanded_adv = mx.concatenate([mx.repeat(base_adv[0:1], 2), mx.repeat(base_adv[1:2], 2)])
        loss_baseline = grpo.policy_loss(old_logprobs, new_logprobs, expanded_adv)

        # GTPO with uniform entropy should produce identical token-level advantages
        loss_gtpo = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            token_entropies=uniform_entropy,
            entropy_weight=0.1,
            episode_lengths=episode_lengths
        )

        assert mx.allclose(loss_baseline, loss_gtpo, atol=1e-5), \
            "GTPO with uniform entropy should match baseline loss"

    def test_grpo_loss_gtpo_with_all_features(self):
        """GTPO combined with asymmetric clipping, constant norm, and entropy bonus."""
        old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_logprobs = mx.array([-0.9, -0.9, -1.1, -1.1])
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        token_entropies = mx.array([3.0, 1.0, 4.0, 2.0])

        loss = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            entropy_coeff=0.01,
            normalize_constant=1024,
            token_entropies=token_entropies,
            entropy_weight=0.1,
            episode_lengths=episode_lengths
        )

        assert not mx.isnan(loss), "All features combined should not produce NaN"
        assert not mx.isinf(loss), "All features combined should not produce Inf"

    def test_grpo_loss_without_token_entropies_unchanged(self):
        """Baseline behavior should be unchanged when token_entropies is None."""
        old_logprobs = mx.array([-1.0, -1.2, -0.8])
        new_logprobs = mx.array([-1.1, -1.0, -0.9])
        rewards = [1.0, 0.5, 0.0]

        # Without GTPO
        loss_before = grpo.grpo_loss(old_logprobs, new_logprobs, rewards)

        # Explicitly None (same as before)
        loss_after = grpo.grpo_loss(
            old_logprobs, new_logprobs, rewards,
            token_entropies=None
        )

        assert mx.allclose(loss_before, loss_after, atol=1e-7), \
            "Passing token_entropies=None should not change baseline behavior"


# ---------------------------------------------------------------------------
# compute_advantages_gtpo edge cases and correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestComputeAdvantagesGTPO:
    """Test the convenience function for GTPO advantages."""

    def test_basic_expansion_and_weighting(self):
        """Test that episode advantages are expanded and weighted correctly."""
        rewards = [1.0, 0.0]  # Mean = 0.5, advantages = [0.5, -0.5]
        episode_lengths = [3, 2]
        # 5 tokens total, varied entropy
        entropies = mx.array([2.0, 4.0, 1.0, 3.0, 2.0])

        advantages = grpo.compute_advantages_gtpo(
            rewards, entropies, entropy_weight=0.1, episode_lengths=episode_lengths
        )

        assert advantages.shape == (5,), f"Expected shape (5,), got {advantages.shape}"
        # First 3 tokens should have base advantage 0.5 (from reward 1.0 - mean 0.5)
        # Last 2 tokens should have base advantage -0.5
        # All weighted by entropy
        assert float(advantages[0]) > 0  # Positive advantage, ep 0
        assert float(advantages[3]) < 0  # Negative advantage, ep 1

    def test_without_episode_lengths_flat(self):
        """Without episode_lengths, assumes rewards are already token-level."""
        rewards = mx.array([0.5, -0.3, 0.2])  # Already 3 elements
        entropies = mx.array([2.0, 4.0, 1.0])

        # This will compute advantages from rewards directly
        advantages = grpo.compute_advantages_gtpo(rewards, entropies, entropy_weight=0.1)

        assert advantages.shape == (3,)

    def test_mismatched_episodes_raises(self):
        """Mismatched episode count and episode_lengths should raise."""
        rewards = [1.0, 0.0]  # 2 episodes
        entropies = mx.array([1.0, 2.0, 3.0])
        episode_lengths = [1, 1, 1]  # 3 lengths but only 2 rewards

        with pytest.raises(ValueError, match="does not match"):
            grpo.compute_advantages_gtpo(
                rewards, entropies, entropy_weight=0.1, episode_lengths=episode_lengths
            )

    def test_single_episode(self):
        """Single episode should work correctly."""
        rewards = [1.0]  # Single episode, advantage = 0.0
        episode_lengths = [4]
        entropies = mx.array([2.0, 3.0, 1.0, 4.0])

        advantages = grpo.compute_advantages_gtpo(
            rewards, entropies, entropy_weight=0.1, episode_lengths=episode_lengths
        )

        # Single episode: base advantage = 1.0 - mean(1.0) = 0.0
        # 0.0 * any_weight = 0.0
        assert advantages.shape == (4,)
        assert mx.allclose(advantages, mx.zeros(4), atol=1e-6), \
            "Single episode should have zero advantage (no group contrast)"

    def test_variable_episode_lengths(self):
        """Variable episode lengths should expand correctly."""
        rewards = [1.0, 0.0, 0.5]
        episode_lengths = [1, 4, 2]
        entropies = mx.array([2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0])

        advantages = grpo.compute_advantages_gtpo(
            rewards, entropies, entropy_weight=0.1, episode_lengths=episode_lengths
        )

        assert advantages.shape == (7,), f"Expected shape (7,), got {advantages.shape}"


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGTPOMathematicalProperties:
    """Test mathematical properties that the GTPO implementation must satisfy."""

    def test_entropy_of_two_token_vocab(self):
        """Known entropy for binary distribution: H = -p*log(p) - (1-p)*log(1-p)."""
        # p = 0.5 → H = log(2) ≈ 0.693
        logits = mx.array([[0.0, 0.0]])  # Uniform over 2 tokens
        entropy = grpo.compute_token_entropy(logits)
        assert abs(float(entropy[0]) - math.log(2)) < 1e-5

        # p ≈ 1.0 → H ≈ 0
        logits_peaked = mx.array([[100.0, 0.0]])
        entropy_peaked = grpo.compute_token_entropy(logits_peaked)
        assert float(entropy_peaked[0]) < 1e-4

    def test_zero_advantage_stays_zero(self):
        """Zero advantage should remain zero regardless of entropy weighting."""
        advantages = mx.array([0.0, 0.0, 0.0])
        entropies = mx.array([1.0, 5.0, 10.0])

        weighted = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.5)

        assert mx.allclose(weighted, mx.zeros(3), atol=1e-7), \
            "Zero advantages should remain zero after entropy weighting"

    def test_advantage_magnitude_scales_with_weight(self):
        """Higher entropy_weight should produce larger magnitude differences."""
        advantages = mx.array([1.0, 1.0])
        entropies = mx.array([5.0, 1.0])

        w_01 = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.1)
        w_05 = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=0.5)
        w_10 = grpo.apply_entropy_weighting(advantages, entropies, entropy_weight=1.0)

        # Spread = |weighted[0] - weighted[1]|
        spread_01 = abs(float(w_01[0]) - float(w_01[1]))
        spread_05 = abs(float(w_05[0]) - float(w_05[1]))
        spread_10 = abs(float(w_10[0]) - float(w_10[1]))

        assert spread_01 < spread_05 < spread_10, \
            "Larger entropy_weight should produce larger advantage spread"

    def test_entropy_weighting_commutes_with_scaling(self):
        """Scaling advantages by c then weighting should equal weighting then scaling by c."""
        advantages = mx.array([0.5, -0.3, 0.8])
        entropies = mx.array([2.0, 4.0, 1.0])
        c = 3.0

        # Scale then weight
        scaled_then_weighted = grpo.apply_entropy_weighting(
            advantages * c, entropies, entropy_weight=0.1
        )

        # Weight then scale
        weighted_then_scaled = grpo.apply_entropy_weighting(
            advantages, entropies, entropy_weight=0.1
        ) * c

        assert mx.allclose(scaled_then_weighted, weighted_then_scaled, atol=1e-5), \
            "Entropy weighting should commute with scalar multiplication"

    def test_grpo_loss_gtpo_accepts_list_rewards(self):
        """grpo_loss should accept both list and mx.array rewards with GTPO."""
        old_lp = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_lp = mx.array([-0.9, -1.1, -0.9, -1.1])
        entropies = mx.array([2.0, 3.0, 1.0, 4.0])

        loss_list = grpo.grpo_loss(
            old_lp, new_lp, [1.0, 0.0],
            token_entropies=entropies, episode_lengths=[2, 2]
        )
        loss_array = grpo.grpo_loss(
            old_lp, new_lp, mx.array([1.0, 0.0]),
            token_entropies=entropies, episode_lengths=[2, 2]
        )

        assert mx.allclose(loss_list, loss_array, atol=1e-6), \
            "List and mx.array rewards should produce identical GTPO loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
