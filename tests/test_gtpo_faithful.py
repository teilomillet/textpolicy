"""
Tests for GTPO Faithful: Paper-Exact Implementation (arXiv 2508.04349).

Validates the paper-faithful GTPO implementation against every equation,
proposition, and remark from "GTPO and GRPO-S: Token and Sequence-Level
Reward Shaping with Policy Entropy" (Tan et al., 2025).

Hypotheses tested:
  H1: Eq. 3 (O+ shaping) produces correct shaped rewards with manual verification
  H2: Eq. 5 (O- shaping) uses inverse entropy — confident mistakes penalized harder
  H3: Eq. 6 (advantage normalization) normalizes O+ and O- separately
  H4: Eq. 7 (GTPO objective) produces finite loss and valid gradients
  H5: Proposition 2.2 — reward conservation when α₁ + α₂ = 1
  H6: Position-dependent d_t / h_t handle variable-length episodes correctly
  H7: Edge cases — empty O+, empty O-, single episode, zero entropy
  H8: mx.stop_gradient — entropy does not contribute to gradient

References:
    GTPO and GRPO-S: Token and Sequence-Level Reward Shaping with Policy Entropy
    https://arxiv.org/abs/2508.04349
"""

import math

import pytest
import mlx.core as mx
import mlx.nn as nn

from textpolicy.algorithms.grpo import (
    compute_gtpo_shaped_rewards,
    normalize_gtpo_advantages,
    gtpo_loss_faithful,
)


# ---------------------------------------------------------------------------
# H1: Eq. 3 — Positive reward shaping
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestPositiveRewardShaping:
    """Verify Eq. 3: r̃⁺ᵢ,ₜ = α₁·rᵢ + α₂·(Hᵢ,ₜ/ΣH)·dₜ."""

    def test_exact_values_two_positive_episodes(self):
        """H1: Manual verification of Eq. 3 with two O+ episodes."""
        # 2 O+ episodes (reward=1), 1 O- episode (reward=0)
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        # ep0: H=[3.0, 1.0], ep1: H=[1.0, 3.0], ep2: H=[2.0, 2.0]
        entropies = mx.array([3.0, 1.0, 1.0, 3.0, 2.0, 2.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=1.0, alpha_2=0.1
        )
        mx.eval(shaped, is_pos)
        shaped_list = shaped.tolist()

        # Position t=0: O+ entropies [3.0, 1.0], sum=4.0, d_0=2
        # ep0 t=0: 1.0*1 + 0.1*(3.0/4.0)*2 = 1.0 + 0.15 = 1.15
        assert abs(shaped_list[0] - 1.15) < 1e-4, f"ep0 t=0: got {shaped_list[0]}"
        # ep1 t=0: 1.0*1 + 0.1*(1.0/4.0)*2 = 1.0 + 0.05 = 1.05
        assert abs(shaped_list[2] - 1.05) < 1e-4, f"ep1 t=0: got {shaped_list[2]}"

        # Position t=1: O+ entropies [1.0, 3.0], sum=4.0, d_1=2
        # ep0 t=1: 1.0*1 + 0.1*(1.0/4.0)*2 = 1.05
        assert abs(shaped_list[1] - 1.05) < 1e-4, f"ep0 t=1: got {shaped_list[1]}"
        # ep1 t=1: 1.0*1 + 0.1*(3.0/4.0)*2 = 1.15
        assert abs(shaped_list[3] - 1.15) < 1e-4, f"ep1 t=1: got {shaped_list[3]}"

    def test_high_entropy_gets_more_positive_reward(self):
        """H1: Within O+, high-entropy tokens get higher shaped reward."""
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([5.0, 1.0, 1.0, 1.0, 2.0, 2.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped)

        # ep0 t=0 (H=5.0) should have higher reward than ep1 t=0 (H=1.0)
        assert float(shaped[0]) > float(shaped[2]), \
            "Higher entropy O+ token should get more reward"

    def test_o_minus_tokens_get_zero_positive_reward(self):
        """Eq. 3: r̃⁺ⱼ,ₜ = 0 for oⱼ ∈ O⁻ (O- tokens get no positive reward)."""
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([3.0, 1.0, 2.0, 2.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped, is_pos)

        # O- tokens should have negative shaped rewards (from Eq. 5, not Eq. 3)
        # The positive contribution is 0 for O-
        for i in range(2, 4):
            assert float(shaped[i]) < 0, \
                f"O- token {i} should have negative shaped reward"


# ---------------------------------------------------------------------------
# H2: Eq. 5 — Negative reward shaping with inverse entropy
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestNegativeRewardShaping:
    """Verify Eq. 5: confident mistakes (low H) get stronger penalty."""

    def test_inverse_entropy_penalizes_confident_mistakes(self):
        """H2: O- token with low entropy (confident mistake) gets stronger penalty."""
        rewards = [1.0, 0.0, 0.0]
        episode_lengths = [2, 2, 2]
        # ep1 (O-): H=[0.5, 2.0] — first token is confident mistake
        # ep2 (O-): H=[4.0, 2.0] — first token is uncertain
        entropies = mx.array([3.0, 1.0, 0.5, 2.0, 4.0, 2.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped)

        # ep1 t=0 (H=0.5, confident mistake): stronger penalty (more negative)
        # ep2 t=0 (H=4.0, uncertain): weaker penalty (less negative)
        assert float(shaped[2]) < float(shaped[4]), \
            f"Confident mistake (H=0.5) should be more negative than uncertain (H=4.0). " \
            f"Got ep1={float(shaped[2]):.4f}, ep2={float(shaped[4]):.4f}"

    def test_exact_inverse_entropy_formula(self):
        """H2: Exact manual verification of Eq. 5."""
        rewards = [0.0, 0.0]
        episode_lengths = [2, 2]
        # ep0 (O-): H=[4.0, 1.0], ep1 (O-): H=[1.0, 4.0]
        entropies = mx.array([4.0, 1.0, 1.0, 4.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=1.0, alpha_2=0.1
        )
        mx.eval(shaped)
        s = shaped.tolist()

        # Position t=0: inv_H = [1/4=0.25, 1/1=1.0], sum=1.25, h_0=2
        # ep0: 1*(-1) + 0.1*(0.25/1.25)*2*(-1) = -1 - 0.04 = -1.04
        assert abs(s[0] - (-1.04)) < 1e-4, f"ep0 t=0: got {s[0]}"
        # ep1: 1*(-1) + 0.1*(1.0/1.25)*2*(-1) = -1 - 0.16 = -1.16
        assert abs(s[2] - (-1.16)) < 1e-4, f"ep1 t=0: got {s[2]}"

        # Position t=1: inv_H = [1/1=1.0, 1/4=0.25], sum=1.25, h_1=2
        # ep0: -1 - 0.1*(1.0/1.25)*2 = -1 - 0.16 = -1.16
        assert abs(s[1] - (-1.16)) < 1e-4, f"ep0 t=1: got {s[1]}"
        # ep1: -1 - 0.1*(0.25/1.25)*2 = -1 - 0.04 = -1.04
        assert abs(s[3] - (-1.04)) < 1e-4, f"ep1 t=1: got {s[3]}"

    def test_o_plus_tokens_get_zero_negative_reward(self):
        """Eq. 5: r̃⁻ᵢ,ₜ = 0 for oᵢ ∈ O⁺ (O+ tokens get no negative reward)."""
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([3.0, 1.0, 2.0, 2.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped, is_pos)

        # O+ tokens should have positive shaped rewards
        for i in range(2):
            assert float(shaped[i]) > 0, \
                f"O+ token {i} should have positive shaped reward"


# ---------------------------------------------------------------------------
# H3: Eq. 6 — Separate advantage normalization
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestAdvantageNormalization:
    """Verify Eq. 6: O+ and O- are normalized independently."""

    def test_separate_normalization(self):
        """H3: O+ and O- groups get independent mean/std normalization."""
        # 3 O+ tokens and 3 O- tokens with different distributions
        shaped = mx.array([1.0, 1.2, 0.8, -1.0, -1.2, -0.8])
        is_positive = mx.array([True, True, True, False, False, False])

        advantages = normalize_gtpo_advantages(shaped, is_positive)
        mx.eval(advantages)
        adv = advantages.tolist()

        # O+ group: mean=1.0, values=[1.0, 1.2, 0.8]
        # std = sqrt(((0)^2 + (0.2)^2 + (-0.2)^2)/3) = sqrt(0.08/3) ≈ 0.1633
        # O- group: mean=-1.0, values=[-1.0, -1.2, -0.8]
        # Same std (symmetric)
        # Normalized: [0, +, -, 0, -, +]
        assert abs(adv[0]) < 1e-4, "Mean of O+ should normalize to ~0"
        assert abs(adv[3]) < 1e-4, "Mean of O- should normalize to ~0"
        assert adv[1] > 0, "Above-mean O+ should be positive"
        assert adv[2] < 0, "Below-mean O+ should be negative"
        assert adv[4] < 0, "Below-mean O- should be negative"
        assert adv[5] > 0, "Above-mean O- should be positive"

    def test_single_token_per_group(self):
        """Edge case: single token in each group → advantage = 0."""
        shaped = mx.array([1.5, -1.3])
        is_positive = mx.array([True, False])

        advantages = normalize_gtpo_advantages(shaped, is_positive)
        mx.eval(advantages)

        # Single token: (x - mean(x)) / std(x) = 0/eps ≈ 0
        assert abs(float(advantages[0])) < 1e-2, "Single O+ token should be ~0"
        assert abs(float(advantages[1])) < 1e-2, "Single O- token should be ~0"


# ---------------------------------------------------------------------------
# H4: Eq. 7 — Full GTPO loss
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGTPOLossFaithful:
    """Verify the full GTPO objective (Eq. 7)."""

    def test_produces_finite_loss(self):
        """H4: Loss should be finite with normal inputs."""
        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9, -1.0])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0, -1.1])
        rewards = [1.0, 0.5, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([3.0, 1.5, 4.2, 2.0, 1.0, 3.5])

        loss = gtpo_loss_faithful(
            old_lp, new_lp, rewards, entropies, episode_lengths
        )
        mx.eval(loss)

        assert not mx.isnan(loss).item(), "Loss should not be NaN"
        assert not mx.isinf(loss).item(), "Loss should not be Inf"

    def test_differs_from_baseline_grpo(self):
        """H4: GTPO faithful should produce different advantages than GRPO.

        We compare shaped rewards directly rather than final loss values,
        because loss can be near zero for symmetric logprob patterns.
        The shaped rewards from GTPO (per-token, asymmetric O+/O-)
        must differ from GRPO (uniform per-episode).
        """
        from textpolicy.algorithms.grpo import compute_advantages

        rewards = [1.0, 1.0, 0.0, 0.0]
        episode_lengths = [2, 2, 2, 2]
        # Non-uniform entropy: different within each episode
        entropies = mx.array([5.0, 1.0, 1.0, 5.0, 4.0, 0.5, 0.5, 4.0])

        # Baseline GRPO: uniform per-episode advantages
        base_adv = compute_advantages(rewards)
        parts = [mx.repeat(base_adv[i:i+1], 2) for i in range(4)]
        grpo_advantages = mx.concatenate(parts)

        # GTPO faithful: per-token shaped rewards → normalized advantages
        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        gtpo_advantages = normalize_gtpo_advantages(shaped, is_pos)
        mx.eval(grpo_advantages, gtpo_advantages)

        # GTPO advantages should NOT be uniform per-episode (that's the whole point)
        # ep0 tokens should have different advantages from each other (H=5 vs H=1)
        assert not mx.allclose(
            gtpo_advantages[:2],
            mx.array([float(gtpo_advantages[0])] * 2),
            atol=1e-4
        ), "GTPO should produce non-uniform per-token advantages within an episode"

        # And overall should differ from GRPO
        assert not mx.allclose(grpo_advantages, gtpo_advantages, atol=1e-4), \
            "GTPO faithful advantages should differ from standard GRPO"

    def test_loss_with_all_features(self):
        """H4: Full loss with asymmetric clipping and varied entropy."""
        old_lp = mx.array([-1.0] * 6)
        new_lp = mx.array([-0.9, -1.1, -0.8, -1.2, -0.95, -1.05])
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([3.0, 1.0, 1.0, 4.0, 2.0, 2.0])

        loss = gtpo_loss_faithful(
            old_lp, new_lp, rewards, entropies, episode_lengths,
            alpha_1=1.0, alpha_2=0.1, clip_epsilon=0.2
        )
        mx.eval(loss)

        assert not mx.isnan(loss).item()
        assert not mx.isinf(loss).item()


# ---------------------------------------------------------------------------
# H5: Proposition 2.2 — Reward conservation
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestRewardConservation:
    """Verify Proposition 2.2: Σ r̃⁺ = Σ rᵢ when α₁+α₂=1."""

    def test_conservation_at_every_position(self):
        """H5: Sum of O+ shaped rewards = d_t at each position when α₁+α₂=1."""
        rewards = [1.0, 1.0, 0.0, 0.0]
        episode_lengths = [3, 2, 3, 2]
        entropies = mx.array([2.0, 4.0, 1.0, 3.0, 1.0, 4.0, 0.5, 3.0, 0.5, 4.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=0.9, alpha_2=0.1  # α₁+α₂ = 1.0
        )
        mx.eval(shaped)

        # Split back per episode
        ep_shaped = [
            shaped[:3].tolist(),   # ep0 (O+, len=3)
            shaped[3:5].tolist(),  # ep1 (O+, len=2)
            shaped[5:8].tolist(),  # ep2 (O-, len=3)
            shaped[8:10].tolist(), # ep3 (O-, len=2)
        ]

        for t in range(3):
            pos_sum = sum(ep_shaped[i][t] for i in [0, 1] if t < episode_lengths[i])
            neg_sum = sum(ep_shaped[i][t] for i in [2, 3] if t < episode_lengths[i])
            d_t = sum(1 for i in [0, 1] if t < episode_lengths[i])
            h_t = sum(1 for i in [2, 3] if t < episode_lengths[i])

            assert abs(pos_sum - d_t) < 1e-4, \
                f"t={t}: O+ sum={pos_sum:.4f} != d_t={d_t}"
            assert abs(neg_sum - (-h_t)) < 1e-4, \
                f"t={t}: O- sum={neg_sum:.4f} != -h_t={-h_t}"

    def test_conservation_violated_when_constraint_not_met(self):
        """H5 negative: Conservation does NOT hold when α₁+α₂ ≠ 1."""
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([3.0, 1.0, 1.0, 3.0, 2.0, 2.0])

        # α₁+α₂ = 1.1 ≠ 1 (paper experimental values)
        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=1.0, alpha_2=0.1
        )
        mx.eval(shaped)

        # Sum at t=0: should be 1.0*1+0.1*(3/4)*2 + 1.0*1+0.1*(1/4)*2 = 2.2 ≠ d_0=2
        pos_sum_t0 = float(shaped[0]) + float(shaped[2])
        assert abs(pos_sum_t0 - 2.0) > 0.01, \
            "Conservation should NOT hold when α₁+α₂ ≠ 1"


# ---------------------------------------------------------------------------
# H6: Position-dependent d_t / h_t
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestPositionDependentCounts:
    """Verify d_t and h_t decrease as shorter episodes become inactive."""

    def test_variable_length_episodes(self):
        """H6: d_t and h_t decrease as shorter episodes drop out."""
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [4, 2, 3]  # ep1 drops at t=2, ep2 drops at t=3
        # 4+2+3 = 9 tokens
        entropies = mx.array([
            2.0, 2.0, 2.0, 2.0,  # ep0 (O+, len=4)
            2.0, 2.0,             # ep1 (O+, len=2)
            2.0, 2.0, 2.0,       # ep2 (O-, len=3)
        ])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=0.9, alpha_2=0.1  # α₁+α₂=1 for conservation check
        )
        mx.eval(shaped)

        ep_shaped = [
            shaped[:4].tolist(),   # ep0 (O+, len=4)
            shaped[4:6].tolist(),  # ep1 (O+, len=2)
            shaped[6:9].tolist(),  # ep2 (O-, len=3)
        ]

        # t=0: d_0=2 (both O+ active), h_0=1
        d_0_sum = ep_shaped[0][0] + ep_shaped[1][0]
        assert abs(d_0_sum - 2.0) < 1e-4, f"d_0 sum: {d_0_sum}"

        # t=1: d_1=2 (both O+ still active), h_1=1
        d_1_sum = ep_shaped[0][1] + ep_shaped[1][1]
        assert abs(d_1_sum - 2.0) < 1e-4, f"d_1 sum: {d_1_sum}"

        # t=2: d_2=1 (only ep0 active, ep1 len=2 is inactive), h_2=1
        # ep0 t=2 should get full entropy share → shaped=1.0
        assert abs(ep_shaped[0][2] - 1.0) < 1e-4, \
            f"ep0 t=2 (sole O+ survivor): expected 1.0, got {ep_shaped[0][2]}"

        # t=3: d_3=1 (only ep0), h_3=0 (ep2 len=3 is now inactive)
        assert abs(ep_shaped[0][3] - 1.0) < 1e-4, \
            f"ep0 t=3 (sole survivor): expected 1.0, got {ep_shaped[0][3]}"


# ---------------------------------------------------------------------------
# H7: Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestEdgeCases:
    """Test edge cases: empty groups, single episode, zero entropy."""

    def test_all_positive_no_negative(self):
        """H7: When O- is empty, only O+ rewards are shaped."""
        rewards = [1.0, 1.0]
        episode_lengths = [2, 2]
        entropies = mx.array([3.0, 1.0, 1.0, 3.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped, is_pos)

        assert mx.all(is_pos).item(), "All tokens should be O+"
        assert mx.all(shaped > 0).item(), "All shaped rewards should be positive"

    def test_all_negative_no_positive(self):
        """H7: When O+ is empty, only O- rewards are shaped."""
        rewards = [0.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([3.0, 1.0, 1.0, 3.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped, is_pos)

        assert not mx.any(is_pos).item(), "All tokens should be O-"
        assert mx.all(shaped < 0).item(), "All shaped rewards should be negative"

    def test_single_episode(self):
        """H7: Single episode should still produce valid shaped rewards."""
        rewards = [1.0]
        episode_lengths = [3]
        entropies = mx.array([2.0, 4.0, 1.0])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped)

        assert shaped.shape == (3,)
        assert mx.all(shaped > 0).item()

    def test_zero_entropy_o_plus(self):
        """H7: Zero entropy in O+ — should produce valid results (Remark 2.1)."""
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([0.0, 0.0, 2.0, 2.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped)

        assert not mx.any(mx.isnan(shaped)).item(), "Should not produce NaN"
        assert not mx.any(mx.isinf(shaped)).item(), "Should not produce Inf"

    def test_zero_entropy_o_minus(self):
        """H7: Zero entropy in O- — 1/H is very large but handled by eps."""
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([2.0, 2.0, 0.0, 0.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped)

        assert not mx.any(mx.isnan(shaped)).item(), "Should not produce NaN"
        assert not mx.any(mx.isinf(shaped)).item(), "Should not produce Inf"

    def test_empty_episodes(self):
        """H7: Empty input should return empty arrays."""
        rewards = []
        episode_lengths = []
        entropies = mx.array([])

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths
        )
        mx.eval(shaped, is_pos)

        assert shaped.size == 0
        assert is_pos.size == 0

    def test_mismatched_lengths_raises(self):
        """Validation: sum(episode_lengths) != len(entropies) should raise."""
        rewards = [1.0, 0.0]
        episode_lengths = [3, 3]  # sum=6
        entropies = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])  # len=5

        with pytest.raises(ValueError, match="token_entropies length"):
            compute_gtpo_shaped_rewards(rewards, entropies, episode_lengths)

    def test_mismatched_rewards_episodes_raises(self):
        """Validation: rewards length != episode_lengths length should raise."""
        rewards = [1.0, 0.0]  # 2 episodes
        episode_lengths = [2, 2, 2]  # 3 episodes
        entropies = mx.array([1.0] * 6)

        with pytest.raises(ValueError, match="rewards length"):
            compute_gtpo_shaped_rewards(rewards, entropies, episode_lengths)


# ---------------------------------------------------------------------------
# H8: Gradient detachment (Remark 2.5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGradientDetachment:
    """Verify entropy weights are detached from gradient (Remark 2.5)."""

    def test_loss_produces_valid_gradient(self):
        """H8: gtpo_loss_faithful should produce valid gradients for model params."""
        model = nn.Linear(4, 4)

        def loss_fn(x):
            logits = model(x)
            # Use logits to compute "logprobs" and "entropies"
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            new_logprobs = log_probs[:, 0]  # pick first vocab entry

            old_logprobs = mx.array([-1.0, -1.0, -1.0, -1.0])
            rewards = [1.0, 0.0]
            episode_lengths = [2, 2]
            # Compute entropy from logits (which depends on model params)
            probs = mx.exp(log_probs)
            token_entropies = -mx.sum(probs * log_probs, axis=-1)

            return gtpo_loss_faithful(
                old_logprobs, new_logprobs, rewards,
                token_entropies, episode_lengths
            )

        x = mx.ones((4, 4))
        grad_fn = nn.value_and_grad(model, loss_fn)
        loss_val, grads = grad_fn(x)
        mx.eval(loss_val)

        assert not mx.isnan(loss_val).item(), "Loss should not be NaN"
        assert not mx.isinf(loss_val).item(), "Loss should not be Inf"

    def test_shaped_rewards_are_stop_gradiented(self):
        """H8: Shaped rewards use mx.stop_gradient — verify numerically."""
        def fn(token_entropies):
            rewards = [1.0, 0.0]
            episode_lengths = [2, 2]
            shaped, is_pos = compute_gtpo_shaped_rewards(
                rewards, token_entropies, episode_lengths
            )
            # Sum shaped rewards — if stop_gradient works, grad w.r.t. entropies = 0
            return mx.sum(shaped)

        entropies = mx.array([3.0, 1.0, 2.0, 2.0])
        grad_fn = mx.grad(fn)
        grad_entropies = grad_fn(entropies)
        mx.eval(grad_entropies)

        assert mx.allclose(
            grad_entropies, mx.zeros_like(grad_entropies), atol=1e-12
        ), f"Gradient w.r.t. entropies should be zero, got {grad_entropies}"


# ---------------------------------------------------------------------------
# Comprehensive numerical verification
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestNumericalVerification:
    """Paper-exact numerical verification with fully worked examples."""

    def test_four_episode_full_verification(self):
        """Complete hand-verified example with 4 episodes, variable lengths."""
        rewards = [1.0, 1.0, 0.0, 0.0]
        episode_lengths = [3, 2, 3, 2]
        entropies = mx.array([
            2.0, 4.0, 1.0,   # ep0 (O+, len=3)
            3.0, 1.0,         # ep1 (O+, len=2)
            4.0, 0.5, 3.0,   # ep2 (O-, len=3)
            0.5, 4.0,         # ep3 (O-, len=2)
        ])
        alpha_1, alpha_2 = 1.0, 0.1
        eps = 1e-8

        shaped, is_pos = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=alpha_1, alpha_2=alpha_2
        )
        mx.eval(shaped)
        s = shaped.tolist()

        # --- Position t=0 (all active) ---
        # O+ entropies: [2.0, 3.0], sum=5.0, d_0=2
        # ep0: 1.0*1 + 0.1*(2.0/5.0)*2 = 1.08
        assert abs(s[0] - 1.08) < 1e-4
        # ep1: 1.0*1 + 0.1*(3.0/5.0)*2 = 1.12
        assert abs(s[3] - 1.12) < 1e-4

        # O- inv_entropies: [1/4=0.25, 1/0.5=2.0], sum=2.25, h_0=2
        # ep2: -1 + 0.1*(0.25/2.25)*2*(-1) = -1 - 0.0222 = -1.0222
        assert abs(s[5] - (-1.0222)) < 1e-3
        # ep3: -1 + 0.1*(2.0/2.25)*2*(-1) = -1 - 0.1778 = -1.1778
        assert abs(s[8] - (-1.1778)) < 1e-3

        # --- Position t=1 (all active) ---
        # O+ entropies: [4.0, 1.0], sum=5.0, d_1=2
        # ep0: 1.0 + 0.1*(4.0/5.0)*2 = 1.16
        assert abs(s[1] - 1.16) < 1e-4
        # ep1: 1.0 + 0.1*(1.0/5.0)*2 = 1.04
        assert abs(s[4] - 1.04) < 1e-4

        # O- inv: ep2 H=0.5 → 1/0.5=2.0, ep3 H=4.0 → 1/4=0.25, sum=2.25
        # ep2: -1 - 0.1*(2.0/2.25)*2 = -1.1778
        assert abs(s[6] - (-1.1778)) < 1e-3
        # ep3: -1 - 0.1*(0.25/2.25)*2 = -1.0222
        assert abs(s[9] - (-1.0222)) < 1e-3

        # --- Position t=2 (ep1 and ep3 inactive) ---
        # O+ only ep0: H=1.0, sum=1.0, d_2=1
        # ep0: 1.0 + 0.1*(1.0/1.0)*1 = 1.1
        assert abs(s[2] - 1.1) < 1e-4

        # O- only ep2: H=3.0, inv=1/3, sum=1/3, h_2=1
        # ep2: -1 - 0.1*(1.0/1.0)*1 = -1.1
        assert abs(s[7] - (-1.1)) < 1e-4

    def test_uniform_entropy_equal_distribution(self):
        """When all O+ tokens have equal entropy, shaped rewards are uniform."""
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        shaped, _ = compute_gtpo_shaped_rewards(
            rewards, entropies, episode_lengths,
            alpha_1=0.9, alpha_2=0.1
        )
        mx.eval(shaped)

        # Equal entropy → each O+ gets fair share: α₁*1 + α₂*(1/d_t)*d_t = 1.0
        for i in range(4):  # O+ tokens
            assert abs(float(shaped[i]) - 1.0) < 1e-4, \
                f"Uniform entropy O+ token {i}: expected 1.0, got {float(shaped[i])}"

        # O- with equal entropy: α₁*(-1) + α₂*(1/h_t)*h_t*(-1) = -1.0
        for i in range(4, 6):
            assert abs(float(shaped[i]) - (-1.0)) < 1e-4, \
                f"Uniform entropy O- token {i}: expected -1.0, got {float(shaped[i])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
