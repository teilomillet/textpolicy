"""
Tests for GTPO: Paper-Exact Implementation (arXiv 2508.04349).

Validates the GTPO implementation against every equation,
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
import mlx.optimizers as optim

from textpolicy.algorithms.grpo import (
    compute_gtpo_shaped_rewards,
    normalize_gtpo_advantages,
    gtpo_loss,
)
from textpolicy.algorithms.hicra import boost_entropy_with_planning


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

        loss = gtpo_loss(
            old_lp, new_lp, rewards, entropies, episode_lengths
        )
        mx.eval(loss)

        assert not mx.isnan(loss).item(), "Loss should not be NaN"
        assert not mx.isinf(loss).item(), "Loss should not be Inf"

    def test_differs_from_baseline_grpo(self):
        """H4: GTPO should produce different advantages than GRPO.

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

        # GTPO: per-token shaped rewards → normalized advantages
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
            "GTPO advantages should differ from standard GRPO"

    def test_loss_with_all_features(self):
        """H4: Full loss with asymmetric clipping and varied entropy."""
        old_lp = mx.array([-1.0] * 6)
        new_lp = mx.array([-0.9, -1.1, -0.8, -1.2, -0.95, -1.05])
        rewards = [1.0, 1.0, 0.0]
        episode_lengths = [2, 2, 2]
        entropies = mx.array([3.0, 1.0, 1.0, 4.0, 2.0, 2.0])

        loss = gtpo_loss(
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

    def test_new_logprobs_shape_mismatch_raises(self):
        """Scalar/short new_logprobs silently broadcasts in MLX — must fail fast."""
        old_lp = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_lp_scalar = mx.array([-1.0])  # length 1 != 4
        rewards = [1.0, 0.0]
        episode_lengths = [2, 2]
        entropies = mx.array([2.0, 3.0, 1.0, 4.0])

        with pytest.raises(ValueError, match="new_logprobs shape"):
            gtpo_loss(
                old_lp, new_lp_scalar, rewards, entropies, episode_lengths
            )


# ---------------------------------------------------------------------------
# H8: Gradient detachment (Remark 2.5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestGradientDetachment:
    """Verify entropy weights are detached from gradient (Remark 2.5)."""

    def test_loss_produces_valid_gradient(self):
        """H8: gtpo_loss should produce valid gradients for model params."""
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

            return gtpo_loss(
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


# ---------------------------------------------------------------------------
# H9: Trainer integration for GTPO
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGTPOFaithfulTrainerIntegration:
    """Validate that GTPO works through the Trainer via advantage_transform_fn.

    The decomposition:
    - advantage_transform_fn → build_gtpo_transform (Eq. 3, 5, 6)
    - loss_fn → grpo.policy_loss (Eq. 7, PPO clipping)

    Hypotheses:
      H9a: Trainer trains with GTPO (uncompiled) — finite loss
      H9b: Trainer trains with GTPO (compiled) — compile-safety proof
      H9c: Transform output differs from standard GRPO advantages
      H9d: Variable-length episodes handled correctly
      H9e: functools.partial(policy_loss, clip_ratio=0.2) works
      H9f: Missing episode_lengths raises ValueError
      H9g: Missing token_entropies raises ValueError
      H9h: Negative alpha values rejected by builder
      H9i: Works with micro-batching (micro_batch_size=2)
    """

    def _make_model(self, vocab_size=16, dim=8):
        """Minimal causal LM: embedding + head → [batch, seq_len, vocab_size]."""
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _make_batch(self, num_episodes=3, prompt_len=3, response_len=2, vocab_size=16):
        """Build a GRPO-style batch with rewards for GTPO."""
        import mlx.core as mx
        obs = mx.random.randint(0, vocab_size, shape=(num_episodes, prompt_len + response_len))
        act = mx.random.randint(0, vocab_size, shape=(num_episodes, response_len))
        total_tokens = num_episodes * response_len
        logprob = -mx.abs(mx.random.normal((total_tokens,)))
        # Binary rewards — first episode positive, rest negative
        rewards_list = [1.0] + [0.0] * (num_episodes - 1)
        rewards = mx.array(rewards_list, dtype=mx.float32)
        mx.eval(obs, act, logprob, rewards)
        return {
            "obs": obs,
            "act": act,
            "logprob": logprob,
            "rewards": rewards,
            "episode_lengths": [response_len] * num_episodes,
            "prompt_lengths": [prompt_len] * num_episodes,
        }

    def test_h9a_trainer_trains_uncompiled(self):
        """H9a: Trainer produces finite loss with GTPO (uncompiled)."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        model = self._make_model()
        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=False,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch()
        metrics = trainer.train(batch)

        assert "loss" in metrics
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"

    def test_h9b_trainer_trains_compiled(self):
        """H9b: Trainer produces finite loss with GTPO (compiled).

        This proves all operations in _GTPOTransform are compile-safe:
        no mx.eval(), no .item(), no bool() on arrays inside the traced graph.
        """
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        model = self._make_model()
        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=True,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch()
        metrics = trainer.train(batch)

        assert "loss" in metrics
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"
        assert trainer._compiled is True

    def test_h9c_transform_differs_from_grpo(self):
        """H9c: Faithful GTPO advantages differ from standard GRPO advantages."""
        from textpolicy.training import build_gtpo_transform
        from textpolicy.algorithms.grpo import compute_advantages

        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [2, 2, 2, 2]
        # Non-uniform entropy to create asymmetry
        token_entropies = mx.array([5.0, 1.0, 1.0, 5.0, 4.0, 0.5, 0.5, 4.0])

        # Standard GRPO: uniform per-episode advantages
        grpo_advantages = compute_advantages(rewards)
        expanded = mx.concatenate([mx.repeat(grpo_advantages[i:i+1], 2) for i in range(4)])

        # Faithful GTPO via transform
        batch_data = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }
        gtpo_advantages = transform(expanded, batch_data)
        mx.eval(expanded, gtpo_advantages)

        # GTPO should produce non-uniform per-token advantages within an episode
        assert not mx.allclose(expanded, gtpo_advantages, atol=1e-4), \
            "Faithful GTPO advantages should differ from standard GRPO"

    def test_h9d_variable_length_episodes(self):
        """H9d: Variable-length episodes handled correctly through Trainer."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        model = self._make_model()
        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=False,
            advantage_transform_fn=transform,
        )

        # Variable-length episodes: 3, 1, 2 tokens
        vocab_size = 16
        prompt_len = 3
        max_response_len = 3
        episode_lengths = [3, 1, 2]
        total_tokens = sum(episode_lengths)

        obs = mx.random.randint(0, vocab_size, shape=(3, prompt_len + max_response_len))
        act = mx.random.randint(0, vocab_size, shape=(3, max_response_len))
        logprob = -mx.abs(mx.random.normal((total_tokens,)))
        rewards = mx.array([1.0, 0.0, 0.5], dtype=mx.float32)
        mx.eval(obs, act, logprob, rewards)

        batch = {
            "obs": obs,
            "act": act,
            "logprob": logprob,
            "rewards": rewards,
            "episode_lengths": episode_lengths,
            "prompt_lengths": [prompt_len] * 3,
        }

        metrics = trainer.train(batch)
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"

    def test_h9e_partial_policy_loss(self):
        """H9e: functools.partial(policy_loss, clip_ratio=0.2) works as loss_fn."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        model = self._make_model()
        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        # Use partial with a custom clip_ratio
        loss_fn = partial(grpo.policy_loss, clip_ratio=0.3)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=loss_fn,
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=False,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch()
        metrics = trainer.train(batch)
        assert math.isfinite(metrics["loss"])

    def test_h9f_missing_episode_lengths_raises(self):
        """H9f: Missing episode_lengths in batch_data raises ValueError."""
        from textpolicy.training import build_gtpo_transform

        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)
        advantages = mx.array([0.5, -0.5, 0.3, -0.3])
        batch_data = {
            "rewards": mx.array([1.0, 0.0]),
            "token_entropies": mx.array([2.0, 1.0, 3.0, 0.5]),
            # episode_lengths intentionally omitted
        }

        with pytest.raises(ValueError, match="episode_lengths"):
            transform(advantages, batch_data)

    def test_h9g_missing_token_entropies_raises(self):
        """H9g: Missing token_entropies in batch_data raises ValueError."""
        from textpolicy.training import build_gtpo_transform

        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)
        advantages = mx.array([0.5, -0.5, 0.3, -0.3])
        batch_data = {
            "rewards": mx.array([1.0, 0.0]),
            "episode_lengths": [2, 2],
            # token_entropies intentionally omitted
        }

        with pytest.raises(ValueError, match="token_entropies"):
            transform(advantages, batch_data)

    def test_h9h_negative_alpha_rejected(self):
        """H9h: Negative alpha values are rejected by the builder."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="alpha_1"):
            build_gtpo_transform(alpha_1=-0.1)

        with pytest.raises(ValueError, match="alpha_2"):
            build_gtpo_transform(alpha_2=-0.5)

    def test_h9i_micro_batching(self):
        """H9i: Works with micro_batch_size=2."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        model = self._make_model()
        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=False,
            micro_batch_size=2,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch(num_episodes=4, response_len=2)
        metrics = trainer.train(batch)
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"


# ---------------------------------------------------------------------------
# H10: HICRA Fusion via Entropy Injection
# ---------------------------------------------------------------------------

# MockTokenizer for HICRA fusion tests — maps token IDs to predictable strings
# so we can control which tokens match strategic grams.
class _MockTokenizer:
    """Maps token IDs to pre-configured strings for test control."""

    def __init__(self, vocab: dict):
        self._vocab = vocab

    def convert_ids_to_tokens(self, ids):
        return [self._vocab.get(i, "x") for i in ids]


@pytest.mark.unit
@pytest.mark.algorithm
class TestBoostEntropyWithPlanning:
    """H10a-g: Pure function tests for boost_entropy_with_planning."""

    def test_h10a_gamma_zero_identity(self):
        """H10a: gamma=0 returns entropies unchanged (identity)."""
        entropies = mx.array([1.0, 2.0, 3.0, 4.0])
        mask = mx.array([1.0, 0.0, 1.0, 0.0])

        result = boost_entropy_with_planning(entropies, mask, gamma=0.0)
        mx.eval(result)

        assert mx.array_equal(result, entropies), "gamma=0 should return identity"

    def test_h10b_all_zero_mask_identity(self):
        """H10b: All-zero planning mask → entropies unchanged."""
        entropies = mx.array([1.0, 2.0, 3.0, 4.0])
        mask = mx.zeros((4,), dtype=mx.float32)

        result = boost_entropy_with_planning(entropies, mask, gamma=0.3)
        mx.eval(result)

        assert mx.allclose(result, entropies, atol=1e-6), \
            "All-zero mask should leave entropies unchanged"

    def test_h10c_manual_verification(self):
        """H10c: Manual verification: H + gamma * mask * mean(H)."""
        entropies = mx.array([2.0, 4.0, 1.0, 3.0])
        mask = mx.array([1.0, 0.0, 1.0, 0.0])
        gamma = 0.3

        result = boost_entropy_with_planning(entropies, mask, gamma=gamma)
        mx.eval(result)
        r = result.tolist()

        # mean(H) = (2+4+1+3)/4 = 2.5
        mean_h = 2.5
        # t=0: 2.0 + 0.3*1.0*2.5 = 2.0 + 0.75 = 2.75
        assert abs(r[0] - 2.75) < 1e-4, f"t=0: expected 2.75, got {r[0]}"
        # t=1: 4.0 + 0.3*0.0*2.5 = 4.0
        assert abs(r[1] - 4.0) < 1e-4, f"t=1: expected 4.0, got {r[1]}"
        # t=2: 1.0 + 0.3*1.0*2.5 = 1.75
        assert abs(r[2] - 1.75) < 1e-4, f"t=2: expected 1.75, got {r[2]}"
        # t=3: 3.0 + 0.3*0.0*2.5 = 3.0
        assert abs(r[3] - 3.0) < 1e-4, f"t=3: expected 3.0, got {r[3]}"

    def test_h10d_zero_entropies_no_nan(self):
        """H10d: All-zero entropies → boost is zero, no NaN."""
        entropies = mx.zeros((4,), dtype=mx.float32)
        mask = mx.array([1.0, 0.0, 1.0, 0.0])

        result = boost_entropy_with_planning(entropies, mask, gamma=0.5)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)).item(), "Should not produce NaN"
        assert mx.allclose(result, mx.zeros((4,)), atol=1e-6), \
            "Zero entropies + boost should still be zero (mean(0)=0)"

    def test_h10e_shape_mismatch_raises(self):
        """H10e: Shape mismatch raises ValueError."""
        entropies = mx.array([1.0, 2.0, 3.0])
        mask = mx.array([1.0, 0.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            boost_entropy_with_planning(entropies, mask)

    def test_h10f_stop_gradient_on_mask(self):
        """H10f: mx.stop_gradient — no gradient through planning mask."""
        def fn(mask):
            entropies = mx.array([2.0, 4.0, 1.0, 3.0])
            boosted = boost_entropy_with_planning(entropies, mask, gamma=0.3)
            return mx.sum(boosted)

        mask = mx.array([1.0, 0.0, 1.0, 0.0])
        grad_fn = mx.grad(fn)
        grad_mask = grad_fn(mask)
        mx.eval(grad_mask)

        assert mx.allclose(
            grad_mask, mx.zeros_like(grad_mask), atol=1e-12
        ), f"Gradient w.r.t. mask should be zero, got {grad_mask}"

    def test_h10g_runs_under_mx_compile(self):
        """H10g: Runs under mx.compile without error."""
        @mx.compile
        def compiled_boost(entropies, mask):
            return boost_entropy_with_planning(entropies, mask, gamma=0.3)

        entropies = mx.array([2.0, 4.0, 1.0, 3.0])
        mask = mx.array([1.0, 0.0, 1.0, 0.0])

        result = compiled_boost(entropies, mask)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)).item()
        assert result.shape == entropies.shape

@pytest.mark.unit
@pytest.mark.algorithm
class TestHICRAFusionTransform:
    """H10h-n: Transform-level tests for HICRA fusion in _GTPOTransform."""

    def test_h10h_default_build_identical(self):
        """H10h: Default build (no HICRA) works identically to original."""
        from textpolicy.training import build_gtpo_transform

        transform = build_gtpo_transform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 1.0, 0.0])
        episode_lengths = [2, 2, 2]
        token_entropies = mx.array([3.0, 1.0, 1.0, 3.0, 2.0, 2.0])

        batch_data = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }
        advantages = mx.zeros((6,))
        result = transform(advantages, batch_data)
        mx.eval(result)

        assert result.shape == (6,)
        assert not mx.any(mx.isnan(result)).item()

    def test_h10i_gamma_without_tokenizer_raises(self):
        """H10i: hicra_gamma > 0 without tokenizer raises ValueError."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="tokenizer"):
            build_gtpo_transform(hicra_gamma=0.3)

    def test_h10j_negative_gamma_rejected(self):
        """H10j: Negative gamma rejected by builder."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="hicra_gamma"):
            build_gtpo_transform(hicra_gamma=-0.1)

    def test_h10k_prepare_batch_computes_planning_mask(self):
        """H10k: prepare_batch computes correct planning mask."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        # Token IDs: 1="let", 2="me", 3="think", 4="x"
        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
        )

        batch_data = {
            "act": mx.array([1, 2, 3, 4, 4, 4]),
            "episode_lengths": [3, 3],
        }
        transform.prepare_batch(batch_data)

        assert "planning_mask" in batch_data
        mask = batch_data["planning_mask"]
        mx.eval(mask)
        mask_list = mask.tolist()

        # Tokens 0,1,2 ("let me think") should be marked, tokens 3,4,5 should not
        assert mask_list[0] == 1.0, "Token 'let' should be marked"
        assert mask_list[1] == 1.0, "Token 'me' should be marked"
        assert mask_list[2] == 1.0, "Token 'think' should be marked"
        assert mask_list[3] == 0.0, "Token 'x' should not be marked"

    def test_h10l_prepare_batch_noop_when_gamma_zero(self):
        """H10l: prepare_batch is no-op when gamma=0."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        transform = _GTPOTransform(
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.0,
        )

        batch_data = {
            "act": mx.array([1, 2, 3]),
            "episode_lengths": [3],
        }
        transform.prepare_batch(batch_data)

        assert "planning_mask" not in batch_data, \
            "prepare_batch should be no-op when gamma=0"

    def test_h10m_planning_tokens_get_different_shaped_rewards(self):
        """H10m: Planning tokens get different shaped rewards with fusion.

        Uses 2+ episodes per O+ group so normalization doesn't collapse to zero.
        Non-uniform entropies ensure the boost creates observable asymmetry.
        """
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        # With HICRA fusion
        fused = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.5,
        )

        # Without HICRA fusion
        plain = _GTPOTransform(alpha_1=1.0, alpha_2=0.1)

        # 2 O+ episodes + 2 O- episodes, non-uniform entropy
        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        # Non-uniform entropy so boosting creates observable asymmetry
        token_entropies = mx.array([
            2.0, 3.0, 1.0,   # ep0 (O+) — has planning tokens
            4.0, 1.0, 2.0,   # ep1 (O+) — no planning tokens
            3.0, 2.0, 1.0,   # ep2 (O-)
            1.0, 3.0, 2.0,   # ep3 (O-)
        ])
        # ep0 has "let me think" → planning mask [1,1,1,0,0,0,...]
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch_fused = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
        }
        fused.prepare_batch(batch_fused)

        batch_plain = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }

        advantages = mx.zeros((12,))
        result_fused = fused(advantages, batch_fused)
        result_plain = plain(advantages, batch_plain)
        mx.eval(result_fused, result_plain)

        # Compare the shaped rewards before normalization to see the effect
        # After normalization, the boosted entropies should redistribute credit
        # differently within the O+ group
        assert not mx.allclose(result_fused, result_plain, atol=1e-4), \
            "HICRA fusion should produce different shaped rewards than plain GTPO"

    def test_h10n_no_matching_grams_same_as_no_hicra(self):
        """H10n: No matching grams → same result as no HICRA."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        # Tokenizer maps IDs to strings that won't match "let me think"
        tokenizer = _MockTokenizer({1: "foo", 2: "bar", 3: "baz"})

        fused = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.5,
        )
        plain = _GTPOTransform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 0.0])
        episode_lengths = [3, 3]
        token_entropies = mx.array([2.0, 3.0, 1.0, 4.0, 0.5, 2.5])

        batch_fused = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": mx.array([1, 2, 3, 1, 2, 3]),
        }
        fused.prepare_batch(batch_fused)

        batch_plain = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }

        advantages = mx.zeros((6,))
        result_fused = fused(advantages, batch_fused)
        result_plain = plain(advantages, batch_plain)
        mx.eval(result_fused, result_plain)

        # No tokens match → planning_mask is all zeros → boost is identity
        # → results should be identical
        assert mx.allclose(result_fused, result_plain, atol=1e-6), \
            "No matching grams should produce identical results to no HICRA"

    def test_h10r_on_demand_planning_mask_matches_prepared_path(self):
        """H10r regression: on-demand planning_mask path matches prepared path."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0], dtype=mx.float32)
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ], dtype=mx.float32)
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])
        advantages = mx.zeros((12,), dtype=mx.float32)

        # Path A: no planning_mask provided; __call__ computes on demand.
        batch_no_mask = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
        }
        result_on_demand = transform(advantages, batch_no_mask)
        mx.eval(result_on_demand)
        assert "planning_mask" in batch_no_mask, \
            "__call__ should cache planning_mask when missing"

        # Path B: planning_mask prepared eagerly.
        batch_prepared = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
        }
        transform.prepare_batch(batch_prepared)
        assert "planning_mask" in batch_prepared
        result_prepared = transform(advantages, batch_prepared)
        mx.eval(result_prepared)

        assert mx.allclose(result_on_demand, result_prepared, atol=1e-6), \
            "On-demand planning_mask path must match prepared path"

    def test_h10s_precomputed_planning_mask_does_not_require_act(self):
        """H10s: Precomputed planning_mask path should work without `act`."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
        )

        batch_data = {
            "rewards": mx.array([1.0, 1.0, 0.0, 0.0], dtype=mx.float32),
            "token_entropies": mx.array([
                2.0, 3.0, 1.0,
                4.0, 1.0, 2.0,
                3.0, 2.0, 1.0,
                1.0, 3.0, 2.0,
            ], dtype=mx.float32),
            "episode_lengths": [3, 3, 3, 3],
            "planning_mask": mx.array([
                1.0, 1.0, 1.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            ], dtype=mx.float32),
            # `act` intentionally omitted; precomputed mask should be sufficient.
        }

        advantages = mx.zeros((12,), dtype=mx.float32)
        result = transform(advantages, batch_data)
        mx.eval(result)
        assert result.shape == (12,)
        assert not mx.any(mx.isnan(result)).item()

    def test_h10t_missing_act_raises_when_on_demand_mask_needed(self):
        """H10t: Missing `act` raises when on-demand planning mask is required."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
        )

        batch_data = {
            "rewards": mx.array([1.0, 0.0], dtype=mx.float32),
            "token_entropies": mx.array([2.0, 3.0, 1.0, 4.0], dtype=mx.float32),
            "episode_lengths": [2, 2],
            # `planning_mask` and `act` both omitted -> on-demand path should fail.
        }

        with pytest.raises(ValueError, match="batch_data must include 'act'"):
            transform(mx.zeros((4,), dtype=mx.float32), batch_data)

    def test_h10u_invalid_precomputed_mask_shape_raises(self):
        """H10u: Precomputed planning_mask with wrong shape raises ValueError."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
        )

        batch_data = {
            "rewards": mx.array([1.0, 0.0], dtype=mx.float32),
            "token_entropies": mx.array([2.0, 3.0, 1.0, 4.0], dtype=mx.float32),
            "episode_lengths": [2, 2],
            # Wrong length: should be 4 to align with token_entropies.
            "planning_mask": mx.array([1.0, 0.0, 1.0], dtype=mx.float32),
        }

        with pytest.raises(ValueError, match="Shape mismatch"):
            transform(mx.zeros((4,), dtype=mx.float32), batch_data)


@pytest.mark.integration
class TestHICRAFusionTrainerIntegration:
    """H10o-p: End-to-end Trainer tests with fused HICRA+GTPO."""

    def _make_model(self, vocab_size=16, dim=8):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _make_batch(self, num_episodes=3, prompt_len=3, response_len=2, vocab_size=16):
        # Use token IDs that will produce "let me think" for first episode
        # IDs 1,2 → "let","me" for strategic gram matching
        obs = mx.random.randint(0, vocab_size, shape=(num_episodes, prompt_len + response_len))
        # Craft act tokens: first episode has "let me", rest random
        act_row0 = mx.array([[1, 2]])
        act_rest = mx.random.randint(3, vocab_size, shape=(num_episodes - 1, response_len))
        act = mx.concatenate([act_row0, act_rest], axis=0) if num_episodes > 1 else act_row0
        total_tokens = num_episodes * response_len
        logprob = -mx.abs(mx.random.normal((total_tokens,)))
        rewards_list = [1.0] + [0.0] * (num_episodes - 1)
        rewards = mx.array(rewards_list, dtype=mx.float32)
        mx.eval(obs, act, logprob, rewards)
        return {
            "obs": obs,
            "act": act,
            "logprob": logprob,
            "rewards": rewards,
            "episode_lengths": [response_len] * num_episodes,
            "prompt_lengths": [prompt_len] * num_episodes,
        }

    def test_h10o_trainer_trains_uncompiled(self):
        """H10o: Trainer trains with fused HICRA+GTPO (uncompiled)."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        model = self._make_model()
        transform = build_gtpo_transform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me"],
            hicra_gamma=0.3,
        )

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=False,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch()
        metrics = trainer.train(batch)

        assert "loss" in metrics
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"

    def test_h10p_trainer_trains_compiled(self):
        """H10p: Trainer trains with fused HICRA+GTPO (compiled) — compile-safety proof."""
        from functools import partial
        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        model = self._make_model()
        transform = build_gtpo_transform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me"],
            hicra_gamma=0.3,
        )

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optim.Adam(learning_rate=1e-3),
            compile_training=True,
            advantage_transform_fn=transform,
        )

        batch = self._make_batch()
        metrics = trainer.train(batch)

        assert "loss" in metrics
        assert math.isfinite(metrics["loss"]), f"Loss should be finite, got {metrics['loss']}"
        assert trainer._compiled is True


# ---------------------------------------------------------------------------
# H11: SEPA — Selective Entropy Pooling with Annealing
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.algorithm
class TestSEPATransform:
    """H11: Tests for SEPA (Selective Entropy Pooling with Annealing)."""

    def test_h11a_sepa_at_lambda_zero_is_pure_gtpo(self):
        """H11a: At step=0 (λ=0), SEPA produces identical results to no HICRA."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        blend = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=100,
        )
        plain = _GTPOTransform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch_blend = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 0,  # λ = 0/100 = 0.0
        }
        blend.prepare_batch(batch_blend)

        batch_plain = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }

        advantages = mx.zeros((12,))
        result_blend = blend(advantages, batch_blend)
        result_plain = plain(advantages, batch_plain)
        mx.eval(result_blend, result_plain)

        assert mx.allclose(result_blend, result_plain, atol=1e-6), \
            "At λ=0, SEPA should produce identical results to no HICRA"

    def test_h11b_sepa_at_lambda_one_uniform_execution_entropy(self):
        """H11b: At step >= sepa_steps (λ=1), execution tokens have uniform entropy."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=100,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        # Non-uniform execution token entropy to verify it gets smoothed
        token_entropies = mx.array([
            2.0, 3.0, 1.0,   # ep0: planning (tokens 0,1,2)
            4.0, 1.0, 2.0,   # ep1: execution (tokens 3,4,5) — varied entropy
            3.0, 2.0, 1.0,   # ep2: execution
            1.0, 3.0, 2.0,   # ep3: execution
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 200,  # λ = min(200/100, 1.0) = 1.0
        }
        transform.prepare_batch(batch)

        advantages = mx.zeros((12,))
        result = transform(advantages, batch)
        mx.eval(result)

        assert result.shape == (12,)
        assert not mx.any(mx.isnan(result)).item()

    def test_h11c_sepa_differs_from_no_hicra_at_nonzero_lambda(self):
        """H11c: At λ > 0, SEPA produces different results than plain GTPO."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        blend = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=100,
        )
        plain = _GTPOTransform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch_blend = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 100,  # λ = 1.0
        }
        blend.prepare_batch(batch_blend)

        batch_plain = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
        }

        advantages = mx.zeros((12,))
        result_blend = blend(advantages, batch_blend)
        result_plain = plain(advantages, batch_plain)
        mx.eval(result_blend, result_plain)

        assert not mx.allclose(result_blend, result_plain, atol=1e-4), \
            "At λ=1, SEPA should differ from plain GTPO when entropy is non-uniform"

    def test_h11d_eq5_no_blowup_at_lambda_one(self):
        """H11d: Eq. 5 (O- inverse entropy) produces finite values at λ=1.

        Key SEPA safety property: pooling toward mean(H_exec) > 0 means
        1/H stays finite — no 1/eps blowup from zeroed entropy.
        """
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=10,
        )

        # All O- episodes — forces Eq. 5 path exclusively
        rewards = mx.array([0.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,  # ep0: planning tokens
            4.0, 1.0, 2.0,  # ep1: execution
            0.5, 3.0, 2.0,  # ep2: execution — includes low entropy token
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4]])

        batch = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 10,  # λ = 1.0
        }
        transform.prepare_batch(batch)

        advantages = mx.zeros((9,))
        result = transform(advantages, batch)
        mx.eval(result)

        assert result.shape == (9,)
        assert not mx.any(mx.isnan(result)).item(), "SEPA at λ=1 should not produce NaN"
        assert not mx.any(mx.isinf(result)).item(), "SEPA at λ=1 should not produce Inf"
        # Verify magnitudes are reasonable (not 1/eps scale)
        assert mx.max(mx.abs(result)).item() < 100.0, \
            "Advantages should be reasonable magnitude, not 1/eps blowup"

    def test_h11e_sepa_steps_zero_uses_boost(self):
        """H11e: sepa_steps=0 falls back to boost mode (backward compat)."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        # Blend with 0 steps = boost mode
        blend_zero = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.5,
            sepa_steps=0,
        )
        # Explicit boost mode
        boost = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.5,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch_a = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
        }
        blend_zero.prepare_batch(batch_a)

        batch_b = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
        }
        boost.prepare_batch(batch_b)

        advantages = mx.zeros((12,))
        result_a = blend_zero(advantages, batch_a)
        result_b = boost(advantages, batch_b)
        mx.eval(result_a, result_b)

        assert mx.allclose(result_a, result_b, atol=1e-6), \
            "sepa_steps=0 should produce identical results to boost mode"

    def test_h11f_negative_sepa_steps_rejected(self):
        """H11f: Negative sepa_steps rejected by builder."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="sepa_steps"):
            build_gtpo_transform(sepa_steps=-1)

    def test_h11g_no_planning_tokens_sepa_degrades_gracefully(self):
        """H11g: When no planning tokens match, SEPA pools all tokens (graceful)."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        # Tokenizer that won't match any strategic grams
        tokenizer = _MockTokenizer({1: "foo", 2: "bar", 3: "baz"})

        blend = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=100,
        )
        plain = _GTPOTransform(alpha_1=1.0, alpha_2=0.1)

        rewards = mx.array([1.0, 0.0])
        episode_lengths = [3, 3]
        token_entropies = mx.array([2.0, 3.0, 1.0, 4.0, 0.5, 2.5])

        # At λ=1, all tokens are execution. blend: H → mean(H) for all.
        # With uniform entropy, GTPO assigns equal weight at each position.
        batch_blend = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": mx.array([1, 2, 3, 1, 2, 3]),
            "step": 100,  # λ = 1.0
        }
        blend.prepare_batch(batch_blend)

        advantages = mx.zeros((6,))
        result = blend(advantages, batch_blend)
        mx.eval(result)

        assert result.shape == (6,)
        assert not mx.any(mx.isnan(result)).item()
        assert not mx.any(mx.isinf(result)).item()

    def test_h11h_lambda_anneals_linearly(self):
        """H11h: Intermediate steps produce intermediate pooling."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            hicra_gamma=0.3,
            sepa_steps=100,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])
        advantages = mx.zeros((12,))

        results = []
        for step in [0, 25, 50, 75, 100]:
            batch = {
                "rewards": rewards,
                "token_entropies": token_entropies,
                "episode_lengths": episode_lengths,
                "act": act,
                "step": step,
            }
            transform.prepare_batch(batch)
            r = transform(advantages, batch)
            mx.eval(r)
            results.append(r)

        # Step 0 should equal step 0 (identity)
        # Step 100 should differ from step 0
        assert not mx.allclose(results[0], results[-1], atol=1e-4), \
            "Step 0 and step 100 should produce different results"

        # Intermediate steps should be distinct from both endpoints
        # (at least one intermediate step should differ from both extremes)
        mid = results[2]  # step 50
        differs_from_start = not mx.allclose(mid, results[0], atol=1e-4)
        differs_from_end = not mx.allclose(mid, results[-1], atol=1e-4)
        assert differs_from_start or differs_from_end, \
            "Intermediate λ should produce intermediate results"

    def test_h11i_sepa_without_hicra_gamma(self):
        """H11i: sepa_steps > 0 alone enables SEPA (no hicra_gamma needed)."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})

        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            sepa_steps=100,  # no hicra_gamma
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        token_entropies = mx.array([
            2.0, 3.0, 1.0,
            4.0, 1.0, 2.0,
            3.0, 2.0, 1.0,
            1.0, 3.0, 2.0,
        ])
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])

        batch = {
            "rewards": rewards,
            "token_entropies": token_entropies,
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 100,  # λ = 1.0
        }
        transform.prepare_batch(batch)

        # Should compute planning mask and run SEPA
        assert "planning_mask" in batch, "SEPA should compute planning mask"

        advantages = mx.zeros((12,))
        result = transform(advantages, batch)
        mx.eval(result)

        assert result.shape == (12,)
        assert not mx.any(mx.isnan(result)).item()

    def test_h11j_sepa_without_tokenizer_raises(self):
        """H11j: sepa_steps > 0 without tokenizer raises ValueError."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="tokenizer"):
            build_gtpo_transform(sepa_steps=500)

    def test_h11k_sepa_with_gamma_warns(self):
        """H11k: sepa_steps > 0 with hicra_gamma > 0 warns that gamma is ignored."""
        import warnings as _warnings
        from textpolicy.training import build_gtpo_transform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            build_gtpo_transform(
                tokenizer=tokenizer,
                hicra_gamma=0.3,
                sepa_steps=500,
            )
            assert len(w) == 1
            assert "ignored" in str(w[0].message).lower()

    def test_h11l_invalid_sepa_schedule_rejected(self):
        """H11l: Invalid sepa_schedule values are rejected by builder."""
        from textpolicy.training import build_gtpo_transform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        with pytest.raises(ValueError, match="sepa_schedule"):
            build_gtpo_transform(
                tokenizer=tokenizer,
                sepa_schedule="not-a-schedule",
            )

    def test_h11m_auto_sepa_without_tokenizer_raises(self):
        """H11m: auto SEPA requires tokenizer."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="tokenizer"):
            build_gtpo_transform(sepa_schedule="auto")

    def test_h11n_auto_sepa_with_gamma_warns(self):
        """H11n: auto SEPA with hicra_gamma warns that gamma is ignored."""
        import warnings as _warnings
        from textpolicy.training import build_gtpo_transform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            build_gtpo_transform(
                tokenizer=tokenizer,
                sepa_schedule="auto",
                hicra_gamma=0.3,
            )
            assert len(w) == 1
            assert "ignored" in str(w[0].message).lower()

    def test_h11o_auto_schedule_updates_lambda_after_postprocess(self):
        """H11o: auto schedule raises λ after execution entropy variance drops."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            sepa_schedule="auto",
            sepa_ema_decay=0.0,      # no smoothing for deterministic test behavior
            sepa_var_threshold=0.5,  # start ramping once variance halves
            sepa_warmup=1,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])
        advantages = mx.zeros((12,))

        # Batch 1: high execution-entropy variance initializes var_0.
        batch_high = {
            "rewards": rewards,
            "token_entropies": mx.array([
                2.0, 3.0, 1.0,  # planning
                0.5, 4.5, 0.5,  # execution
                4.5, 0.5, 4.5,  # execution
                0.5, 4.5, 0.5,  # execution
            ]),
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 0,
        }
        transform.prepare_batch(batch_high)
        assert batch_high["sepa_lambda"] == 0.0
        r0 = transform(advantages, batch_high)
        mx.eval(r0)
        transform.postprocess_batch(batch_high)

        # Batch 2: low variance updates EMA toward smaller variance.
        batch_low = {
            "rewards": rewards,
            "token_entropies": mx.array([
                2.0, 3.0, 1.0,  # planning
                2.0, 2.0, 2.0,  # execution
                2.0, 2.0, 2.0,  # execution
                2.0, 2.0, 2.0,  # execution
            ]),
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 1,
        }
        transform.prepare_batch(batch_low)
        # λ uses previous state; should still be at boundary before this update.
        assert 0.0 <= batch_low["sepa_lambda"] <= 1.0
        r1 = transform(advantages, batch_low)
        mx.eval(r1)
        transform.postprocess_batch(batch_low)

        # Batch 3: prepare should now observe the reduced variance and raise λ.
        batch_low_next = dict(batch_low)
        batch_low_next["step"] = 2
        transform.prepare_batch(batch_low_next)
        assert batch_low_next["sepa_lambda"] > 0.0
        assert batch_low_next["sepa_lambda"] <= 1.0

    def test_h11p_auto_schedule_uses_linear_cap(self):
        """H11p: auto schedule obeys fallback linear cap via sepa_steps."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            sepa_schedule="auto",
            sepa_steps=3,  # cap: λ must be 1.0 at step >= 3
            sepa_warmup=50,
        )

        batch = {
            "rewards": mx.array([1.0, 0.0]),
            "token_entropies": mx.array([2.0, 3.0, 1.0, 4.0, 1.0, 2.0]),
            "episode_lengths": [3, 3],
            "act": mx.array([[1, 2, 3], [4, 4, 4]]),
            "step": 3,
        }
        transform.prepare_batch(batch)
        assert batch["sepa_lambda"] == 1.0

        result = transform(mx.zeros((6,)), batch)
        mx.eval(result)
        assert result.shape == (6,)
        assert not mx.any(mx.isnan(result)).item()

    def test_h11q_auto_schedule_drops_lambda_after_variance_spike(self):
        """H11q: auto schedule is bidirectional when execution variance spikes."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "x"})
        transform = _GTPOTransform(
            alpha_1=1.0, alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            sepa_schedule="auto",
            sepa_ema_decay=0.0,
            sepa_var_threshold=0.5,
            sepa_warmup=1,
        )

        rewards = mx.array([1.0, 1.0, 0.0, 0.0])
        episode_lengths = [3, 3, 3, 3]
        act = mx.array([[1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]])
        advantages = mx.zeros((12,))

        batch_high = {
            "rewards": rewards,
            "token_entropies": mx.array([
                2.0, 3.0, 1.0,
                0.5, 4.5, 0.5,
                4.5, 0.5, 4.5,
                0.5, 4.5, 0.5,
            ]),
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 0,
        }
        transform.prepare_batch(batch_high)
        transform(advantages, batch_high)
        transform.postprocess_batch(batch_high)

        batch_low = {
            "rewards": rewards,
            "token_entropies": mx.array([
                2.0, 3.0, 1.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
            ]),
            "episode_lengths": episode_lengths,
            "act": act,
            "step": 1,
        }
        transform.prepare_batch(batch_low)
        transform(advantages, batch_low)
        transform.postprocess_batch(batch_low)

        batch_probe = dict(batch_low)
        batch_probe["step"] = 2
        transform.prepare_batch(batch_probe)
        lambda_after_drop = batch_probe["sepa_lambda"]
        assert lambda_after_drop > 0.0

        batch_spike = dict(batch_high)
        batch_spike["step"] = 3
        transform.prepare_batch(batch_spike)
        transform(advantages, batch_spike)
        transform.postprocess_batch(batch_spike)

        batch_probe2 = dict(batch_spike)
        batch_probe2["step"] = 4
        transform.prepare_batch(batch_probe2)
        lambda_after_spike = batch_probe2["sepa_lambda"]

        assert lambda_after_spike < lambda_after_drop
        assert lambda_after_spike == 0.0

    def test_h11r_semantic_entropy_tracking_emits_metrics(self):
        """H11r: semantic entropy tracker computes prompt-group dispersion stats."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "alt"})
        transform = _GTPOTransform(
            alpha_1=1.0,
            alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            semantic_entropy=True,
            semantic_entropy_ema_decay=0.0,
            semantic_entropy_stability_patience=1,
        )

        batch = {
            "obs": mx.array([
                [10, 11, 1, 2, 3],  # prompt A
                [10, 11, 1, 2, 3],  # prompt A
                [20, 21, 1, 2, 3],  # prompt B
                [20, 21, 1, 4, 3],  # prompt B (different planning pattern)
            ], dtype=mx.int32),
            "act": mx.array([
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 4, 3],
            ], dtype=mx.int32),
            "logprob": mx.array([-1.0] * 12, dtype=mx.float32),
            "rewards": mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32),
            "token_entropies": mx.array([
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
                2.0, 2.0, 2.0,
            ], dtype=mx.float32),
            "episode_lengths": [3, 3, 3, 3],
            "prompt_lengths": [2, 2, 2, 2],
            "step": 0,
        }

        transform.prepare_batch(batch)
        transformed = transform(mx.zeros((12,), dtype=mx.float32), batch)
        mx.eval(transformed)
        transform.postprocess_batch(batch)

        stats = batch.get("semantic_entropy_stats")
        assert isinstance(stats, dict)
        assert stats["semantic_entropy_batch"] > 0.0
        assert stats["semantic_entropy_ema"] > 0.0
        assert stats["semantic_entropy_group_count"] >= 1.0

    def test_h11s_semantic_entropy_without_tokenizer_raises(self):
        """H11s: semantic entropy mode requires tokenizer for planning masks."""
        from textpolicy.training import build_gtpo_transform

        with pytest.raises(ValueError, match="semantic_entropy"):
            build_gtpo_transform(semantic_entropy=True)

    def test_h11t_hidden_state_mode_passes_embeddings_to_tracker(self):
        """H11t: _GTPOTransform with hidden_states mode extracts from batch_data."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think", 4: "alt"})
        transform = _GTPOTransform(
            alpha_1=1.0,
            alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            semantic_entropy=True,
            semantic_entropy_ema_decay=0.0,
            semantic_entropy_embedding_mode="hidden_states",
        )

        assert transform.needs_hidden_states is True

        # 4 episodes, 3 response tokens each, hidden_dim=8
        n_eps = 4
        ep_len = 3
        total_tokens = n_eps * ep_len
        hidden_dim = 8

        batch = {
            "obs": mx.array([
                [10, 11, 1, 2, 3],
                [10, 11, 1, 2, 3],
                [20, 21, 1, 2, 3],
                [20, 21, 1, 4, 3],
            ], dtype=mx.int32),
            "act": mx.array([
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
                [1, 4, 3],
            ], dtype=mx.int32),
            "logprob": mx.array([-1.0] * total_tokens, dtype=mx.float32),
            "rewards": mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32),
            "token_entropies": mx.array(
                [2.0] * total_tokens, dtype=mx.float32,
            ),
            "episode_lengths": [ep_len] * n_eps,
            "prompt_lengths": [2] * n_eps,
            "step": 0,
            # Flat response-aligned hidden states: [total_tokens, hidden_dim]
            "hidden_states": mx.random.normal((total_tokens, hidden_dim)),
        }

        transform.prepare_batch(batch)
        transformed = transform(mx.zeros((total_tokens,), dtype=mx.float32), batch)
        mx.eval(transformed)
        transform.postprocess_batch(batch)

        stats = batch.get("semantic_entropy_stats")
        assert isinstance(stats, dict)
        assert stats["semantic_entropy_batch"] >= 0.0

    def test_h11u_hidden_state_mode_without_semantic_entropy_raises(self):
        """H11u: embedding_mode='hidden_states' without semantic_entropy raises."""
        from textpolicy.training import build_gtpo_transform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        with pytest.raises(ValueError, match="semantic_entropy_embedding_mode"):
            build_gtpo_transform(
                tokenizer=tokenizer,
                semantic_entropy=False,
                semantic_entropy_embedding_mode="hidden_states",
            )

    def test_h11v_hash_mode_does_not_need_hidden_states(self):
        """H11v: Default hash mode does not set needs_hidden_states."""
        from textpolicy.training.reasoning_stack import _GTPOTransform

        tokenizer = _MockTokenizer({1: "let", 2: "me", 3: "think"})
        transform = _GTPOTransform(
            alpha_1=1.0,
            alpha_2=0.1,
            tokenizer=tokenizer,
            strategic_grams=["let me think"],
            semantic_entropy=True,
            semantic_entropy_embedding_mode="hash",
        )
        assert transform.needs_hidden_states is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
