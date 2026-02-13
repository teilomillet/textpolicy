"""
Tests for MaxRL: Maximum Likelihood Reinforcement Learning advantages.

Validates the MaxRL advantage computation which replaces GRPO's
group-relative baseline (subtract mean) with inverse success-rate
reweighting (divide by mean), recovering the 1/p(x) weighting from
maximum likelihood estimation.

Hypotheses tested:
  H1: Degenerate cases — uniform/zero rewards produce zero advantages
  H2: Binary reward correctness — exact 1/p reweighting emerges
  H3: Hard prompts get exponentially more gradient weight than easy prompts
  H4: Comparison with GRPO — different optimization dynamics
  H5: Numerical stability, interface compatibility, and compiled variant

References:
    MaxRL: Maximum Likelihood Reinforcement Learning.
    Standard RL optimizes only the first-order approximation of the
    MLE objective; MaxRL recovers the full objective via inverse
    success-rate reweighting.
"""

import pytest
import mlx.core as mx

from textpolicy.algorithms import grpo


# ---------------------------------------------------------------------------
# H1: Degenerate cases — MaxRL should produce zero advantages
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLDegenerateCases:
    """Verify that MaxRL produces zero advantages under degenerate conditions."""

    def test_all_same_rewards_produce_zero_advantages(self):
        """H1: When all rollouts have identical reward, (r-mean)/mean = 0 for all."""
        rewards = [1.0, 1.0, 1.0, 1.0]
        advantages = grpo.compute_advantages_maxrl(rewards)

        assert advantages.shape == (4,)
        assert mx.allclose(advantages, mx.zeros(4), atol=1e-5), \
            "Identical rewards should produce zero advantages"

    def test_all_zero_rewards_masked_to_zero(self):
        """H1: When no rollout succeeds (mean=0), all advantages are zeroed.

        With binary rewards, mean=0 means K=0 (no correct rollout).
        There is no positive signal to learn from — MaxRL sets advantages
        to zero rather than producing large negative values.
        """
        rewards = [0.0, 0.0, 0.0, 0.0]
        advantages = grpo.compute_advantages_maxrl(rewards)

        assert advantages.shape == (4,)
        assert mx.allclose(advantages, mx.zeros(4), atol=1e-5), \
            "All-zero rewards (no correct rollout) should produce zero advantages"

    def test_single_episode_zero_advantage(self):
        """H1: Single episode has mean = r, so (r-r)/r = 0."""
        for r in [0.5, 1.0, 0.3]:
            advantages = grpo.compute_advantages_maxrl([r])
            assert mx.allclose(advantages, mx.zeros(1), atol=1e-5), \
                f"Single episode with reward {r} should have zero advantage"

    def test_empty_rewards_empty_output(self):
        """H1: Empty input returns empty array."""
        advantages = grpo.compute_advantages_maxrl([])
        assert advantages.size == 0

    def test_all_ones_binary_produce_zero(self):
        """H1: All correct rollouts → easy problem, nothing to learn.

        K=N, mean=1, A = (1-1)/1 = 0 for all rollouts.
        """
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        advantages = grpo.compute_advantages_maxrl(rewards)

        assert mx.allclose(advantages, mx.zeros(5), atol=1e-5), \
            "All-correct rollouts should produce zero advantages (nothing to learn)"


# ---------------------------------------------------------------------------
# H2: Binary reward correctness — exact 1/p reweighting
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLBinaryRewards:
    """Verify the exact 1/p reweighting for binary {0, 1} rewards."""

    def test_k1_n4_hard_problem(self):
        """H2: K=1 out of N=4. Hard problem → large positive advantage.

        mean = 1/4 = 0.25
        Correct:   (1 - 0.25) / 0.25 = 3.0
        Incorrect: (0 - 0.25) / 0.25 = -1.0
        """
        rewards = [1.0, 0.0, 0.0, 0.0]
        advantages = grpo.compute_advantages_maxrl(rewards)

        mx.eval(advantages)
        assert abs(float(advantages[0]) - 3.0) < 1e-4, \
            f"Correct rollout should get advantage 3.0, got {float(advantages[0])}"
        for i in [1, 2, 3]:
            assert abs(float(advantages[i]) - (-1.0)) < 1e-4, \
                f"Incorrect rollout {i} should get advantage -1.0, got {float(advantages[i])}"

    def test_k3_n4_easy_problem(self):
        """H2: K=3 out of N=4. Easy problem → small positive advantage.

        mean = 3/4 = 0.75
        Correct:   (1 - 0.75) / 0.75 = 1/3 ≈ 0.333
        Incorrect: (0 - 0.75) / 0.75 = -1.0
        """
        rewards = [1.0, 1.0, 1.0, 0.0]
        advantages = grpo.compute_advantages_maxrl(rewards)

        mx.eval(advantages)
        for i in [0, 1, 2]:
            assert abs(float(advantages[i]) - (1.0 / 3.0)) < 1e-4, \
                f"Correct rollout {i} should get advantage 1/3, got {float(advantages[i])}"
        assert abs(float(advantages[3]) - (-1.0)) < 1e-4, \
            f"Incorrect rollout should get advantage -1.0, got {float(advantages[3])}"

    def test_k1_n16_very_hard_problem(self):
        """H2: K=1 out of N=16 → correct rollout gets advantage 15.

        mean = 1/16
        Correct:   (1 - 1/16) / (1/16) = 15
        Incorrect: (0 - 1/16) / (1/16) = -1
        """
        rewards = [1.0] + [0.0] * 15
        advantages = grpo.compute_advantages_maxrl(rewards)

        mx.eval(advantages)
        assert abs(float(advantages[0]) - 15.0) < 1e-3, \
            f"K=1/N=16: correct rollout should get advantage 15, got {float(advantages[0])}"
        for i in range(1, 16):
            assert abs(float(advantages[i]) - (-1.0)) < 1e-3, \
                f"Incorrect rollout {i} should get advantage -1.0"

    def test_incorrect_rollout_always_minus_one(self):
        """H2: For binary rewards, incorrect rollouts ALWAYS get advantage -1.

        Proof: A_incorrect = (0 - K/N) / (K/N) = -1, independent of K, N.
        This is a key structural property of MaxRL with binary rewards.
        """
        for n in [4, 8, 16]:
            for k in [1, 2, n // 2, n - 1]:
                rewards = [1.0] * k + [0.0] * (n - k)
                advantages = grpo.compute_advantages_maxrl(rewards)
                mx.eval(advantages)

                # Check all incorrect rollouts get -1
                for i in range(k, n):
                    assert abs(float(advantages[i]) - (-1.0)) < 1e-3, \
                        f"N={n}, K={k}: incorrect rollout {i} should be -1, " \
                        f"got {float(advantages[i])}"

    def test_correct_rollout_advantage_equals_n_minus_k_over_k(self):
        """H2: For binary rewards, correct rollouts get advantage (N-K)/K.

        This is the exact 1/p reweighting: when p = K/N, the gradient
        contribution scales as 1/p = N/K, centered by subtracting 1.
        """
        test_cases = [
            (4, 1, 3.0),       # (4-1)/1 = 3
            (4, 2, 1.0),       # (4-2)/2 = 1
            (4, 3, 1 / 3),     # (4-3)/3 = 1/3
            (8, 1, 7.0),       # (8-1)/1 = 7
            (8, 4, 1.0),       # (8-4)/4 = 1
            (16, 1, 15.0),     # (16-1)/1 = 15
            (16, 8, 1.0),      # (16-8)/8 = 1
            (16, 15, 1 / 15),  # (16-15)/15 = 1/15
        ]
        for n, k, expected_adv in test_cases:
            rewards = [1.0] * k + [0.0] * (n - k)
            advantages = grpo.compute_advantages_maxrl(rewards)
            mx.eval(advantages)

            actual = float(advantages[0])
            assert abs(actual - expected_adv) < 1e-3, \
                f"N={n}, K={k}: expected advantage {expected_adv:.4f}, got {actual:.4f}"


# ---------------------------------------------------------------------------
# H3: Hard prompts get exponentially more gradient weight
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLGradientConcentration:
    """Verify that MaxRL concentrates gradient on hard, low-success-rate prompts."""

    def test_harder_prompt_larger_positive_advantage(self):
        """H3: Lower success rate → larger positive advantage for correct rollouts.

        MaxRL advantage for correct = (N-K)/K, which is monotonically
        decreasing in K. So K=1 > K=2 > K=4 > K=8 etc.
        """
        n = 16
        prev_advantage = float('inf')
        for k in [1, 2, 4, 8, 15]:
            rewards = [1.0] * k + [0.0] * (n - k)
            advantages = grpo.compute_advantages_maxrl(rewards)
            mx.eval(advantages)

            correct_adv = float(advantages[0])
            assert correct_adv < prev_advantage, \
                f"K={k}: advantage {correct_adv} should be less than previous"
            assert correct_adv > 0, \
                f"K={k}: correct rollout advantage should be positive"
            prev_advantage = correct_adv

    def test_gradient_weight_ratio_matches_success_rate_ratio(self):
        """H3: Advantage ratio between two difficulty levels matches 1/p ratio.

        If prompt A has success rate p_A and prompt B has p_B, then
        the ratio of their correct-rollout advantages is p_B / p_A.

        Proof: A_correct = (1-p)/p, so ratio = [(1-p_A)/p_A] / [(1-p_B)/p_B]
        For p_A << 1 and p_B << 1, this approximates p_B / p_A.
        """
        # Hard prompt: K=1/N=16, p=1/16
        rewards_hard = [1.0] + [0.0] * 15
        adv_hard = grpo.compute_advantages_maxrl(rewards_hard)
        mx.eval(adv_hard)

        # Easy prompt: K=8/N=16, p=1/2
        rewards_easy = [1.0] * 8 + [0.0] * 8
        adv_easy = grpo.compute_advantages_maxrl(rewards_easy)
        mx.eval(adv_easy)

        ratio = float(adv_hard[0]) / float(adv_easy[0])
        # Expected: (15/1) / (1/1) = 15. Exact for binary rewards.
        assert abs(ratio - 15.0) < 0.1, \
            f"Gradient weight ratio should be ~15, got {ratio}"


# ---------------------------------------------------------------------------
# H4: Comparison with GRPO — different optimization dynamics
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLVsGRPO:
    """Verify that MaxRL and GRPO produce different advantages with the same interface."""

    def test_different_advantages_same_binary_rewards(self):
        """H4: MaxRL and GRPO produce different advantages for binary rewards.

        GRPO: A = r - mean(r)
        MaxRL: A = (r - mean(r)) / (mean(r) + eps)

        For K=1, N=4:
          GRPO correct:  1 - 0.25 = 0.75
          MaxRL correct:  (1 - 0.25) / 0.25 = 3.0
        """
        rewards = [1.0, 0.0, 0.0, 0.0]

        grpo_adv = grpo.compute_advantages(rewards)
        maxrl_adv = grpo.compute_advantages_maxrl(rewards)
        mx.eval(grpo_adv, maxrl_adv)

        # Same sign but different magnitudes
        assert float(grpo_adv[0]) > 0 and float(maxrl_adv[0]) > 0, \
            "Both should give positive advantage to correct rollout"
        assert abs(float(grpo_adv[0]) - 0.75) < 1e-4
        assert abs(float(maxrl_adv[0]) - 3.0) < 1e-4

    def test_maxrl_amplifies_harder_more_than_grpo(self):
        """H4: MaxRL amplifies hard prompts more than GRPO relative to easy ones.

        GRPO advantage for correct rollout is always r - mean(r), which
        for binary rewards with K=1 is 1 - 1/N = (N-1)/N — bounded in (0,1).
        MaxRL advantage is (N-K)/K — unbounded, growing as K→0.
        """
        # Hard: K=1/N=8
        rewards_hard = [1.0] + [0.0] * 7
        # Easy: K=4/N=8
        rewards_easy = [1.0] * 4 + [0.0] * 4

        grpo_hard = float(grpo.compute_advantages(rewards_hard)[0].item())
        grpo_easy = float(grpo.compute_advantages(rewards_easy)[0].item())
        maxrl_hard = float(grpo.compute_advantages_maxrl(rewards_hard)[0].item())
        maxrl_easy = float(grpo.compute_advantages_maxrl(rewards_easy)[0].item())

        grpo_ratio = grpo_hard / grpo_easy if grpo_easy != 0 else float('inf')
        maxrl_ratio = maxrl_hard / maxrl_easy if maxrl_easy != 0 else float('inf')

        assert maxrl_ratio > grpo_ratio, \
            f"MaxRL should amplify hard vs easy more: MaxRL ratio={maxrl_ratio:.2f}, " \
            f"GRPO ratio={grpo_ratio:.2f}"

    def test_same_interface_as_compute_advantages(self):
        """H4: MaxRL follows the same interface — can be used as drop-in replacement.

        Both accept Union[List[float], mx.array] and return mx.array.
        Both return episode-level advantages [num_episodes].
        """
        # Test with list input
        rewards_list = [1.0, 0.5, 0.0]
        adv_list = grpo.compute_advantages_maxrl(rewards_list)
        assert isinstance(adv_list, mx.array)
        assert adv_list.shape == (3,)

        # Test with mx.array input
        rewards_arr = mx.array([1.0, 0.5, 0.0])
        adv_arr = grpo.compute_advantages_maxrl(rewards_arr)
        assert isinstance(adv_arr, mx.array)
        assert adv_arr.shape == (3,)

        # Both should give the same result
        assert mx.allclose(adv_list, adv_arr, atol=1e-6)

    def test_invalid_type_raises(self):
        """H4: Bad input type raises TypeError, matching compute_advantages behavior."""
        with pytest.raises(TypeError, match="Expected list or mx.array"):
            grpo.compute_advantages_maxrl("not_valid")

    def test_negative_rewards_raise(self):
        """H4: MaxRL rejects negative rewards (success-rate assumption)."""
        with pytest.raises(ValueError, match="non-negative rewards"):
            grpo.compute_advantages_maxrl([-1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# H5: Numerical stability, compiled variant, and integration
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLNumericalProperties:
    """Test numerical stability, compiled variant, and trainer compatibility."""

    def test_no_nan_or_inf(self):
        """H5: MaxRL should never produce NaN or Inf for valid inputs."""
        test_cases = [
            [0.0, 0.0, 0.0],       # All zero
            [1.0, 1.0, 1.0],       # All one
            [1.0, 0.0],            # Binary mixed
            [0.001, 0.0, 0.0],     # Very small success rate
            [1.0] * 100,           # Large group, all same
            [1.0] + [0.0] * 99,    # Large group, 1 success
        ]
        for rewards in test_cases:
            advantages = grpo.compute_advantages_maxrl(rewards)
            mx.eval(advantages)
            assert not mx.any(mx.isnan(advantages)), \
                f"NaN detected for rewards {rewards[:5]}..."
            assert not mx.any(mx.isinf(advantages)), \
                f"Inf detected for rewards {rewards[:5]}..."

    def test_compiled_matches_eager(self):
        """H5: Compiled variant produces identical results to eager version."""
        rewards = mx.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=mx.float32)

        eager = grpo.compute_advantages_maxrl(rewards)
        compiled = grpo.compute_advantages_maxrl_compiled(rewards)
        mx.eval(eager, compiled)

        assert mx.allclose(eager, compiled, atol=1e-5), \
            f"Compiled and eager should match: eager={eager}, compiled={compiled}"

    def test_compiled_zero_rewards(self):
        """H5: Compiled variant handles zero-mean case (masking to zero)."""
        rewards = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)
        compiled = grpo.compute_advantages_maxrl_compiled(rewards)
        mx.eval(compiled)

        assert mx.allclose(compiled, mx.zeros(3), atol=1e-5), \
            "Compiled variant should mask zero-mean to zero advantages"

    def test_compiled_negative_rewards_raise(self):
        """H5: Compiled variant also rejects negative rewards."""
        rewards = mx.array([-1.0, 1.0, 0.0], dtype=mx.float32)
        with pytest.raises(ValueError, match="non-negative rewards"):
            grpo.compute_advantages_maxrl_compiled(rewards)

    def test_continuous_rewards_still_valid(self):
        """H5: MaxRL works with continuous rewards (not just binary).

        The 1/p interpretation is exact only for binary rewards, but the
        formula (r - mean) / (mean + eps) is well-defined for any non-negative
        rewards. The reweighting still places more emphasis on episodes
        whose reward exceeds the group mean.
        """
        rewards = [0.9, 0.3, 0.7, 0.1]
        advantages = grpo.compute_advantages_maxrl(rewards)
        mx.eval(advantages)

        assert advantages.shape == (4,)
        assert not mx.any(mx.isnan(advantages))
        # Highest reward should get largest advantage
        assert float(advantages[0]) > float(advantages[1])
        assert float(advantages[0]) > float(advantages[3])

    def test_orthogonal_to_token_level_operations(self):
        """H5: MaxRL returns episode-level [num_episodes], composable with GTPO.

        MaxRL operates at the prompt level (which prompts get how much gradient).
        GTPO/SEPA operate at the token level (which tokens within a sequence
        get how much credit). They compose by multiplication after expansion.
        """
        rewards = [1.0, 0.0, 0.0, 1.0]
        episode_lengths = [3, 2, 4, 3]

        # Step 1: Episode-level advantages via MaxRL
        advantages = grpo.compute_advantages_maxrl(rewards)
        assert advantages.shape == (4,), "Should be episode-level"

        # Step 2: Manual expansion to token-level (what the trainer does)
        parts = []
        for i, length in enumerate(episode_lengths):
            parts.append(mx.repeat(advantages[i:i + 1], length))
        expanded = mx.concatenate(parts)
        assert expanded.shape == (12,), "Should expand to total tokens"

        # Step 3: GTPO entropy weighting on top (orthogonal)
        entropies = mx.ones(12) * 2.0  # uniform → no change
        weighted = grpo.apply_entropy_weighting(expanded, entropies, entropy_weight=0.1)
        assert mx.allclose(weighted, expanded, atol=1e-5), \
            "Uniform entropy weighting should not change expanded MaxRL advantages"


# ---------------------------------------------------------------------------
# H6: Strategy registry — MaxRL is selectable as an algorithm key
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMaxRLRegistration:
    """Verify that MaxRL is wired into the rollout strategy registry."""

    def test_create_strategy_maxrl_returns_grpo_strategy(self):
        """H6: 'maxrl' key resolves to GRPOStrategy (same rollout, different advantage_fn)."""
        from textpolicy.rollout.strategy import create_strategy, GRPOStrategy

        strategy = create_strategy("maxrl")
        assert isinstance(strategy, GRPOStrategy), \
            f"Expected GRPOStrategy, got {type(strategy).__name__}"

    def test_maxrl_strategy_same_type_as_grpo(self):
        """H6: MaxRL and GRPO share rollout behavior (no value function, stochastic)."""
        from textpolicy.rollout.strategy import create_strategy

        grpo_strategy = create_strategy("grpo")
        maxrl_strategy = create_strategy("maxrl")
        assert type(grpo_strategy) is type(maxrl_strategy), \
            "MaxRL and GRPO should use identical rollout strategies"


# ---------------------------------------------------------------------------
# H7: Per-prompt MaxRL — Algorithm 1 exact per-prompt grouping
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.algorithm
class TestMaxRLPerPrompt:
    """Verify per-prompt MaxRL advantages match Algorithm 1 of the paper.

    Algorithm 1 computes r_hat(x) = mean(rewards for completions of x)
    per prompt, not a single global mean. This resolves the caveat
    identified during the paper verification.

    References:
        Tajwar et al. (2026), Algorithm 1, line 8.
    """

    def test_two_groups_different_success_rates(self):
        """H7: Hard prompt (K=1/4) gets larger advantage than easy (K=3/4).

        Group 0 (hard): episodes 0-3, rewards [1,0,0,0], mean=0.25
            correct:   (1-0.25)/0.25 = 3.0
            incorrect: (0-0.25)/0.25 = -1.0

        Group 1 (easy): episodes 4-7, rewards [1,1,1,0], mean=0.75
            correct:   (1-0.75)/0.75 = 1/3
            incorrect: (0-0.75)/0.75 = -1.0
        """
        rewards = mx.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        prompt_ids = mx.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=mx.int32)

        advantages = grpo.compute_advantages_maxrl(
            rewards, prompt_ids=prompt_ids, num_prompt_groups=2
        )
        mx.eval(advantages)

        # Hard prompt correct (episode 0): advantage = 3.0
        assert abs(float(advantages[0]) - 3.0) < 1e-4, \
            f"Hard correct: expected 3.0, got {float(advantages[0])}"
        # Hard prompt incorrect (episodes 1-3): advantage = -1.0
        for i in [1, 2, 3]:
            assert abs(float(advantages[i]) - (-1.0)) < 1e-4, \
                f"Hard incorrect {i}: expected -1.0, got {float(advantages[i])}"
        # Easy prompt correct (episodes 4-6): advantage = 1/3
        for i in [4, 5, 6]:
            assert abs(float(advantages[i]) - (1.0 / 3.0)) < 1e-4, \
                f"Easy correct {i}: expected 1/3, got {float(advantages[i])}"
        # Easy prompt incorrect (episode 7): advantage = -1.0
        assert abs(float(advantages[7]) - (-1.0)) < 1e-4, \
            f"Easy incorrect: expected -1.0, got {float(advantages[7])}"

    def test_single_group_matches_global(self):
        """H7: When all episodes share one prompt, per-prompt == global MaxRL."""
        rewards = mx.array([1.0, 0.0, 0.0, 1.0])
        prompt_ids = mx.array([0, 0, 0, 0], dtype=mx.int32)

        per_prompt = grpo.compute_advantages_maxrl(
            rewards, prompt_ids=prompt_ids, num_prompt_groups=1
        )
        global_adv = grpo.compute_advantages_maxrl(rewards)
        mx.eval(per_prompt, global_adv)

        assert mx.allclose(per_prompt, global_adv, atol=1e-5), \
            "Single prompt group should match global MaxRL"

    def test_one_episode_per_group_zeroed(self):
        """H7: One episode per group → mean = r_i → advantage = 0 for all.

        With a single completion per prompt, there is no within-group
        variance. MaxRL correctly produces zero advantages: there is
        no contrastive signal to learn from.
        """
        rewards = mx.array([1.0, 0.0, 0.5])
        prompt_ids = mx.array([0, 1, 2], dtype=mx.int32)

        advantages = grpo.compute_advantages_maxrl(
            rewards, prompt_ids=prompt_ids, num_prompt_groups=3
        )
        mx.eval(advantages)

        assert mx.allclose(advantages, mx.zeros(3), atol=1e-5), \
            "One episode per group should produce zero advantages"

    def test_per_prompt_vs_global_hard_prompt_amplified(self):
        """H7: Per-prompt grouping gives the hard prompt a LARGER advantage
        than global MaxRL, because the global mean dilutes the hard prompt's
        low success rate with the easy prompt's high success rate.

        Group 0 (hard, K=1/4): per-prompt mean = 0.25, correct adv = 3.0
        Group 1 (easy, K=3/4): per-prompt mean = 0.75, correct adv = 1/3
        Global mean = 4/8 = 0.5, correct adv = (1-0.5)/0.5 = 1.0

        Per-prompt gives the hard correct rollout 3.0 vs global's 1.0.
        """
        rewards = mx.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        prompt_ids = mx.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=mx.int32)

        per_prompt = grpo.compute_advantages_maxrl(
            rewards, prompt_ids=prompt_ids, num_prompt_groups=2
        )
        global_adv = grpo.compute_advantages_maxrl(rewards)
        mx.eval(per_prompt, global_adv)

        # Per-prompt hard correct (ep 0): 3.0 > global hard correct (ep 0): 1.0
        assert float(per_prompt[0]) > float(global_adv[0]), \
            f"Per-prompt should amplify hard prompt more: " \
            f"per_prompt={float(per_prompt[0]):.2f} vs global={float(global_adv[0]):.2f}"

    def test_all_zero_group_masked(self):
        """H7: A group with all-zero rewards has all advantages zeroed.

        Group 0: all zero → mean=0 → no signal → advantages zeroed
        Group 1: [1, 0] → mean=0.5 → normal MaxRL
        """
        rewards = mx.array([0.0, 0.0, 1.0, 0.0])
        prompt_ids = mx.array([0, 0, 1, 1], dtype=mx.int32)

        advantages = grpo.compute_advantages_maxrl(
            rewards, prompt_ids=prompt_ids, num_prompt_groups=2
        )
        mx.eval(advantages)

        # Group 0: all zero → advantages should be zero
        assert abs(float(advantages[0])) < 1e-5
        assert abs(float(advantages[1])) < 1e-5
        # Group 1: mean=0.5, correct (ep 2): (1-0.5)/0.5 = 1.0
        assert abs(float(advantages[2]) - 1.0) < 1e-4
        # Group 1: incorrect (ep 3): (0-0.5)/0.5 = -1.0
        assert abs(float(advantages[3]) - (-1.0)) < 1e-4

    def test_prompt_ids_in_pack_episodes(self):
        """H7: _pack_episodes includes prompt_ids in batch output."""
        from textpolicy.algorithms.grpo import _pack_episodes
        from textpolicy.buffer.episode import Episode

        # Create two episodes with same obs (same prompt) and one with different
        ep1 = Episode()
        ep1.obs = [[1, 2, 3]]
        ep1.act = [[10]]
        ep1.rew = [1.0]
        ep1.next_obs = [[1, 2, 4]]
        ep1.done = [True]
        ep1.timeout = [False]
        ep1.logprob = [0.5]

        ep2 = Episode()
        ep2.obs = [[1, 2, 3]]  # Same prompt as ep1
        ep2.act = [[11]]
        ep2.rew = [0.0]
        ep2.next_obs = [[1, 2, 5]]
        ep2.done = [True]
        ep2.timeout = [False]
        ep2.logprob = [0.3]

        ep3 = Episode()
        ep3.obs = [[4, 5, 6]]  # Different prompt
        ep3.act = [[12]]
        ep3.rew = [1.0]
        ep3.next_obs = [[4, 5, 7]]
        ep3.done = [True]
        ep3.timeout = [False]
        ep3.logprob = [0.7]

        batch = _pack_episodes([ep1, ep2, ep3], sort_by_length=False)

        assert 'prompt_ids' in batch, "prompt_ids should be in batch_data"
        assert 'num_prompt_groups' in batch, "num_prompt_groups should be in batch_data"
        assert 'binary_rewards' in batch, "binary_rewards should be in batch_data"
        assert batch['num_prompt_groups'] == 2, \
            f"Expected 2 prompt groups, got {batch['num_prompt_groups']}"

        ids = batch['prompt_ids'].tolist()
        binary = batch["binary_rewards"].tolist()
        # ep1 and ep2 share the same prompt, ep3 is different
        assert ids[0] == ids[1], "ep1 and ep2 should have same prompt_id"
        assert ids[0] != ids[2], "ep3 should have a different prompt_id"
        assert binary == [1.0, 0.0, 1.0], f"Unexpected binary rewards: {binary}"

    def test_accepts_prompt_groups_attribute(self):
        """H7: compute_advantages_maxrl has the accepts_prompt_groups marker."""
        assert getattr(grpo.compute_advantages_maxrl, 'accepts_prompt_groups', False), \
            "compute_advantages_maxrl should have accepts_prompt_groups=True"
        # Other advantage functions should NOT have this marker
        assert not getattr(grpo.compute_advantages, 'accepts_prompt_groups', False), \
            "compute_advantages should NOT have accepts_prompt_groups"

    def test_pack_episodes_prefers_explicit_is_correct_signal(self):
        """Explicit verifier correctness should override reward-based inference."""
        from textpolicy.algorithms.grpo import _pack_episodes
        from textpolicy.buffer.episode import Episode

        ep = Episode()
        ep.obs = [[1, 2, 3]]
        ep.act = [[10]]
        ep.rew = [-0.5]  # malformed completion penalty
        ep.next_obs = [[1, 2, 3]]
        ep.done = [True]
        ep.timeout = [False]
        ep.logprob = [0.1]
        ep.is_correct = [True]  # explicit verifier signal

        batch = _pack_episodes([ep], sort_by_length=False)
        assert batch["binary_rewards"].tolist() == [1.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
