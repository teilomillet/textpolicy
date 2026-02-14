"""
Tests for textpolicy.tinker.advantages — pure-Python advantage pipeline.

Validates the Tinker-compatible advantage functions against the formulas
from the MLX originals (grpo.py, hicra.py, sepa.py).

Hypotheses tested:
    H1: MaxRL advantages zero out when mean reward ≈ 0
    H2: MaxRL gives correct A = (N-K)/K for binary rewards
    H3: GTPO with β=0 or uniform entropy returns uniform advantages
    H4: GTPO amplifies high-entropy tokens and dampens low-entropy
    H5: HICRA with α=0 is identity; with α>0 amplifies masked tokens
    H6: HICRA sign behavior: positive amplified, negative dampened
    H7: SEPA with λ=0 is identity; λ=1 fully pools execution entropy
    H8: Planning token identification finds known strategic grams
    H9: SEPA controller scheduling matches MLX original
    H10: Full pipeline produces token-level advantages per completion
"""

import math
import pytest

from textpolicy.tinker.advantages import (
    compute_maxrl_advantages,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    identify_planning_tokens,
)
from textpolicy.tinker.sepa import SEPAController
from textpolicy.tinker.train_math import compute_token_advantages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def approx_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def approx_list_eq(xs: list, ys: list, tol: float = 1e-6) -> bool:
    return len(xs) == len(ys) and all(abs(x - y) < tol for x, y in zip(xs, ys))


class FakeTokenizer:
    """Minimal tokenizer for testing identify_planning_tokens."""

    def __init__(self, vocab: dict[int, str]):
        self._vocab = vocab

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._vocab.get(i, f"<unk_{i}>") for i in ids]


# ---------------------------------------------------------------------------
# H1, H2: MaxRL advantages
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMaxRLAdvantages:
    """Test MaxRL inverse success-rate reweighting."""

    def test_empty_rewards(self):
        """Edge case: empty input returns empty output."""
        assert compute_maxrl_advantages([]) == []

    def test_all_zero_rewards_returns_zeros(self):
        """H1: When mean ≈ 0, all advantages are zero (no signal)."""
        result = compute_maxrl_advantages([0.0, 0.0, 0.0, 0.0])
        assert all(r == 0.0 for r in result)

    def test_binary_rewards_formula(self):
        """H2: For K=1 correct out of N=4, correct gets A=(N-K)/K=3."""
        rewards = [1.0, 0.0, 0.0, 0.0]  # K=1, N=4, mean=0.25
        result = compute_maxrl_advantages(rewards)

        # Correct: (1.0 - 0.25) / (0.25 + eps) ≈ 3.0
        assert approx_eq(result[0], 3.0, tol=1e-4)
        # Incorrect: (0.0 - 0.25) / (0.25 + eps) ≈ -1.0
        assert approx_eq(result[1], -1.0, tol=1e-4)
        assert approx_eq(result[2], -1.0, tol=1e-4)
        assert approx_eq(result[3], -1.0, tol=1e-4)

    def test_all_correct_advantages_zero(self):
        """When all rewards are equal, advantages are zero."""
        result = compute_maxrl_advantages([1.0, 1.0, 1.0])
        assert all(approx_eq(r, 0.0, tol=1e-4) for r in result)

    def test_harder_problem_higher_advantage(self):
        """H2: Harder problems (lower success rate) → higher correct advantage."""
        # Easy: 3/4 correct
        easy = compute_maxrl_advantages([1.0, 1.0, 1.0, 0.0])
        # Hard: 1/4 correct
        hard = compute_maxrl_advantages([1.0, 0.0, 0.0, 0.0])

        # The correct completion's advantage should be higher for hard
        easy_correct_adv = easy[0]
        hard_correct_adv = hard[0]
        assert hard_correct_adv > easy_correct_adv


# ---------------------------------------------------------------------------
# H3, H4: GTPO entropy weighting
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGTPOWeighting:
    """Test GTPO entropy-weighted credit assignment."""

    def test_empty_entropies(self):
        """Edge case: empty input."""
        assert apply_gtpo_weighting(1.0, []) == []

    def test_beta_zero_returns_uniform(self):
        """H3: β=0 disables weighting → all advantages equal to base."""
        result = apply_gtpo_weighting(0.5, [1.0, 2.0, 3.0], beta=0.0)
        assert all(approx_eq(r, 0.5) for r in result)

    def test_uniform_entropy_returns_uniform(self):
        """H3: Uniform entropy → H_norm(t)=1 → weight=1 → unchanged."""
        result = apply_gtpo_weighting(0.5, [2.0, 2.0, 2.0], beta=0.1)
        assert all(approx_eq(r, 0.5) for r in result)

    def test_high_entropy_amplified(self):
        """H4: High-entropy tokens get larger advantage magnitude."""
        # Token 2 has 3x average entropy
        entropies = [1.0, 1.0, 3.0]
        result = apply_gtpo_weighting(1.0, entropies, beta=0.5)

        # mean_h = 5/3 ≈ 1.667
        # Token 2: h_norm = 3/1.667 ≈ 1.8 → weight = 1 + 0.5*(1.8-1) = 1.4
        # Token 0: h_norm = 1/1.667 ≈ 0.6 → weight = 1 + 0.5*(0.6-1) = 0.8
        assert result[2] > result[0], "High-entropy token should have larger advantage"
        assert result[2] > 1.0, "High-entropy token should be amplified beyond base"
        assert result[0] < 1.0, "Low-entropy token should be dampened below base"

    def test_all_zero_entropy_returns_uniform(self):
        """H3: All-zero entropy → no signal → uniform advantages."""
        result = apply_gtpo_weighting(0.5, [0.0, 0.0, 0.0], beta=0.1)
        assert all(approx_eq(r, 0.5) for r in result)

    def test_negative_advantage_sign_preserved(self):
        """GTPO should preserve sign: negative advantage stays negative."""
        result = apply_gtpo_weighting(-1.0, [1.0, 2.0, 3.0], beta=0.1)
        assert all(r <= 0.0 for r in result)

    def test_weight_clamped_to_nonnegative(self):
        """Large β with low-entropy tokens → weight clamped to 0, not negative."""
        # Token with h=0.1, mean_h=10 → h_norm=0.01 → raw_weight = 1 + 5*(0.01-1) = -3.95
        # Should be clamped to 0
        entropies = [0.1, 10.0, 10.0]
        result = apply_gtpo_weighting(1.0, entropies, beta=5.0)
        # Token 0 should have weight clamped to 0
        assert approx_eq(result[0], 0.0, tol=0.1)


# ---------------------------------------------------------------------------
# H5, H6: HICRA amplification
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHICRA:
    """Test HICRA planning token amplification."""

    def test_alpha_zero_is_identity(self):
        """H5: α=0 → no amplification."""
        advs = [0.5, -0.3, 0.2]
        mask = [1, 0, 1]
        result = apply_hicra(advs, mask, alpha=0.0)
        assert approx_list_eq(result, advs)

    def test_no_mask_is_identity(self):
        """H5: All-zero mask → no amplification."""
        advs = [0.5, -0.3, 0.2]
        mask = [0, 0, 0]
        result = apply_hicra(advs, mask, alpha=0.5)
        assert approx_list_eq(result, advs)

    def test_positive_advantage_amplified(self):
        """H6: A>0, mask=1 → A*(1+alpha)."""
        advs = [1.0]
        mask = [1]
        result = apply_hicra(advs, mask, alpha=0.2)
        # 1.0 + 0.2 * |1.0| * 1 = 1.2
        assert approx_eq(result[0], 1.2)

    def test_negative_advantage_dampened(self):
        """H6: A<0, mask=1 → A*(1-alpha) — blame dampened."""
        advs = [-1.0]
        mask = [1]
        result = apply_hicra(advs, mask, alpha=0.2)
        # -1.0 + 0.2 * |-1.0| * 1 = -1.0 + 0.2 = -0.8
        assert approx_eq(result[0], -0.8)

    def test_length_mismatch_raises(self):
        """Shape mismatch should raise ValueError."""
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_hicra([0.5, 0.3], [1], alpha=0.2)

    def test_mixed_mask(self):
        """Mixed mask: only masked tokens are affected."""
        advs = [0.5, -0.3, 0.2]
        mask = [1, 0, 1]
        result = apply_hicra(advs, mask, alpha=0.5)
        # Token 0: 0.5 + 0.5*0.5 = 0.75
        assert approx_eq(result[0], 0.75)
        # Token 1: unchanged
        assert approx_eq(result[1], -0.3)
        # Token 2: 0.2 + 0.5*0.2 = 0.3
        assert approx_eq(result[2], 0.3)


# ---------------------------------------------------------------------------
# H7: SEPA pooling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSEPAPooling:
    """Test SEPA selective entropy pooling."""

    def test_lambda_zero_is_identity(self):
        """H7: λ=0 → no pooling."""
        entropies = [1.0, 2.0, 3.0, 4.0]
        mask = [0, 1, 0, 0]
        result = apply_sepa_pooling(entropies, mask, lambda_t=0.0)
        assert approx_list_eq(result, entropies)

    def test_lambda_one_fully_pools_execution(self):
        """H7: λ=1 → execution tokens fully pooled to their mean."""
        entropies = [1.0, 5.0, 3.0, 7.0]
        mask = [0, 1, 0, 0]  # Token 1 is planning
        result = apply_sepa_pooling(entropies, mask, lambda_t=1.0)

        # Execution tokens: [1.0, 3.0, 7.0], mean = 11/3 ≈ 3.667
        exec_mean = (1.0 + 3.0 + 7.0) / 3

        # Planning token unchanged
        assert approx_eq(result[1], 5.0)
        # Execution tokens should all be exec_mean
        assert approx_eq(result[0], exec_mean, tol=1e-4)
        assert approx_eq(result[2], exec_mean, tol=1e-4)
        assert approx_eq(result[3], exec_mean, tol=1e-4)

    def test_partial_pooling(self):
        """Intermediate λ → interpolation between original and mean."""
        entropies = [1.0, 5.0, 3.0]
        mask = [0, 1, 0]  # Token 1 is planning
        result = apply_sepa_pooling(entropies, mask, lambda_t=0.5)

        # Token 0: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        assert approx_eq(result[0], 1.5)
        # Token 1: planning, unchanged
        assert approx_eq(result[1], 5.0)
        # Token 2: 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert approx_eq(result[2], 2.5)

    def test_all_planning_unchanged(self):
        """If all tokens are planning, nothing to pool."""
        entropies = [1.0, 2.0, 3.0]
        mask = [1, 1, 1]
        result = apply_sepa_pooling(entropies, mask, lambda_t=1.0)
        assert approx_list_eq(result, entropies)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            apply_sepa_pooling([1.0, 2.0], [0], lambda_t=0.5)


# ---------------------------------------------------------------------------
# H8: Planning token identification
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPlanningTokenIdentification:
    """Test strategic gram detection via tokenizer."""

    def test_simple_match(self):
        """H8: Detects 'let me think' in token stream."""
        vocab = {1: "let", 2: "me", 3: "think", 4: "about", 5: "this"}
        tokenizer = FakeTokenizer(vocab)
        token_ids = [1, 2, 3, 4, 5]  # "let me think about this"

        mask = identify_planning_tokens(
            token_ids, tokenizer, ["let me think"]
        )
        # Tokens 0,1,2 should be marked
        assert mask[0] == 1
        assert mask[1] == 1
        assert mask[2] == 1
        assert mask[3] == 0
        assert mask[4] == 0

    def test_no_match(self):
        """No strategic grams found → all zeros."""
        vocab = {1: "hello", 2: "world"}
        tokenizer = FakeTokenizer(vocab)
        mask = identify_planning_tokens(
            [1, 2], tokenizer, ["let me think"]
        )
        assert mask == [0, 0]

    def test_empty_tokens(self):
        """Empty token list → empty mask."""
        tokenizer = FakeTokenizer({})
        assert identify_planning_tokens([], tokenizer, ["let me think"]) == []

    def test_empty_grams(self):
        """Empty gram list → all zeros."""
        vocab = {1: "let", 2: "me"}
        tokenizer = FakeTokenizer(vocab)
        assert identify_planning_tokens([1, 2], tokenizer, []) == [0, 0]

    def test_default_grams_used_when_none(self):
        """When strategic_grams is None, uses DEFAULT_STRATEGIC_GRAMS."""
        vocab = {1: "let", 2: "me", 3: "think"}
        tokenizer = FakeTokenizer(vocab)
        mask = identify_planning_tokens([1, 2, 3], tokenizer, None)
        # "let me think" is in DEFAULT_STRATEGIC_GRAMS
        assert mask == [1, 1, 1]


# ---------------------------------------------------------------------------
# H9: SEPA Controller scheduling
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSEPAController:
    """Test SEPA scheduling logic."""

    def test_disabled_when_zero_steps(self):
        """Steps=0, schedule=linear → not enabled."""
        ctrl = SEPAController(sepa_steps=0, sepa_schedule="linear")
        assert not ctrl.enabled

    def test_linear_ramp(self):
        """Linear schedule: λ ramps from 0 to 1 over sepa_steps."""
        ctrl = SEPAController(sepa_steps=100, sepa_schedule="linear")
        assert approx_eq(ctrl.resolve_lambda(step=0), 0.0)
        assert approx_eq(ctrl.resolve_lambda(step=50), 0.5)
        assert approx_eq(ctrl.resolve_lambda(step=100), 1.0)
        assert approx_eq(ctrl.resolve_lambda(step=200), 1.0)  # capped

    def test_delay_steps(self):
        """Delay: λ stays 0 until delay_steps passed."""
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear", sepa_delay_steps=50
        )
        assert approx_eq(ctrl.resolve_lambda(step=0), 0.0)
        assert approx_eq(ctrl.resolve_lambda(step=50), 0.0)
        assert approx_eq(ctrl.resolve_lambda(step=100), 0.5)
        assert approx_eq(ctrl.resolve_lambda(step=150), 1.0)

    def test_correctness_gate(self):
        """λ stays 0 until correct rate exceeds gate threshold."""
        ctrl = SEPAController(
            sepa_steps=100, sepa_schedule="linear",
            sepa_correct_rate_gate=0.5,
        )
        # Gate closed → λ = 0 regardless of step
        assert approx_eq(ctrl.resolve_lambda(step=50), 0.0)

        # Below threshold
        ctrl.observe_correct_rate(0.3)
        assert not ctrl.gate_open
        assert approx_eq(ctrl.resolve_lambda(step=50), 0.0)

        # Meets threshold → gate opens (sticky)
        ctrl.observe_correct_rate(0.5)
        assert ctrl.gate_open
        assert ctrl.resolve_lambda(step=50) > 0.0

    def test_state_dict_roundtrip(self):
        """State dict save/load preserves scheduler state."""
        ctrl = SEPAController(sepa_steps=200, sepa_schedule="linear")
        ctrl.observe_correct_rate(0.0)  # no-op (gate already open since threshold=0)

        state = ctrl.state_dict()
        ctrl2 = SEPAController(sepa_steps=200, sepa_schedule="linear")
        ctrl2.load_state_dict(state)

        assert ctrl2.sepa_steps == ctrl.sepa_steps
        assert ctrl2.gate_open == ctrl.gate_open

    def test_auto_schedule_warmup(self):
        """Auto schedule: needs warmup observations before producing λ."""
        ctrl = SEPAController(
            sepa_steps=0, sepa_schedule="auto", sepa_warmup=3
        )
        assert ctrl.enabled

        # Not enough warmup
        ctrl.update_auto_state([1.0, 2.0, 3.0])
        ctrl.update_auto_state([1.0, 2.0, 3.0])
        # After 2 updates, var_0 is still None
        assert ctrl._var_0 is None

        # Third update triggers var_0 latch
        ctrl.update_auto_state([1.0, 2.0, 3.0])
        assert ctrl._var_0 is not None


# ---------------------------------------------------------------------------
# H10: Full pipeline integration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFullPipeline:
    """Test the complete advantage computation pipeline."""

    def test_compute_token_advantages_basic(self):
        """Full pipeline produces per-completion token advantages."""
        vocab = {1: "The", 2: "answer", 3: "is", 4: "42"}
        tokenizer = FakeTokenizer(vocab)

        rewards_G = [1.0, 0.0, 0.0, 0.0]
        tokens_G = [[1, 2, 3, 4]] * 4
        logprobs_G = [[-0.5, -1.0, -0.3, -2.0]] * 4

        result = compute_token_advantages(
            rewards_G=rewards_G,
            sampled_tokens_G=tokens_G,
            logprobs_G=logprobs_G,
            tokenizer=tokenizer,
            gtpo_beta=0.1,
            hicra_alpha=0.2,
            sepa_lambda=0.0,
        )

        # 4 completions, each with 4 tokens
        assert len(result) == 4
        for token_advs in result:
            assert len(token_advs) == 4
            # All values should be finite
            assert all(math.isfinite(a) for a in token_advs)

        # Correct completion should have positive advantages
        assert all(a >= 0.0 for a in result[0])
        # Incorrect completions should have non-positive advantages
        assert all(a <= 0.0 for a in result[1])

    def test_pipeline_with_sepa(self):
        """Pipeline with SEPA enabled still produces valid output."""
        vocab = {1: "let", 2: "me", 3: "think", 4: "ok"}
        tokenizer = FakeTokenizer(vocab)

        rewards_G = [1.0, 0.0]
        tokens_G = [[1, 2, 3, 4], [1, 2, 3, 4]]
        logprobs_G = [[-0.5, -1.0, -0.3, -2.0], [-0.5, -1.0, -0.3, -2.0]]

        result = compute_token_advantages(
            rewards_G=rewards_G,
            sampled_tokens_G=tokens_G,
            logprobs_G=logprobs_G,
            tokenizer=tokenizer,
            gtpo_beta=0.1,
            hicra_alpha=0.2,
            sepa_lambda=0.5,
            strategic_grams=["let me think"],
        )

        assert len(result) == 2
        for token_advs in result:
            assert len(token_advs) == 4
            assert all(math.isfinite(a) for a in token_advs)

    def test_single_completion_group(self):
        """Single completion → advantage = 0 (no group contrast)."""
        vocab = {1: "hello"}
        tokenizer = FakeTokenizer(vocab)

        result = compute_token_advantages(
            rewards_G=[1.0],
            sampled_tokens_G=[[1]],
            logprobs_G=[[-0.5]],
            tokenizer=tokenizer,
        )

        assert len(result) == 1
        # Single completion: mean = reward → advantage = 0
        assert all(approx_eq(a, 0.0, tol=1e-4) for a in result[0])


# ---------------------------------------------------------------------------
# Cross-validation with MLX originals
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCrossValidation:
    """Verify pure-Python port matches MLX originals numerically."""

    def test_maxrl_matches_mlx(self):
        """MaxRL: Python port matches grpo.compute_advantages_maxrl."""
        try:
            from textpolicy.algorithms.grpo import compute_advantages_maxrl
        except ImportError:
            pytest.skip("MLX not available")

        rewards = [1.0, 0.0, 0.0, 1.0, 0.0]
        py_result = compute_maxrl_advantages(rewards)
        mx_result = compute_advantages_maxrl(rewards).tolist()

        assert approx_list_eq(py_result, mx_result, tol=1e-4), \
            f"Python: {py_result}, MLX: {mx_result}"

    def test_gtpo_matches_mlx(self):
        """GTPO weighting: Python port matches grpo.apply_entropy_weighting."""
        try:
            import mlx.core as mx
            from textpolicy.algorithms.grpo import apply_entropy_weighting
        except ImportError:
            pytest.skip("MLX not available")

        advantage = 0.75
        entropies = [1.0, 2.0, 3.0, 0.5, 4.0]
        beta = 0.15

        py_result = apply_gtpo_weighting(advantage, entropies, beta=beta)

        # MLX version takes already-expanded advantages + entropies
        mx_advantages = mx.array([advantage] * len(entropies))
        mx_entropies = mx.array(entropies)
        mx_result = apply_entropy_weighting(
            mx_advantages, mx_entropies, entropy_weight=beta
        ).tolist()

        assert approx_list_eq(py_result, mx_result, tol=1e-4), \
            f"Python: {py_result}, MLX: {mx_result}"

    def test_hicra_matches_mlx(self):
        """HICRA amplification: Python port matches hicra.apply_hicra_amplification."""
        try:
            import mlx.core as mx
            from textpolicy.algorithms.hicra import apply_hicra_amplification
        except ImportError:
            pytest.skip("MLX not available")

        token_advs = [0.5, -0.3, 0.2, -0.1, 0.4]
        planning_mask = [1, 0, 1, 0, 1]
        alpha = 0.25

        py_result = apply_hicra(token_advs, planning_mask, alpha=alpha)
        mx_result = apply_hicra_amplification(
            mx.array(token_advs),
            mx.array(planning_mask, dtype=mx.float32),
            alpha=alpha,
        ).tolist()

        assert approx_list_eq(py_result, mx_result, tol=1e-5), \
            f"Python: {py_result}, MLX: {mx_result}"

    def test_sepa_pooling_matches_mlx(self):
        """SEPA pooling: Python port matches SEPAController.apply."""
        try:
            import mlx.core as mx
            from textpolicy.training.sepa import SEPAController as MLXSEPAController
        except ImportError:
            pytest.skip("MLX not available")

        entropies = [1.0, 5.0, 3.0, 7.0, 2.0]
        planning_mask = [0, 1, 0, 0, 1]
        lambda_t = 0.6

        py_result = apply_sepa_pooling(entropies, planning_mask, lambda_t)

        mlx_ctrl = MLXSEPAController(sepa_steps=100)
        mx_result = mlx_ctrl.apply(
            mx.array(entropies),
            mx.array(planning_mask, dtype=mx.float32),
            lambda_t=lambda_t,
        ).tolist()

        assert approx_list_eq(py_result, mx_result, tol=1e-4), \
            f"Python: {py_result}, MLX: {mx_result}"
