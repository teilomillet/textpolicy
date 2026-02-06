"""
Tests for HICRA: Planning Token Amplification (Issue #11).

Validates the HICRA (High-Impact Credit Re-Assignment) implementation
which amplifies advantage signal at tokens participating in strategic
reasoning phrases (hesitation, verification, backtracking, etc.).

Hypotheses tested:
  TestIdentifyPlanningTokens:
    H1: Empty sequence → all-zero mask
    H2: No grams found → all-zero mask
    H3: Known phrase gets mask=1 at correct positions
    H4: Overlapping grams mark all participating tokens
    H5: Case-insensitive matching

  TestApplyHicraAmplification:
    H1: alpha=0 → unchanged (degenerate)
    H2: All-zero mask → unchanged (degenerate)
    H3: Positive advantage amplified: A*(1+alpha)
    H4: Negative advantage dampened: A*(1-alpha)
    H5: Zero advantage stays zero
    H6: Shape mismatch → ValueError
    H7: stop_gradient prevents gradient flow through mask
    H8: Exact formula verification (manual computation)

  TestHicraGradient:
    H1: Gradient w.r.t. planning_mask is zero
    H2: Gradient w.r.t. advantages is nonzero
    H3: Sign preserved under gradient

  TestHicraComposition:
    H1: HICRA alone = GRPO + amplification
    H2: GTPO + HICRA composes correctly

References:
    Issue #11 — HICRA Planning Token Amplification
    Issue #15 — Strategic Gram Mining Pipeline (produces the vocabulary)
"""

import pytest
import mlx.core as mx
import mlx.nn as nn

from textpolicy.algorithms.hicra import (
    apply_hicra_amplification,
    compute_advantages_hicra,
    identify_planning_tokens,
)
from textpolicy.algorithms.grpo import (
    apply_entropy_weighting,
    compute_advantages,
)


# ---------------------------------------------------------------------------
# Mock tokenizer for controlled testing
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock for identify_planning_tokens tests.

    Maps token IDs to pre-configured string tokens so we can control
    exactly what the sliding window sees.
    """

    def __init__(self, vocab: dict[int, str]):
        """
        Args:
            vocab: Mapping from token ID (int) to token string.
        """
        self._vocab = vocab

    def convert_ids_to_tokens(self, ids):
        return [self._vocab.get(i, "<unk>") for i in ids]


# ---------------------------------------------------------------------------
# TestIdentifyPlanningTokens
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIdentifyPlanningTokens:
    """Validate planning token identification via sliding window."""

    def test_empty_sequence(self):
        """H1: Empty token sequence → all-zero mask."""
        tok = MockTokenizer({})
        result = identify_planning_tokens(
            mx.array([], dtype=mx.int32), tok, ["let me think"]
        )
        assert result.shape == (0,)

    def test_no_grams_found(self):
        """H2: When no strategic grams match, mask is all zeros."""
        vocab = {0: "hello", 1: "world", 2: "foo"}
        tok = MockTokenizer(vocab)
        ids = mx.array([0, 1, 2], dtype=mx.int32)
        result = identify_planning_tokens(ids, tok, ["let me think"])
        expected = mx.array([0.0, 0.0, 0.0])
        assert mx.array_equal(result, expected)

    def test_known_phrase_matches(self):
        """H3: A known phrase gets mask=1 at correct token positions."""
        # Tokens: "let" "me" "think" "about" "it"
        vocab = {0: "let", 1: "me", 2: "think", 3: "about", 4: "it"}
        tok = MockTokenizer(vocab)
        ids = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        result = identify_planning_tokens(ids, tok, ["let me think"])
        # First 3 tokens should be marked
        expected = mx.array([1.0, 1.0, 1.0, 0.0, 0.0])
        assert mx.array_equal(result, expected), (
            f"Expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_overlapping_grams(self):
        """H4: Overlapping gram matches mark all participating tokens."""
        # Tokens: "let" "me" "think" "and" "check"
        vocab = {0: "let", 1: "me", 2: "think", 3: "and", 4: "check"}
        tok = MockTokenizer(vocab)
        ids = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        # Two grams: "let me think" (0,1,2) and "let me" which is a sub-match
        result = identify_planning_tokens(ids, tok, ["let me think", "and check"])
        # Tokens 0-2 match first gram, tokens 3-4 match second
        expected = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        assert mx.array_equal(result, expected), (
            f"Expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_case_insensitive(self):
        """H5: Matching is case-insensitive."""
        vocab = {0: "Let", 1: "Me", 2: "Think"}
        tok = MockTokenizer(vocab)
        ids = mx.array([0, 1, 2], dtype=mx.int32)
        result = identify_planning_tokens(ids, tok, ["let me think"])
        expected = mx.array([1.0, 1.0, 1.0])
        assert mx.array_equal(result, expected)

    def test_empty_grams_list(self):
        """Empty strategic grams → all-zero mask."""
        vocab = {0: "hello", 1: "world"}
        tok = MockTokenizer(vocab)
        ids = mx.array([0, 1], dtype=mx.int32)
        result = identify_planning_tokens(ids, tok, [])
        expected = mx.array([0.0, 0.0])
        assert mx.array_equal(result, expected)


# ---------------------------------------------------------------------------
# TestApplyHicraAmplification
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyHicraAmplification:
    """Validate the HICRA amplification formula."""

    def test_alpha_zero_unchanged(self):
        """H1: alpha=0 → advantages returned unchanged."""
        adv = mx.array([0.5, -0.3, 0.2])
        mask = mx.array([1.0, 1.0, 0.0])
        result = apply_hicra_amplification(adv, mask, alpha=0.0)
        assert mx.allclose(result, adv, atol=1e-6)

    def test_zero_mask_unchanged(self):
        """H2: All-zero mask → advantages unchanged regardless of alpha."""
        adv = mx.array([0.5, -0.3, 0.2])
        mask = mx.zeros(3)
        result = apply_hicra_amplification(adv, mask, alpha=0.5)
        assert mx.allclose(result, adv, atol=1e-6)

    def test_positive_advantage_amplified(self):
        """H3: Positive advantage + mask=1 → A*(1+alpha)."""
        adv = mx.array([1.0])
        mask = mx.array([1.0])
        alpha = 0.2
        result = apply_hicra_amplification(adv, mask, alpha=alpha)
        expected = mx.array([1.0 * (1 + alpha)])
        assert mx.allclose(result, expected, atol=1e-6)

    def test_negative_advantage_dampened(self):
        """H4: Negative advantage + mask=1 → A*(1-alpha)."""
        adv = mx.array([-1.0])
        mask = mx.array([1.0])
        alpha = 0.2
        result = apply_hicra_amplification(adv, mask, alpha=alpha)
        # A + alpha * |A| * mask = -1 + 0.2 * 1 * 1 = -0.8 = -1*(1-0.2)
        expected = mx.array([-1.0 * (1 - alpha)])
        assert mx.allclose(result, expected, atol=1e-6)

    def test_zero_advantage_stays_zero(self):
        """H5: Zero advantage stays zero regardless of mask and alpha."""
        adv = mx.array([0.0])
        mask = mx.array([1.0])
        result = apply_hicra_amplification(adv, mask, alpha=0.5)
        assert mx.allclose(result, mx.array([0.0]), atol=1e-6)

    def test_shape_mismatch_raises(self):
        """H6: Mismatched shapes raise ValueError."""
        adv = mx.array([0.5, -0.3])
        mask = mx.array([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            apply_hicra_amplification(adv, mask)

    def test_stop_gradient_on_mask(self):
        """H7: Gradient does not flow through the planning mask.

        We verify by differentiating a function that depends on the mask
        through apply_hicra_amplification. The gradient w.r.t. the mask
        input should be zero.
        """
        def loss_wrt_mask(mask):
            adv = mx.array([1.0, -0.5, 0.3])
            result = apply_hicra_amplification(adv, mask, alpha=0.2)
            return mx.sum(result)

        mask_in = mx.array([1.0, 0.0, 1.0])
        grad_fn = mx.grad(loss_wrt_mask)
        grad_mask = grad_fn(mask_in)
        mx.eval(grad_mask)
        assert mx.allclose(grad_mask, mx.zeros_like(grad_mask), atol=1e-6), (
            f"Gradient w.r.t. mask should be zero, got {grad_mask.tolist()}"
        )

    def test_exact_formula(self):
        """H8: Manual formula verification on a mixed batch."""
        adv = mx.array([0.5, -0.3, 0.2, -0.1, 0.0])
        mask = mx.array([1.0, 0.0, 1.0, 1.0, 1.0])
        alpha = 0.2

        result = apply_hicra_amplification(adv, mask, alpha=alpha)
        mx.eval(result)

        # Manual computation:
        # [0] A=0.5, mask=1: 0.5 + 0.2*0.5*1 = 0.5 + 0.1 = 0.6
        # [1] A=-0.3, mask=0: -0.3 + 0.2*0.3*0 = -0.3
        # [2] A=0.2, mask=1: 0.2 + 0.2*0.2*1 = 0.2 + 0.04 = 0.24
        # [3] A=-0.1, mask=1: -0.1 + 0.2*0.1*1 = -0.1 + 0.02 = -0.08
        # [4] A=0.0, mask=1: 0.0 + 0.2*0.0*1 = 0.0
        expected = mx.array([0.6, -0.3, 0.24, -0.08, 0.0])
        assert mx.allclose(result, expected, atol=1e-6), (
            f"Expected {expected.tolist()}, got {result.tolist()}"
        )


# ---------------------------------------------------------------------------
# TestHicraGradient
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHicraGradient:
    """Validate gradient properties of HICRA amplification."""

    def test_gradient_wrt_mask_is_zero(self):
        """H1: stop_gradient ensures zero gradient w.r.t. mask."""
        def f(mask):
            adv = mx.array([1.0, -0.5])
            return mx.sum(apply_hicra_amplification(adv, mask, alpha=0.3))

        mask = mx.array([1.0, 1.0])
        grad = mx.grad(f)(mask)
        mx.eval(grad)
        assert mx.allclose(grad, mx.zeros(2), atol=1e-6)

    def test_gradient_wrt_advantages_nonzero(self):
        """H2: Gradient w.r.t. advantages is nonzero for non-trivial mask."""
        def f(adv):
            mask = mx.array([1.0, 0.0, 1.0])
            return mx.sum(apply_hicra_amplification(adv, mask, alpha=0.2))

        adv = mx.array([1.0, -0.5, 0.3])
        grad = mx.grad(f)(adv)
        mx.eval(grad)
        # Not all zero
        assert mx.sum(mx.abs(grad)).item() > 0

    def test_sign_preserved_under_gradient(self):
        """H3: The sign of advantage values is preserved after amplification.

        For positive advantages, amplification increases magnitude.
        For negative advantages, amplification decreases magnitude (toward 0).
        In both cases sign is preserved (no flipping).
        """
        adv_pos = mx.array([0.5, 0.8, 0.3])
        adv_neg = mx.array([-0.5, -0.8, -0.3])
        mask = mx.ones(3)
        alpha = 0.3

        result_pos = apply_hicra_amplification(adv_pos, mask, alpha=alpha)
        result_neg = apply_hicra_amplification(adv_neg, mask, alpha=alpha)
        mx.eval(result_pos, result_neg)

        # Positive stays positive, negative stays negative
        assert all(v > 0 for v in result_pos.tolist())
        assert all(v < 0 for v in result_neg.tolist())


# ---------------------------------------------------------------------------
# TestHicraComposition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHicraComposition:
    """Validate HICRA composes correctly with GRPO and GTPO."""

    def test_hicra_alone_equals_grpo_plus_amplification(self):
        """H1: HICRA with no GTPO = base GRPO advantages + amplification."""
        rewards = [1.0, 0.0]
        episode_lengths = [3, 2]
        # Token IDs: "let" "me" "think" "hello" "world"
        vocab = {0: "let", 1: "me", 2: "think", 3: "hello", 4: "world"}
        tok = MockTokenizer(vocab)
        token_ids = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        grams = ["let me think"]
        alpha = 0.2

        # Manual computation
        base_adv = compute_advantages(rewards)
        # Expand: rewards [1.0, 0.0] → advantages [0.5, -0.5]
        # Expanded: [0.5, 0.5, 0.5, -0.5, -0.5]
        expanded = mx.concatenate([
            mx.repeat(base_adv[0:1], 3),
            mx.repeat(base_adv[1:2], 2),
        ])
        mask = identify_planning_tokens(token_ids, tok, grams)
        expected = apply_hicra_amplification(expanded, mask, alpha=alpha)

        result = compute_advantages_hicra(
            rewards, token_ids, tok, grams,
            alpha=alpha, episode_lengths=episode_lengths,
        )
        mx.eval(result, expected)
        assert mx.allclose(result, expected, atol=1e-6), (
            f"Expected {expected.tolist()}, got {result.tolist()}"
        )

    def test_misaligned_episode_lengths_raises(self):
        """H3: sum(episode_lengths) != len(token_ids) gives clear error."""
        rewards = [1.0, 0.0]
        episode_lengths = [3, 2]  # sum = 5
        vocab = {0: "a", 1: "b", 2: "c"}
        tok = MockTokenizer(vocab)
        # Only 3 token IDs, but episode_lengths sums to 5
        token_ids = mx.array([0, 1, 2], dtype=mx.int32)
        with pytest.raises(ValueError, match="sum\\(episode_lengths\\)=5.*token_ids length 3"):
            compute_advantages_hicra(
                rewards, token_ids, tok, ["a b"],
                episode_lengths=episode_lengths,
            )

    def test_gtpo_then_hicra_composition(self):
        """H2: GTPO + HICRA composes correctly (GTPO first, then HICRA)."""
        rewards = [1.0, 0.0]
        episode_lengths = [3, 2]
        vocab = {0: "let", 1: "me", 2: "think", 3: "hello", 4: "world"}
        tok = MockTokenizer(vocab)
        token_ids = mx.array([0, 1, 2, 3, 4], dtype=mx.int32)
        grams = ["let me think"]
        alpha = 0.2
        beta = 0.1

        # Token entropies
        token_entropies = mx.array([3.0, 1.0, 4.0, 2.0, 2.5])

        # Manual: base_adv → expand → GTPO weight → HICRA amplify
        base_adv = compute_advantages(rewards)
        expanded = mx.concatenate([
            mx.repeat(base_adv[0:1], 3),
            mx.repeat(base_adv[1:2], 2),
        ])
        after_gtpo = apply_entropy_weighting(expanded, token_entropies, beta)
        mask = identify_planning_tokens(token_ids, tok, grams)
        expected = apply_hicra_amplification(after_gtpo, mask, alpha=alpha)

        result = compute_advantages_hicra(
            rewards, token_ids, tok, grams,
            alpha=alpha,
            episode_lengths=episode_lengths,
            token_entropies=token_entropies,
            entropy_weight=beta,
        )
        mx.eval(result, expected)
        assert mx.allclose(result, expected, atol=1e-5), (
            f"Expected {expected.tolist()}, got {result.tolist()}"
        )
