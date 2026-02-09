"""
Tests for training/reasoning_stack.py — GTPO+HICRA integration layer.

Covers:
  _flatten_padded_token_rows: 2D padded → flat 1D layout conversion
  build_gtpo_hicra_transform: factory validation and basic wiring
  _GTPOHICRATransform: transform correctness with entropy weighting

These were previously untested despite being the critical bridge between
the Trainer's internal data layout and the HICRA/GTPO algorithm functions.
"""

import pytest
import mlx.core as mx

from textpolicy.training.reasoning_stack import (
    _flatten_padded_token_rows,
    build_gtpo_hicra_transform,
)


# ---------------------------------------------------------------------------
# _flatten_padded_token_rows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFlattenPaddedTokenRows:
    """Validate 2D padded → flat 1D conversion."""

    def test_1d_passthrough(self):
        """1D input is returned unchanged (no flattening needed)."""
        values = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _flatten_padded_token_rows(values, None, field_name="act")
        assert mx.array_equal(result, values)

    def test_2d_uniform_lengths(self):
        """2D with equal episode lengths → correct flat concatenation."""
        # 2 episodes, max_len=3, both have length 3 (no padding)
        values = mx.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        result = _flatten_padded_token_rows(values, [3, 3], field_name="act")
        expected = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert mx.array_equal(result, expected)

    def test_2d_variable_lengths_strips_padding(self):
        """2D with variable lengths → padding tokens are excluded."""
        # 2 episodes: ep0 has 2 tokens, ep1 has 3 tokens, padded to max_len=3
        values = mx.array([
            [1.0, 2.0, 0.0],   # ep0: only first 2 are real
            [4.0, 5.0, 6.0],   # ep1: all 3 are real
        ])
        result = _flatten_padded_token_rows(values, [2, 3], field_name="act")
        expected = mx.array([1.0, 2.0, 4.0, 5.0, 6.0])
        assert mx.array_equal(result, expected)

    def test_zero_length_episode_skipped(self):
        """Episode with length 0 contributes no tokens to output."""
        values = mx.array([
            [1.0, 2.0],
            [0.0, 0.0],  # ep1: length=0, should be skipped entirely
            [3.0, 4.0],
        ])
        result = _flatten_padded_token_rows(values, [2, 0, 2], field_name="act")
        expected = mx.array([1.0, 2.0, 3.0, 4.0])
        assert mx.array_equal(result, expected)

    def test_all_zero_lengths_returns_empty(self):
        """All episodes have length 0 → returns empty array."""
        values = mx.array([[0.0, 0.0], [0.0, 0.0]])
        result = _flatten_padded_token_rows(values, [0, 0], field_name="act")
        assert result.size == 0

    def test_3d_input_raises(self):
        """3D input is invalid — must be 1D or 2D."""
        values = mx.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            _flatten_padded_token_rows(values, [3, 3], field_name="act")

    def test_2d_without_episode_lengths_raises(self):
        """2D input requires episode_lengths."""
        values = mx.zeros((2, 3))
        with pytest.raises(ValueError, match="episode_lengths.*required"):
            _flatten_padded_token_rows(values, None, field_name="act")

    def test_row_count_mismatch_raises(self):
        """Number of rows must match len(episode_lengths)."""
        values = mx.zeros((3, 4))
        with pytest.raises(ValueError, match="does not match"):
            _flatten_padded_token_rows(values, [4, 4], field_name="act")

    def test_negative_length_raises(self):
        """Negative episode length is invalid."""
        values = mx.zeros((2, 3))
        with pytest.raises(ValueError, match="must be >= 0"):
            _flatten_padded_token_rows(values, [3, -1], field_name="act")

    def test_length_exceeds_padded_width_raises(self):
        """Episode length > max_len is invalid (overrun)."""
        values = mx.zeros((2, 3))
        with pytest.raises(ValueError, match="exceeds padded width"):
            _flatten_padded_token_rows(values, [3, 5], field_name="act")

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        values = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
        result = _flatten_padded_token_rows(values, [3, 3], field_name="act")
        assert result.dtype == mx.int32


# ---------------------------------------------------------------------------
# build_gtpo_hicra_transform — factory validation
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal tokenizer mock for transform tests."""
    def convert_ids_to_tokens(self, ids):
        return [f"tok{i}" for i in ids]


@pytest.mark.unit
class TestBuildGtpoHicraTransform:
    """Validate factory function input validation and wiring."""

    def test_negative_entropy_weight_raises(self):
        """Negative entropy_weight is invalid."""
        with pytest.raises(ValueError, match="entropy_weight must be >= 0"):
            build_gtpo_hicra_transform(
                MockTokenizer(), entropy_weight=-0.1
            )

    def test_negative_hicra_alpha_raises(self):
        """Negative hicra_alpha is invalid."""
        with pytest.raises(ValueError, match="hicra_alpha must be >= 0"):
            build_gtpo_hicra_transform(
                MockTokenizer(), hicra_alpha=-0.5
            )

    def test_returns_callable(self):
        """Factory returns a callable transform."""
        transform = build_gtpo_hicra_transform(MockTokenizer())
        assert callable(transform)

    def test_has_prepare_batch_method(self):
        """Transform exposes prepare_batch for Trainer's eager hook."""
        transform = build_gtpo_hicra_transform(MockTokenizer())
        assert hasattr(transform, "prepare_batch")
        assert callable(transform.prepare_batch)

    def test_custom_grams_are_used(self):
        """Custom strategic grams are passed through to the transform."""
        custom_grams = ["step one", "next step"]
        transform = build_gtpo_hicra_transform(
            MockTokenizer(), strategic_grams=custom_grams
        )
        assert transform.grams == custom_grams

    def test_alpha_zero_disables_hicra(self):
        """alpha=0 should effectively disable HICRA amplification."""
        transform = build_gtpo_hicra_transform(
            MockTokenizer(), hicra_alpha=0.0
        )
        # With alpha=0, prepare_batch should be a no-op (no planning mask computed)
        batch = {"act": mx.array([1, 2, 3], dtype=mx.int32)}
        transform.prepare_batch(batch)
        assert "planning_mask" not in batch


# ---------------------------------------------------------------------------
# _GTPOHICRATransform.__call__ — transform correctness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGTPOHICRATransformCall:
    """Validate the transform applies GTPO then HICRA correctly."""

    def _make_transform(self, **kwargs):
        """Helper to build a transform with test defaults."""
        defaults = dict(
            strategic_grams=["let me think"],
            hicra_alpha=0.2,
            entropy_weight=0.0,
        )
        defaults.update(kwargs)
        vocab = {0: "let", 1: "me", 2: "think", 3: "hello", 4: "world"}
        class Tok:
            def convert_ids_to_tokens(self, ids):
                return [vocab.get(int(i), "x") for i in ids]
        return build_gtpo_hicra_transform(Tok(), **defaults)

    def test_hicra_amplifies_planning_tokens(self):
        """Planning tokens get amplified advantages."""
        transform = self._make_transform(hicra_alpha=0.2, entropy_weight=0.0)
        # Tokens 0-2 are "let me think" (planning), 3-4 are not
        advantages = mx.array([1.0, 1.0, 1.0, 1.0, 1.0])
        batch = {
            "act": mx.array([0, 1, 2, 3, 4], dtype=mx.int32),
        }
        result = transform(advantages, batch)
        mx.eval(result)

        # Planning tokens (0-2) should be amplified: 1.0 * (1 + 0.2) = 1.2
        assert result[0].item() == pytest.approx(1.2, abs=1e-5)
        assert result[1].item() == pytest.approx(1.2, abs=1e-5)
        assert result[2].item() == pytest.approx(1.2, abs=1e-5)
        # Non-planning tokens unchanged
        assert result[3].item() == pytest.approx(1.0, abs=1e-5)
        assert result[4].item() == pytest.approx(1.0, abs=1e-5)

    def test_hicra_disabled_when_alpha_zero(self):
        """alpha=0 → advantages pass through unchanged."""
        transform = self._make_transform(hicra_alpha=0.0, entropy_weight=0.0)
        advantages = mx.array([0.5, -0.3, 0.8])
        batch = {"act": mx.array([0, 1, 2], dtype=mx.int32)}
        result = transform(advantages, batch)
        mx.eval(result)
        assert mx.allclose(result, advantages, atol=1e-6)

    def test_shape_preserved(self):
        """Transform must not change the shape of advantages."""
        transform = self._make_transform()
        advantages = mx.array([1.0, -0.5, 0.3, 0.7, -0.2])
        batch = {"act": mx.array([0, 1, 2, 3, 4], dtype=mx.int32)}
        result = transform(advantages, batch)
        assert result.shape == advantages.shape

    def test_shape_mismatch_raises(self):
        """Mismatched act/advantages shapes should raise."""
        transform = self._make_transform()
        advantages = mx.array([1.0, 2.0, 3.0])
        batch = {"act": mx.array([0, 1], dtype=mx.int32)}  # 2 != 3
        with pytest.raises(ValueError, match="does not match"):
            transform(advantages, batch)

    def test_missing_act_raises(self):
        """batch_data without 'act' key should raise."""
        transform = self._make_transform()
        advantages = mx.array([1.0, 2.0])
        with pytest.raises(ValueError, match="act"):
            transform(advantages, {})

    def test_2d_padded_act_flattened_correctly(self):
        """2D padded act tensor is flattened before matching."""
        transform = self._make_transform(hicra_alpha=0.0, entropy_weight=0.0)
        advantages = mx.array([0.5, -0.3, 0.8, 0.1, -0.2])
        # 2D padded: ep0 has 3 tokens, ep1 has 2 tokens, max_len=3
        batch = {
            "act": mx.array([[0, 1, 2], [3, 4, 0]], dtype=mx.int32),
            "episode_lengths": [3, 2],
        }
        result = transform(advantages, batch)
        mx.eval(result)
        # With alpha=0, should pass through unchanged
        assert mx.allclose(result, advantages, atol=1e-6)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReasoningStackPublicAPI:
    """Verify reasoning stack is importable from training package."""

    def test_importable_from_training_package(self):
        """build_gtpo_hicra_transform must be importable from textpolicy.training."""
        from textpolicy.training import build_gtpo_hicra_transform
        assert callable(build_gtpo_hicra_transform)

    def test_build_gtpo_transform_importable(self):
        """build_gtpo_transform (canonical GTPO builder) must be importable."""
        from textpolicy.training import build_gtpo_transform
        assert callable(build_gtpo_transform)

    def test_sepa_controller_importable(self):
        """SEPAController should be importable as a standalone component."""
        from textpolicy.training import SEPAController
        assert callable(SEPAController)

    def test_semantic_entropy_tracker_importable(self):
        """SemanticEntropyTracker should be importable as a standalone component."""
        from textpolicy.training import SemanticEntropyTracker
        assert callable(SemanticEntropyTracker)

    def test_build_gtpo_transform_signature(self):
        """build_gtpo_transform must accept the expected GTPO + HICRA params."""
        import inspect
        from textpolicy.training import build_gtpo_transform

        sig = inspect.signature(build_gtpo_transform)
        for param_name in ("alpha_1", "alpha_2", "reward_threshold",
                           "tokenizer", "hicra_gamma", "strategic_grams",
                           "sepa_steps", "sepa_schedule"):
            assert param_name in sig.parameters, (
                f"build_gtpo_transform must accept '{param_name}'."
            )

