"""
Tests for training/reasoning_stack.py — GTPO+HICRA integration layer.

Covers:
  _flatten_padded_token_rows: 2D padded → flat 1D layout conversion
  build_gtpo_hicra_transform: factory validation and basic wiring
  _GTPOHICRATransform: transform correctness with entropy weighting
  create_tinylora_reasoning_setup: deprecated wrapper legacy-kwargs behavior

These were previously untested despite being the critical bridge between
the Trainer's internal data layout and the HICRA/GTPO algorithm functions.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest
import mlx.core as mx

from textpolicy.training.reasoning_stack import (
    _flatten_padded_token_rows,
    build_gtpo_hicra_transform,
    create_tinylora_reasoning_setup,
    _LEGACY_HICRA_ALPHA_DEFAULT,
    _LEGACY_ENTROPY_WEIGHT_DEFAULT,
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

    def test_create_tinylora_importable_but_deprecated(self):
        """create_tinylora_reasoning_setup must still be importable (deprecated)."""
        from textpolicy.training import create_tinylora_reasoning_setup
        assert callable(create_tinylora_reasoning_setup)

    def test_build_gtpo_transform_importable(self):
        """build_gtpo_transform (canonical GTPO builder) must be importable."""
        from textpolicy.training import build_gtpo_transform
        assert callable(build_gtpo_transform)

    def test_build_gtpo_transform_signature(self):
        """build_gtpo_transform must accept the expected GTPO + HICRA params."""
        import inspect
        from textpolicy.training import build_gtpo_transform

        sig = inspect.signature(build_gtpo_transform)
        for param_name in ("alpha_1", "alpha_2", "reward_threshold",
                           "tokenizer", "hicra_gamma", "strategic_grams"):
            assert param_name in sig.parameters, (
                f"build_gtpo_transform must accept '{param_name}'."
            )


# ---------------------------------------------------------------------------
# create_tinylora_reasoning_setup — deprecated wrapper legacy-kwargs behavior
# ---------------------------------------------------------------------------

# Module paths for patching (target the reasoning_stack module's own imports)
_RS = "textpolicy.training.reasoning_stack"


@pytest.mark.unit
class TestCreateTinyloraDeprecatedWrapper:
    """Validate the deprecated wrapper emits DeprecationWarning and routes
    hicra_alpha / entropy_weight to the correct transform builder without
    leaking them into Trainer().
    """

    def _call_wrapper(self, **overrides):
        """Call create_tinylora_reasoning_setup with mocked dependencies.

        Returns (mock_trainer_cls, mock_build_gtpo, mock_build_hicra) so
        callers can inspect which builder was called and what Trainer received.
        """
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_optimizer = MagicMock()
        mock_lora_model = MagicMock()
        mock_memory_stats = {"peak_memory_gb": 1.0}

        with (
            patch(f"{_RS}.create_lora_setup", return_value=(mock_lora_model, mock_memory_stats)) as mock_lora,
            patch(f"{_RS}.build_gtpo_transform") as mock_build_gtpo,
            patch(f"{_RS}.build_gtpo_hicra_transform") as mock_build_hicra,
            patch(f"{_RS}.Trainer") as mock_trainer_cls,
        ):
            # Emit the warning but don't let it raise
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                create_tinylora_reasoning_setup(
                    mock_model,
                    mock_tokenizer,
                    mock_optimizer,
                    **overrides,
                )

        return mock_trainer_cls, mock_build_gtpo, mock_build_hicra

    def test_emits_deprecation_warning(self):
        """Calling the wrapper must emit a DeprecationWarning."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_optimizer = MagicMock()

        with (
            patch(f"{_RS}.create_lora_setup", return_value=(MagicMock(), {})),
            patch(f"{_RS}.build_gtpo_transform"),
            patch(f"{_RS}.Trainer"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                create_tinylora_reasoning_setup(
                    mock_model, mock_tokenizer, mock_optimizer,
                )

        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert "deprecated" in str(deprecations[0].message).lower()

    def test_default_kwargs_use_build_gtpo_transform(self):
        """With default hicra_alpha and entropy_weight, uses build_gtpo_transform."""
        mock_trainer_cls, mock_build_gtpo, mock_build_hicra = self._call_wrapper()

        mock_build_gtpo.assert_called_once()
        mock_build_hicra.assert_not_called()

    def test_non_default_hicra_alpha_uses_hicra_transform(self):
        """Non-default hicra_alpha triggers build_gtpo_hicra_transform."""
        mock_trainer_cls, mock_build_gtpo, mock_build_hicra = self._call_wrapper(
            hicra_alpha=0.5,
        )

        mock_build_hicra.assert_called_once()
        mock_build_gtpo.assert_not_called()
        # Verify the custom alpha was forwarded
        call_kwargs = mock_build_hicra.call_args
        assert call_kwargs.kwargs["hicra_alpha"] == 0.5

    def test_non_default_entropy_weight_uses_hicra_transform(self):
        """Non-default entropy_weight triggers build_gtpo_hicra_transform."""
        mock_trainer_cls, mock_build_gtpo, mock_build_hicra = self._call_wrapper(
            entropy_weight=0.3,
        )

        mock_build_hicra.assert_called_once()
        mock_build_gtpo.assert_not_called()
        # Verify the custom entropy_weight was forwarded
        call_kwargs = mock_build_hicra.call_args
        assert call_kwargs.kwargs["entropy_weight"] == 0.3

    def test_both_legacy_kwargs_forwarded_to_hicra(self):
        """Both hicra_alpha and entropy_weight are forwarded correctly."""
        mock_trainer_cls, mock_build_gtpo, mock_build_hicra = self._call_wrapper(
            hicra_alpha=0.8,
            entropy_weight=0.05,
        )

        mock_build_hicra.assert_called_once()
        call_kwargs = mock_build_hicra.call_args
        assert call_kwargs.kwargs["hicra_alpha"] == 0.8
        assert call_kwargs.kwargs["entropy_weight"] == 0.05

    def test_legacy_kwargs_do_not_leak_into_trainer(self):
        """hicra_alpha and entropy_weight must NOT appear in Trainer() kwargs."""
        mock_trainer_cls, _, _ = self._call_wrapper(
            hicra_alpha=0.5,
            entropy_weight=0.3,
        )

        mock_trainer_cls.assert_called_once()
        trainer_call_kwargs = mock_trainer_cls.call_args
        all_kwargs = {**dict(zip(["model", "advantage_fn", "loss_fn", "optimizer"],
                                 trainer_call_kwargs.args)),
                      **trainer_call_kwargs.kwargs}
        assert "hicra_alpha" not in all_kwargs
        assert "entropy_weight" not in all_kwargs

    def test_explicit_transform_skips_both_builders(self):
        """When advantage_transform_fn is provided, neither builder is called."""
        custom_transform = MagicMock()
        mock_trainer_cls, mock_build_gtpo, mock_build_hicra = self._call_wrapper(
            advantage_transform_fn=custom_transform,
        )

        mock_build_gtpo.assert_not_called()
        mock_build_hicra.assert_not_called()
        # The custom transform should be passed to Trainer
        trainer_kwargs = mock_trainer_cls.call_args.kwargs
        assert trainer_kwargs["advantage_transform_fn"] is custom_transform

    def test_legacy_defaults_match_module_constants(self):
        """Ensure the test uses the same defaults as the module."""
        assert _LEGACY_HICRA_ALPHA_DEFAULT == 0.2
        assert _LEGACY_ENTROPY_WEIGHT_DEFAULT == 0.1

