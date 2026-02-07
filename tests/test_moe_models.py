"""
Tests for Mixture-of-Experts (MoE) model support.

Validates that the full textpolicy pipeline — loading, detection, LoRA
application, freeze/unfreeze, forward pass, training step, and save/load —
works correctly with MoE architectures (SwitchLinear/SwitchGLU).

Hypotheses tested:
  H1: Detection correctly classifies MoE vs dense models
  H2: LoRA applies to SwitchLinear layers via LoRASwitchLinear
  H3: freeze_base handles LoRASwitchLinear (3D lora_a/lora_b)
  H4: Forward pass produces valid logprobs through MoE routing
  H5: Training step produces finite loss with MoE + LoRA
  H6: 3D LoRA params survive save/load roundtrip

Test model: ``arcee-ai/Trinity-Nano-Preview`` (AfMoE, ~3GB at 4-bit).
Override with ``TEXTPOLICY_MOE_MODEL`` env var.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest
from mlx.utils import tree_flatten

from textpolicy.generation import (
    detect_moe_model,
    get_moe_config,
    load_model,
)
from textpolicy.generation.lora import (
    apply_lora,
    extract_params,
    freeze_base,
)


pytestmark = [pytest.mark.requires_moe_model, pytest.mark.slow]

_MOE_MODEL_ENV = "TEXTPOLICY_MOE_MODEL"
_DEFAULT_MOE_MODEL = "arcee-ai/Trinity-Nano-Preview"


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def moe_model_and_tokenizer():
    """Load the MoE model once per module to amortise download/init cost."""
    model_id = os.environ.get(_MOE_MODEL_ENV, _DEFAULT_MOE_MODEL)
    try:
        model, tokenizer = load_model(model_id)
    except Exception as exc:
        pytest.skip(f"Could not load MoE model {model_id!r}: {exc}")
    return model, tokenizer


@pytest.fixture(scope="module")
def moe_model(moe_model_and_tokenizer):
    return moe_model_and_tokenizer[0]


@pytest.fixture(scope="module")
def moe_tokenizer(moe_model_and_tokenizer):
    return moe_model_and_tokenizer[1]


@pytest.fixture(scope="module")
def dense_model():
    """Load a known-dense model for negative detection tests."""
    try:
        model, _tok = load_model("mlx-community/Qwen3-0.6B-4bit")
    except Exception as exc:
        pytest.skip(f"Could not load dense model: {exc}")
    return model


@pytest.fixture(scope="module")
def moe_model_with_lora(moe_model_and_tokenizer):
    """Return MoE model with LoRA applied (module-scoped, applied once)."""
    model, _tok = moe_model_and_tokenizer
    apply_lora(model, lora_layers=2, lora_rank=4, lora_scale=20.0)
    freeze_base(model)
    # MoE models require training mode so that SwitchGLU/SwitchMLP apply
    # mx.stop_gradient to routing indices (otherwise gather_mm VJP fails).
    # Trainer.__init__ does this automatically; tests must do it manually.
    model.train()
    return model


# ── H1: Detection correctly classifies MoE vs dense ──────────────────────


class TestMoEDetection:
    """H1: detect_moe_model and get_moe_config correctly classify architectures."""

    def test_moe_model_detected(self, moe_model):
        """MoE model should be detected as MoE."""
        assert detect_moe_model(moe_model) is True

    def test_moe_config_extracted(self, moe_model):
        """get_moe_config should return a dict with num_experts for MoE models."""
        config = get_moe_config(moe_model)
        assert config is not None
        assert config["is_moe"] is True
        assert "num_experts" in config, (
            f"Expected 'num_experts' in MoE config, got keys: {list(config.keys())}"
        )
        assert config["num_experts"] > 1

    def test_dense_not_detected_as_moe(self, dense_model):
        """Dense model must not be detected as MoE."""
        assert detect_moe_model(dense_model) is False
        assert get_moe_config(dense_model) is None


# ── H2: LoRA applies to SwitchLinear layers ───────────────────────────────


class TestMoELoRAApplication:
    """H2: apply_lora correctly converts SwitchLinear → LoRASwitchLinear."""

    def test_lora_applied_without_error(self, moe_model_with_lora):
        """LoRA application should complete without raising."""
        # If we get here, apply_lora + freeze_base didn't raise.
        assert moe_model_with_lora is not None

    def test_lora_switch_linear_present(self, moe_model_with_lora):
        """At least one LoRASwitchLinear module should exist after apply_lora."""
        try:
            from mlx_lm.tuner.lora import LoRASwitchLinear
        except ImportError:
            pytest.skip("LoRASwitchLinear not available in this MLX-LM version")

        found = False
        for _name, module in moe_model_with_lora.named_modules():
            if isinstance(module, LoRASwitchLinear):
                found = True
                break
        assert found, "Expected at least one LoRASwitchLinear module after apply_lora"

    def test_lora_params_3d_shape(self, moe_model_with_lora):
        """MoE LoRA params should be 3D: [num_experts, rank, dim] or [num_experts, dim, rank]."""
        params = extract_params(moe_model_with_lora)
        assert len(params) > 0, "No LoRA parameters found"

        found_3d = False
        for name, param in params.items():
            if param.ndim == 3:
                found_3d = True
                break

        assert found_3d, (
            f"Expected at least one 3D LoRA param (per-expert), "
            f"but all params have dims: "
            f"{set(p.ndim for p in params.values())}"
        )


# ── H3: freeze_base handles LoRASwitchLinear ──────────────────────────────


class TestMoEFreezeUnfreeze:
    """H3: freeze_base correctly freezes base weights and unfreezes LoRA weights."""

    def test_freeze_unfreezes_lora_only(self, moe_model_with_lora):
        """After freeze_base, only LoRA params should be trainable."""
        trainable = dict(tree_flatten(moe_model_with_lora.trainable_parameters()))
        assert len(trainable) > 0, "No trainable parameters after freeze_base"

        for name in trainable:
            assert "lora" in name.lower(), (
                f"Non-LoRA parameter {name!r} is trainable after freeze_base"
            )

    def test_trainable_includes_3d_params(self, moe_model_with_lora):
        """Trainable params should include 3D per-expert LoRA weights."""
        trainable = dict(tree_flatten(moe_model_with_lora.trainable_parameters()))

        dims = {name: p.ndim for name, p in trainable.items()}
        has_3d = any(d == 3 for d in dims.values())
        assert has_3d, (
            f"Expected 3D trainable LoRA params, got dims: "
            f"{set(dims.values())}"
        )


# ── H4: Forward pass produces valid logprobs ──────────────────────────────


class TestMoEForwardPass:
    """H4: Model forward pass through MoE routing produces valid logits."""

    def test_forward_pass_finite(self, moe_model_with_lora, moe_tokenizer):
        """Forward pass should produce finite logits."""
        tokens = moe_tokenizer.encode("Hello, world!")
        input_ids = mx.array([tokens], dtype=mx.int32)

        logits = moe_model_with_lora(input_ids)
        mx.eval(logits)

        assert mx.all(mx.isfinite(logits)).item(), "Logits contain non-finite values"

    def test_logprob_extraction(self, moe_model_with_lora, moe_tokenizer):
        """Log probabilities should be valid (negative, finite)."""
        tokens = moe_tokenizer.encode("The quick brown fox")
        input_ids = mx.array([tokens], dtype=mx.int32)

        logits = moe_model_with_lora(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(log_probs)

        assert mx.all(mx.isfinite(log_probs)).item(), "Log probs contain non-finite values"
        # Log probs should be <= 0
        assert mx.all(log_probs <= 1e-6).item(), "Log probs should be non-positive"


# ── H5: Training step produces finite loss ────────────────────────────────


class TestMoETrainingStep:
    """H5: A single training step with MoE + LoRA produces a finite loss."""

    def test_training_step_finite_loss(self, moe_model_with_lora, moe_tokenizer):
        """A single nn.value_and_grad step should produce finite loss and grads."""
        from textpolicy.algorithms import grpo

        # Encode a short prompt
        tokens = moe_tokenizer.encode("Count: 1, 2, 3")
        input_ids = mx.array([tokens], dtype=mx.int32)

        # Get initial logprobs (simulating a rollout)
        logits = moe_model_with_lora(input_ids)
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # Pick logprob of the actual next token for each position
        old_logprobs = mx.zeros(len(tokens) - 1)
        for i in range(len(tokens) - 1):
            old_logprobs = old_logprobs.at[i].add(log_probs[0, i, tokens[i + 1]])
        mx.eval(old_logprobs)

        # Define a minimal loss function
        def loss_fn(model, old_lp, input_ids, target_ids):
            logits = model(input_ids)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            new_lp = mx.zeros(target_ids.shape[0])
            for i in range(target_ids.shape[0]):
                new_lp = new_lp.at[i].add(lp[0, i, target_ids[i]])
            advantages = mx.ones(1)  # single-episode uniform advantage
            token_advantages = mx.repeat(advantages, target_ids.shape[0])
            return grpo.policy_loss(old_lp, new_lp, token_advantages)

        target_ids = mx.array(tokens[1:], dtype=mx.int32)
        loss_and_grad = nn.value_and_grad(
            moe_model_with_lora,
            lambda m: loss_fn(m, old_logprobs, input_ids, target_ids),
        )

        loss, grads = loss_and_grad(moe_model_with_lora)
        mx.eval(loss)
        assert mx.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"


# ── H6: 3D params survive save/load roundtrip ────────────────────────────


class TestMoELoRASaveLoad:
    """H6: LoRA parameters (including 3D per-expert) survive safetensors roundtrip."""

    def test_extract_includes_3d(self, moe_model_with_lora):
        """extract_params should include 3D LoRA parameters."""
        params = extract_params(moe_model_with_lora)
        dims_3d = {k: v.shape for k, v in params.items() if v.ndim == 3}
        assert len(dims_3d) > 0, "extract_params should include 3D per-expert params"

    def test_save_load_roundtrip(self, moe_model_with_lora):
        """Saving and reloading LoRA params should preserve shapes and values."""
        params = extract_params(moe_model_with_lora)
        assert len(params) > 0

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name

        try:
            # Save
            mx.save_safetensors(path, params)

            # Load
            loaded = dict(mx.load(path))

            assert set(loaded.keys()) == set(params.keys()), (
                f"Key mismatch: saved {set(params.keys())}, loaded {set(loaded.keys())}"
            )

            for name in params:
                assert loaded[name].shape == params[name].shape, (
                    f"{name}: shape mismatch {loaded[name].shape} vs {params[name].shape}"
                )
                assert mx.allclose(loaded[name], params[name], atol=1e-6).item(), (
                    f"{name}: value mismatch after roundtrip"
                )
        finally:
            os.unlink(path)
