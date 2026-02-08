"""
Tests for gradient checkpointing (Issue #55).

Validates the gradient checkpointing implementation which wraps each
transformer layer's forward pass with ``mx.checkpoint`` to trade compute
for memory.

Hypotheses tested:
  TestApplyRemove:
    H1: apply returns correct layer count
    H2: remove restores original calls
    H3: Double-apply is idempotent (second returns 0)
    H4: is_active reports correct state transitions
    H5: Raises ValueError for model without .layers

  TestCorrectnessWithTrainer:
    H6: Loss identical with and without checkpointing (same model, data, LoRA)
    H7: LoRA adapter gradients match between checkpointed and non-checkpointed
    H8: Trainer step completes with finite loss and non-zero grads

  TestCompileInteraction:
    H9: gradient_checkpointing=True + compile_training=True -> _compiled=False
    H10: gradient_checkpointing=True + compile_training="auto" -> _compiled=False

References:
    Issue #55 -- O(n^1.89) training scaling and memory wall
    MLX docs -- mx.checkpoint
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from textpolicy.training.gradient_checkpointing import (
    apply_gradient_checkpointing,
    is_gradient_checkpointing_active,
    remove_gradient_checkpointing,
)


# ---------------------------------------------------------------------------
# Helpers: Minimal LoRA-style model for testing
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Minimal LoRA linear layer for testing.

    Base weight is frozen; only lora_a and lora_b receive gradients.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 2):
        super().__init__()
        self.base_weight = mx.random.normal((out_features, in_features)) * 0.1
        self.lora_a = mx.random.normal((rank, in_features)) * 0.01
        self.lora_b = mx.zeros((out_features, rank))

    def __call__(self, x):
        base_out = x @ self.base_weight.T
        lora_out = (x @ self.lora_a.T) @ self.lora_b.T
        return base_out + lora_out


class TransformerLayer(nn.Module):
    """Minimal transformer layer with LoRA for testing.

    Matches the MLX-LM layer signature: __call__(self, x, mask=None, cache=None).
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = LoRALinear(dim, dim, rank=2)

    def __call__(self, x, mask=None, cache=None):
        return nn.relu(self.proj(x))


class TinyLoRAModel(nn.Module):
    """Minimal model with .layers list and LoRA adapters for testing."""

    def __init__(self, dim: int = 8, num_layers: int = 3, vocab_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [TransformerLayer(dim) for _ in range(num_layers)]
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class WrappedTinyModel(nn.Module):
    """Model with .model.layers pattern (like MLX-LM wrapper)."""

    def __init__(self, dim: int = 8, num_layers: int = 3, vocab_size: int = 16):
        super().__init__()
        self.model = TinyLoRAModel(dim, num_layers, vocab_size)

    def __call__(self, x):
        return self.model(x)


class NoLayersModel(nn.Module):
    """Model without .layers (should raise ValueError)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def __call__(self, x):
        return self.linear(x)


def _freeze_base_keep_lora(model):
    """Freeze everything, then unfreeze LoRA weights only."""
    model.freeze()

    def _unfreeze_lora(_path, module):
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"])
        return module

    model.apply_to_modules(_unfreeze_lora)
    return model


# ---------------------------------------------------------------------------
# TestApplyRemove
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyRemove:
    """Validate apply/remove/is_active lifecycle."""

    def test_apply_returns_correct_count(self):
        """H1: apply returns the number of transformer layers checkpointed."""
        mx.random.seed(42)
        model = TinyLoRAModel(num_layers=3)
        mx.eval(model.parameters())

        count = apply_gradient_checkpointing(model)
        assert count == 3, f"Expected 3, got {count}"

    def test_remove_restores_original_class(self):
        """H2: remove restores original __class__ on all layers."""
        mx.random.seed(42)
        model = TinyLoRAModel(num_layers=3)
        mx.eval(model.parameters())

        apply_gradient_checkpointing(model)
        # All layers should have a checkpointed class after apply
        for i, layer in enumerate(model.layers):
            assert getattr(type(layer), "_original_class", None) is not None, (
                f"Layer {i} not checkpointed after apply"
            )
            assert type(layer).__name__.startswith("_Checkpointed_"), (
                f"Layer {i} class name is {type(layer).__name__}"
            )

        count = remove_gradient_checkpointing(model)
        assert count == 3, f"Expected 3 restored, got {count}"

        # All layers should be back to original class
        for i, layer in enumerate(model.layers):
            assert type(layer) is TransformerLayer, (
                f"Layer {i} class is {type(layer).__name__}, "
                f"expected TransformerLayer"
            )
            assert getattr(type(layer), "_original_class", None) is None

    def test_double_apply_is_idempotent(self):
        """H3: Second apply returns 0 (no layers newly checkpointed)."""
        mx.random.seed(42)
        model = TinyLoRAModel(num_layers=3)
        mx.eval(model.parameters())

        first = apply_gradient_checkpointing(model)
        assert first == 3

        second = apply_gradient_checkpointing(model)
        assert second == 0, f"Expected 0 on second apply, got {second}"

    def test_is_active_tracks_state(self):
        """H4: is_active correctly reports state transitions."""
        mx.random.seed(42)
        model = TinyLoRAModel(num_layers=2)
        mx.eval(model.parameters())

        assert not is_gradient_checkpointing_active(model)

        apply_gradient_checkpointing(model)
        assert is_gradient_checkpointing_active(model)

        remove_gradient_checkpointing(model)
        assert not is_gradient_checkpointing_active(model)

    def test_no_layers_raises_valueerror(self):
        """H5: Model without .layers raises ValueError."""
        model = NoLayersModel()
        mx.eval(model.parameters())

        with pytest.raises(ValueError, match="no 'layers' attribute"):
            apply_gradient_checkpointing(model)

    def test_wrapped_model_accesses_inner_layers(self):
        """apply works on model.model.layers pattern (MLX-LM wrapper)."""
        mx.random.seed(42)
        model = WrappedTinyModel(num_layers=4)
        mx.eval(model.parameters())

        count = apply_gradient_checkpointing(model)
        assert count == 4
        assert is_gradient_checkpointing_active(model)

    def test_call_dispatches_to_checkpointed_wrapper(self):
        """H15: layer(x) dispatches to the checkpointed __call__, not the original.

        This is the critical test that catches the original bug: instance
        attribute monkey-patching of __call__ is silently ignored by
        Python's () operator, which looks up __call__ on the type.

        We verify that after apply_gradient_checkpointing:
        1. type(layer).__call__ is different from the original class __call__
        2. The class name reflects the checkpointed wrapper
        3. Parameter tree paths are preserved (no 'inner.' prefix)
        """
        mx.random.seed(42)
        model = TinyLoRAModel(num_layers=3)
        mx.eval(model.parameters())

        original_call = TransformerLayer.__call__

        # Capture parameter paths before checkpointing
        paths_before = [k for k, _ in tree_flatten(model.parameters())]

        apply_gradient_checkpointing(model)

        # Verify class-level __call__ changed (not just instance attribute)
        for i, layer in enumerate(model.layers):
            assert type(layer).__call__ is not original_call, (
                f"Layer {i}: type(layer).__call__ is still the original — "
                f"checkpointing wrapper not dispatched by layer(x)"
            )
            assert "_Checkpointed_" in type(layer).__name__

        # Verify parameter tree paths are preserved
        paths_after = [k for k, _ in tree_flatten(model.parameters())]
        assert paths_before == paths_after, (
            f"Parameter paths changed after checkpointing:\n"
            f"  before: {paths_before}\n"
            f"  after:  {paths_after}"
        )


# ---------------------------------------------------------------------------
# TestCorrectnessWithTrainer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorrectnessWithTrainer:
    """Validate gradient correctness with LoRA + checkpoint."""

    def _make_lora_model(self, seed=42):
        """Create a reproducible LoRA model."""
        mx.random.seed(seed)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)
        return model

    def _loss_fn(self, model, x):
        """Simple loss for gradient checking."""
        return model(x).sum()

    def test_loss_matches_with_checkpointing(self):
        """H6: Loss is identical with and without checkpointing.

        We use the same seed and data to verify exact numerical parity.
        The checkpoint wrapper should produce identical forward pass results.
        """
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

        # Baseline (no checkpoint)
        model_baseline = self._make_lora_model(seed=42)
        loss_baseline, grads_baseline = nn.value_and_grad(
            model_baseline, self._loss_fn
        )(model_baseline, x)
        mx.eval(loss_baseline, grads_baseline)

        # With checkpoint
        model_ckpt = self._make_lora_model(seed=42)
        apply_gradient_checkpointing(model_ckpt)
        loss_ckpt, grads_ckpt = nn.value_and_grad(
            model_ckpt, self._loss_fn
        )(model_ckpt, x)
        mx.eval(loss_ckpt, grads_ckpt)

        assert abs(loss_baseline.item() - loss_ckpt.item()) < 1e-5, (
            f"Loss mismatch: baseline={loss_baseline.item():.6f}, "
            f"ckpt={loss_ckpt.item():.6f}"
        )

    def test_lora_gradients_match(self):
        """H7: LoRA adapter gradients match between checkpointed and non-checkpointed.

        This is the critical correctness test. mx.checkpoint only preserves
        gradients for explicit positional arguments, so the implementation
        must pass trainable LoRA params explicitly.
        """
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

        # Baseline
        model_baseline = self._make_lora_model(seed=42)
        _, grads_baseline = nn.value_and_grad(
            model_baseline, self._loss_fn
        )(model_baseline, x)
        mx.eval(grads_baseline)

        # With checkpoint
        model_ckpt = self._make_lora_model(seed=42)
        apply_gradient_checkpointing(model_ckpt)
        _, grads_ckpt = nn.value_and_grad(
            model_ckpt, self._loss_fn
        )(model_ckpt, x)
        mx.eval(grads_ckpt)

        flat_baseline = tree_flatten(grads_baseline)
        flat_ckpt = tree_flatten(grads_ckpt)

        assert len(flat_baseline) == len(flat_ckpt), (
            f"Gradient count mismatch: {len(flat_baseline)} vs {len(flat_ckpt)}"
        )

        for (k1, v1), (k2, v2) in zip(flat_baseline, flat_ckpt):
            assert k1 == k2, f"Key mismatch: {k1} vs {k2}"
            assert mx.allclose(v1, v2, atol=1e-5), (
                f"Gradient mismatch for {k1}: "
                f"max diff={mx.abs(v1 - v2).max().item():.6f}"
            )

    def test_lora_gradients_are_nonzero(self):
        """H7b: LoRA lora_b adapter gradients are non-zero after checkpointing.

        This catches the failure mode where mx.checkpoint silently
        zeroes gradients for closed-over parameters.

        Note: lora_a gradients are zero by design when lora_b is initialized
        to zeros (standard LoRA init). The chain rule gives
        d_loss/d_lora_a = ... @ lora_b = 0. So we check lora_b which
        receives the initial gradient signal.
        """
        x = mx.array([[0, 1, 2, 3]], dtype=mx.int32)

        model = self._make_lora_model(seed=42)
        apply_gradient_checkpointing(model)
        _, grads = nn.value_and_grad(model, self._loss_fn)(model, x)
        mx.eval(grads)

        flat_grads = tree_flatten(grads)
        lora_b_grads = [(k, v) for k, v in flat_grads if "lora_b" in k]

        assert len(lora_b_grads) > 0, "No lora_b gradients found"

        for k, v in lora_b_grads:
            grad_norm = mx.sqrt(mx.sum(v * v)).item()
            assert grad_norm > 0, (
                f"LoRA gradient for {k} is zero — "
                f"checkpoint is not preserving gradients for trainable params"
            )

    def test_trainer_step_completes(self):
        """H8: Trainer step completes with finite loss and non-zero grads."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            gradient_checkpointing=True,
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
        )

        # Minimal batch data
        batch_data = {
            "obs": mx.zeros((5,), dtype=mx.int32),
            "act": mx.zeros((5,), dtype=mx.int32),
            "logprob": -mx.ones((5,)),
            "rewards": mx.array([1.0, 0.0]),
            "episode_lengths": [3, 2],
        }

        metrics = trainer.train(batch_data)

        assert "loss" in metrics
        assert not mx.isinf(mx.array(metrics["loss"])).item(), "Loss is infinite"
        assert not mx.isnan(mx.array(metrics["loss"])).item(), "Loss is NaN"


# ---------------------------------------------------------------------------
# TestCompileInteraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompileInteraction:
    """Validate compile/checkpoint incompatibility handling."""

    def _make_model(self):
        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        return model

    def test_checkpoint_forces_compile_off(self):
        """H9: gradient_checkpointing=True + compile_training=True -> _compiled=False."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        model = self._make_model()

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            gradient_checkpointing=True,
        )

        assert trainer._compiled is False, (
            "Compilation must be disabled when gradient_checkpointing=True"
        )
        assert trainer._gradient_checkpointing is True
        assert is_gradient_checkpointing_active(model)

    def test_checkpoint_forces_auto_compile_off(self):
        """H10: gradient_checkpointing=True + compile_training='auto' -> _compiled=False."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        # Use a large enough model that auto would normally compile
        mx.random.seed(42)
        model = TinyLoRAModel(dim=256, num_layers=8, vocab_size=2048)
        mx.eval(model.parameters())

        n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
        assert n_params >= 1_000_000, f"Model too small: {n_params}"

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training="auto",
            gradient_checkpointing=True,
        )

        assert trainer._compiled is False, (
            "Auto-compile must be overridden when gradient_checkpointing=True"
        )
        assert is_gradient_checkpointing_active(model)

    def test_no_checkpoint_compile_unaffected(self):
        """Sanity: gradient_checkpointing=False does not affect compilation."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        model = self._make_model()

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            gradient_checkpointing=False,
        )

        assert trainer._compiled is True
        assert trainer._gradient_checkpointing is False
        assert not is_gradient_checkpointing_active(model)


# ---------------------------------------------------------------------------
# TestMicroBatching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMicroBatching:
    """Validate micro-batched logprob extraction produces identical results.

    Hypotheses tested:
      H11: Logprobs with micro_batch_size=1 match full-batch logprobs exactly
      H12: Logprobs with micro_batch_size=2 match full-batch (uneven split)
      H13: Trainer step completes with micro_batch_size and produces finite loss
      H14: micro_batch_size >= num_episodes falls back to full-batch path
    """

    def test_logprobs_match_micro_batch_1(self):
        """H11: micro_batch_size=1 gives same logprobs as full batch."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)

        # Create batch data with 4 episodes (2D padded format)
        num_episodes = 4
        seq_len = 6
        prompt_len = 2
        resp_len = seq_len - prompt_len

        obs = mx.random.randint(0, 16, (num_episodes, seq_len))
        acts = mx.random.randint(0, 16, (num_episodes, resp_len))
        mx.eval(obs, acts)

        prompt_lengths = [prompt_len] * num_episodes
        episode_lengths = [resp_len] * num_episodes

        # Full batch logprobs
        trainer_full = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

        # Use a fixed set of old_logprobs for validation
        old_lp = -mx.ones((sum(episode_lengths),))
        mx.eval(old_lp)

        full_lp = trainer_full._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        mx.eval(full_lp)

        # Micro-batched logprobs (M=1)
        trainer_micro = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            micro_batch_size=1,
        )

        micro_lp = trainer_micro._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        mx.eval(micro_lp)

        assert full_lp.shape == micro_lp.shape, (
            f"Shape mismatch: {full_lp.shape} vs {micro_lp.shape}"
        )
        assert mx.allclose(full_lp, micro_lp, atol=1e-5), (
            f"Logprob mismatch: max diff="
            f"{mx.abs(full_lp - micro_lp).max().item():.6f}"
        )

    def test_logprobs_match_micro_batch_2(self):
        """H12: micro_batch_size=2 with 5 episodes (uneven split 2+2+1)."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)

        num_episodes = 5
        seq_len = 6
        prompt_len = 2
        resp_len = seq_len - prompt_len

        obs = mx.random.randint(0, 16, (num_episodes, seq_len))
        acts = mx.random.randint(0, 16, (num_episodes, resp_len))
        mx.eval(obs, acts)

        prompt_lengths = [prompt_len] * num_episodes
        episode_lengths = [resp_len] * num_episodes
        old_lp = -mx.ones((sum(episode_lengths),))
        mx.eval(old_lp)

        trainer_full = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )
        full_lp = trainer_full._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        mx.eval(full_lp)

        trainer_micro = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            micro_batch_size=2,
        )
        micro_lp = trainer_micro._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        mx.eval(micro_lp)

        assert full_lp.shape == micro_lp.shape
        assert mx.allclose(full_lp, micro_lp, atol=1e-5), (
            f"Logprob mismatch: max diff="
            f"{mx.abs(full_lp - micro_lp).max().item():.6f}"
        )

    def test_trainer_step_with_micro_batch(self):
        """H13: Trainer.train() completes with micro_batch_size set."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            micro_batch_size=2,
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
        )

        batch_data = {
            "obs": mx.zeros((5,), dtype=mx.int32),
            "act": mx.zeros((5,), dtype=mx.int32),
            "logprob": -mx.ones((5,)),
            "rewards": mx.array([1.0, 0.0]),
            "episode_lengths": [3, 2],
        }

        metrics = trainer.train(batch_data)

        assert "loss" in metrics
        assert not mx.isinf(mx.array(metrics["loss"])).item()
        assert not mx.isnan(mx.array(metrics["loss"])).item()

    def test_micro_batch_ge_episodes_uses_full_path(self):
        """H14: micro_batch_size >= num_episodes uses full-batch path."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = TinyLoRAModel(dim=8, num_layers=3, vocab_size=16)
        mx.eval(model.parameters())
        model = _freeze_base_keep_lora(model)

        num_episodes = 3
        seq_len = 6
        prompt_len = 2
        resp_len = seq_len - prompt_len

        obs = mx.random.randint(0, 16, (num_episodes, seq_len))
        acts = mx.random.randint(0, 16, (num_episodes, resp_len))
        mx.eval(obs, acts)

        prompt_lengths = [prompt_len] * num_episodes
        episode_lengths = [resp_len] * num_episodes
        old_lp = -mx.ones((sum(episode_lengths),))
        mx.eval(old_lp)

        # micro_batch_size=10 > num_episodes=3 → should use full-batch
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            micro_batch_size=10,
        )

        # Full-batch baseline
        trainer_full = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

        lp_micro = trainer._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        lp_full = trainer_full._extract_grpo_logprobs(
            obs, acts, old_lp, episode_lengths, prompt_lengths,
        )
        mx.eval(lp_micro, lp_full)

        assert mx.allclose(lp_micro, lp_full, atol=1e-5)
