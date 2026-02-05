"""
MLX and mlx-lm Compatibility Tests

These tests verify that textpolicy works correctly with MLX 0.30.x and mlx-lm 0.30.x.
They cover the core APIs used throughout the codebase.

Run with: pytest tests/test_mlx_compatibility.py -v
"""

import pytest
import mlx.core as mx
import mlx.nn as nn


@pytest.mark.unit
class TestMLXCoreAPIs:
    """Test core MLX APIs used in textpolicy."""

    def test_mx_compile_decorator(self):
        """Test @mx.compile decorator works (used in grpo.py, gspo.py, trainer.py)."""
        @mx.compile
        def compiled_fn(x, y):
            return x + y

        result = compiled_fn(mx.array([1.0]), mx.array([2.0]))
        assert float(result[0]) == 3.0

    def test_mx_compile_with_value_and_grad(self):
        """Test mx.compile with nn.value_and_grad (used in trainer.py:90)."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 2)
            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        def loss_fn(model, x, y):
            pred = model(x)
            return mx.mean((pred - y) ** 2)

        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 4))
        y = mx.random.normal((2, 2))
        loss, grads = loss_and_grad_fn(model, x, y)

        assert not mx.isnan(loss)
        assert isinstance(grads, dict)

    def test_array_operations(self):
        """Test array operations used in GRPO/GSPO algorithms."""
        # Ratio computation (grpo.py, gspo.py)
        old_lp = mx.array([-1.0, -1.2, -0.8])
        new_lp = mx.array([-1.1, -1.0, -0.9])
        ratios = mx.exp(new_lp - old_lp)

        assert ratios.shape == (3,)
        assert all(not mx.isnan(r) for r in ratios)

        # Clipping (PPO-style)
        clipped = mx.clip(ratios, 0.8, 1.2)
        # Use small epsilon for float comparison
        assert all(0.8 - 1e-6 <= float(c) <= 1.2 + 1e-6 for c in clipped)

    def test_array_slicing_with_python_ints(self):
        """Test array slicing with Python integers (used in gspo.py)."""
        arr = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # This pattern is used in compute_sequence_importance_weights
        current_idx = 0
        for seq_len in [2, 3]:
            result = arr[current_idx:current_idx + seq_len]
            assert result.shape[0] == seq_len
            current_idx += seq_len


@pytest.mark.unit
class TestMLXLMAPIs:
    """Test mlx-lm APIs used in textpolicy."""

    def test_mlx_lm_imports(self):
        """Test mlx_lm core imports (used in mlx_generation.py)."""
        from mlx_lm import load, generate
        assert callable(load)
        assert callable(generate)

    def test_sample_utils_imports(self):
        """Test sample_utils imports (used in mlx_generation.py:27)."""
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        assert callable(make_sampler)
        assert callable(make_logits_processors)

    def test_make_sampler_signature(self):
        """Test make_sampler accepts expected parameters."""
        from mlx_lm.sample_utils import make_sampler

        # These params are used in mlx_generation.py:77-81
        sampler = make_sampler(
            temp=0.7,
            top_p=0.9,
            min_p=0.0,
            min_tokens_to_keep=2
        )
        assert sampler is not None

    def test_quantize_model_signature(self):
        """Test quantize_model has expected parameters (used in lora.py:335)."""
        from mlx_lm.utils import quantize_model
        import inspect

        sig = inspect.signature(quantize_model)
        params = list(sig.parameters.keys())

        # These params are used in lora.py:342-348
        expected = ['model', 'config', 'q_group_size', 'q_bits', 'quant_predicate']
        for p in expected:
            assert p in params, f"Expected param '{p}' not found in quantize_model"


@pytest.mark.unit
@pytest.mark.algorithm
class TestGRPOWithMLX:
    """Test GRPO algorithms with current MLX version."""

    def test_compute_advantages(self):
        """Test GRPO compute_advantages."""
        from textpolicy.algorithms import grpo

        rewards = mx.array([1.0, 0.5, -0.5, 0.8, 0.2])
        advantages = grpo.compute_advantages(rewards)

        assert advantages.shape == rewards.shape
        # Group-relative: mean should be ~0
        assert abs(float(mx.mean(advantages))) < 1e-5

    def test_compute_advantages_compiled(self):
        """Test compiled version of compute_advantages."""
        from textpolicy.algorithms import grpo

        rewards = mx.array([1.0, 0.5, -0.5, 0.8])
        advantages = grpo.compute_advantages_compiled(rewards)

        assert advantages.shape == rewards.shape

    def test_policy_loss(self):
        """Test GRPO policy_loss computation."""
        from textpolicy.algorithms import grpo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.6, 0.1, -0.9, 0.4, -0.2])

        loss = grpo.policy_loss(old_lp, new_lp, advantages, clip_ratio=0.2)

        assert not mx.isnan(loss)
        assert loss.shape == ()  # Scalar

    def test_entropy_bonus(self):
        """Test entropy bonus computation."""
        from textpolicy.algorithms import grpo

        logprobs = mx.array([-1.0, -2.0, -0.5, -1.5])
        entropy = grpo.entropy_bonus(logprobs, coefficient=0.01)

        assert not mx.isnan(entropy)


@pytest.mark.unit
@pytest.mark.algorithm
class TestGSPOWithMLX:
    """Test GSPO algorithms with current MLX version."""

    def test_compute_sequence_importance_weights(self):
        """Test sequence-level importance weights."""
        from textpolicy.algorithms import gspo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]

        weights = gspo.compute_sequence_importance_weights(
            old_lp, new_lp, seq_lens, clip_ratio=0.2
        )

        assert len(weights) == len(seq_lens)
        assert all(not mx.isnan(w) for w in weights)

    def test_compute_hybrid_importance_weights(self):
        """Test hybrid importance weights."""
        from textpolicy.algorithms import gspo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]

        weights = gspo.compute_hybrid_importance_weights(
            old_lp, new_lp, seq_lens, alpha=0.5, beta=0.5
        )

        assert len(weights) == sum(seq_lens)  # Token-level weights

    def test_gspo_policy_loss_sequence_variant(self):
        """Test GSPO policy loss with sequence variant."""
        from textpolicy.algorithms import gspo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]
        advantages = mx.array([0.5, -0.3])

        loss = gspo.gspo_policy_loss(
            old_lp, new_lp, advantages, seq_lens,
            variant='sequence', clip_ratio=0.2
        )

        assert not mx.isnan(loss)
        assert loss.shape == ()

    def test_gspo_policy_loss_hybrid_variant(self):
        """Test GSPO policy loss with hybrid variant."""
        from textpolicy.algorithms import gspo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]
        advantages = mx.array([0.5, -0.3])

        loss = gspo.gspo_policy_loss(
            old_lp, new_lp, advantages, seq_lens,
            variant='hybrid', clip_ratio=0.2
        )

        assert not mx.isnan(loss)

    def test_gspo_policy_loss_token_variant(self):
        """Test GSPO policy loss with token variant (GRPO fallback)."""
        from textpolicy.algorithms import gspo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        seq_lens = [2, 3]
        advantages = mx.array([0.5, -0.3])

        loss = gspo.gspo_policy_loss(
            old_lp, new_lp, advantages, seq_lens,
            variant='token', clip_ratio=0.2
        )

        assert not mx.isnan(loss)


@pytest.mark.unit
class TestTrainerWithMLX:
    """Test Trainer compilation with MLX."""

    def test_trainer_compiles_loss_function(self):
        """Test that Trainer correctly compiles the loss function."""
        import mlx.optimizers as optim
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            def __call__(self, x):
                return self.linear(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        def loss_fn(model, batch):
            obs = batch.get('observations', mx.zeros((1, 10)))
            return mx.mean(model(obs) ** 2)

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True
        )

        # Verify compilation happened
        assert trainer.loss_and_grad_fn is not None


@pytest.mark.unit
class TestLoRAWithMLX:
    """Test LoRA functions with MLX."""

    def test_lora_functions_importable(self):
        """Test all LoRA functions can be imported."""
        from textpolicy.generation.lora import (
            apply_lora,
            freeze_base,
            extract_params,
            merge_weights,
            create_lora_setup,
            create_qlora_setup,
            apply_quantization_to_model,
            compute_lora_memory_savings
        )

        # All should be callable
        assert callable(apply_lora)
        assert callable(freeze_base)
        assert callable(extract_params)
        assert callable(merge_weights)
        assert callable(create_lora_setup)
        assert callable(create_qlora_setup)
        assert callable(apply_quantization_to_model)
        assert callable(compute_lora_memory_savings)
