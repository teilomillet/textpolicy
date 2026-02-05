"""
Training Pipeline Integration Tests

These tests verify that the complete training pipeline works correctly,
including gradient computation and parameter updates.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


@pytest.mark.integration
class TestTrainingPipeline:
    """Test the complete training pipeline with gradient updates."""

    def test_grpo_training_step_produces_finite_loss(self):
        """Test that GRPO training produces finite loss values."""
        from textpolicy.algorithms import grpo

        # Create simple model
        class TinyLM(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())

        # Prepare batch data (mimics what data_selector returns)
        old_logprobs = mx.array([-2.5, -3.1, -2.8, -2.9, -3.0])
        observations = mx.array([10, 20, 30, 40, 50])
        actions = mx.array([15, 25, 35, 45, 55])
        rewards = mx.array([1.0, 0.5, -0.5, 0.8, 0.2])

        # Compute new logprobs from model
        logits = model(observations)
        log_probs = nn.log_softmax(logits, axis=-1)
        new_logprobs = mx.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

        # Compute GRPO loss
        advantages = grpo.compute_advantages(rewards)
        loss = grpo.policy_loss(old_logprobs, new_logprobs, advantages, clip_ratio=0.2)

        mx.eval(loss)
        assert not mx.isnan(loss), "Loss should not be NaN"
        assert not mx.isinf(loss), "Loss should not be infinite"

    def test_grpo_gradients_flow(self):
        """Test that gradients flow through the GRPO loss."""
        from textpolicy.algorithms import grpo

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())

        def loss_fn(model):
            observations = mx.array([10, 20, 30, 40, 50])
            actions = mx.array([15, 25, 35, 45, 55])
            old_logprobs = mx.array([-2.5, -3.1, -2.8, -2.9, -3.0])
            rewards = mx.array([1.0, 0.5, -0.5, 0.8, 0.2])

            logits = model(observations)
            log_probs = nn.log_softmax(logits, axis=-1)
            new_logprobs = mx.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

            advantages = grpo.compute_advantages(rewards)
            return grpo.policy_loss(old_logprobs, new_logprobs, advantages, clip_ratio=0.2)

        # Compute gradients
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad(model)

        mx.eval(loss, grads)

        assert not mx.isnan(loss), "Loss should not be NaN"

        # Check that at least some gradients are non-zero
        has_nonzero_grad = False
        for name, grad in grads.items():
            if isinstance(grad, dict):
                for subname, subgrad in grad.items():
                    if mx.any(subgrad != 0):
                        has_nonzero_grad = True
                        break
            elif mx.any(grad != 0):
                has_nonzero_grad = True
                break

        assert has_nonzero_grad, "Should have at least some non-zero gradients"

    def test_optimizer_updates_parameters(self):
        """Test that optimizer actually updates model parameters."""
        from textpolicy.algorithms import grpo

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        model = TinyLM()
        optimizer = optim.Adam(learning_rate=0.01)
        mx.eval(model.parameters())

        # Store initial parameter values
        initial_weight = mx.array(model.proj.weight)  # Create a copy

        def loss_fn(model):
            observations = mx.array([10, 20, 30])
            actions = mx.array([15, 25, 35])
            old_logprobs = mx.array([-2.5, -3.1, -2.8])
            rewards = mx.array([1.0, 0.5, -0.5])

            logits = model(observations)
            log_probs = nn.log_softmax(logits, axis=-1)
            new_logprobs = mx.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

            advantages = grpo.compute_advantages(rewards)
            return grpo.policy_loss(old_logprobs, new_logprobs, advantages)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Run a few training steps
        for _ in range(3):
            loss, grads = loss_and_grad(model)
            optimizer.update(model, grads)
            mx.eval(model.parameters())

        # Check parameters changed
        final_weight = model.proj.weight
        params_changed = not mx.allclose(initial_weight, final_weight)

        assert params_changed, "Parameters should change after optimization"

    def test_gspo_training_step_produces_finite_loss(self):
        """Test that GSPO training produces finite loss values."""
        from textpolicy.algorithms import gspo

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=100, hidden=32):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())

        # Prepare batch data
        old_logprobs = mx.array([-2.5, -3.1, -2.8, -2.9, -3.0])
        observations = mx.array([10, 20, 30, 40, 50])
        actions = mx.array([15, 25, 35, 45, 55])

        # Compute new logprobs from model
        logits = model(observations)
        log_probs = nn.log_softmax(logits, axis=-1)
        new_logprobs = mx.take_along_axis(log_probs, actions[:, None], axis=-1).squeeze(-1)

        # GSPO needs sequence-level advantages
        sequence_lengths = [2, 3]  # 2 sequences
        advantages = mx.array([0.5, -0.3])  # Per-sequence

        # Test all GSPO variants
        for variant in ['sequence', 'hybrid', 'token']:
            loss = gspo.gspo_policy_loss(
                old_logprobs, new_logprobs, advantages, sequence_lengths,
                variant=variant, clip_ratio=0.2
            )
            mx.eval(loss)
            assert not mx.isnan(loss), f"GSPO {variant} loss should not be NaN"
            assert not mx.isinf(loss), f"GSPO {variant} loss should not be infinite"


@pytest.mark.integration
class TestCompiledTraining:
    """Test that @mx.compile works correctly with training."""

    def test_compiled_function_works(self):
        """Test that @mx.compile decorator works with loss functions."""
        @mx.compile
        def compiled_loss(x, y):
            return mx.mean((x - y) ** 2)

        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([1.1, 2.1, 3.1])

        loss = compiled_loss(x, y)
        mx.eval(loss)

        assert not mx.isnan(loss), "Compiled loss should not be NaN"
        assert float(loss) > 0, "Loss should be positive"

    def test_trainer_compiled_loss_function(self):
        """Test that Trainer's compiled loss function works correctly.

        This mimics how Trainer.py:90 compiles the loss function.
        """
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            def __call__(self, x):
                return self.linear(x)

        model = TinyLM()
        mx.eval(model.parameters())

        # Create trainer with compilation enabled
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True
        )

        # Verify the compiled function was created
        assert trainer.loss_and_grad_fn is not None, "Compiled loss function should exist"
