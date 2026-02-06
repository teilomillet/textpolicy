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


@pytest.mark.unit
class TestAdvantageExpansion:
    """Validate that _loss_fn expands advantages using real episode lengths."""

    def _make_trainer(self):
        """Create a minimal Trainer for testing _loss_fn."""
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
        return Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

    def test_variable_length_episodes_use_real_lengths(self):
        """Regression: advantage expansion must use batch_data['episode_lengths'].

        With 2 episodes of lengths [4, 1] and advantages [0.5, -0.5]:
          correct (real lengths): [0.5, 0.5, 0.5, 0.5, -0.5]
          wrong   (even dist):    [0.5, 0.5, 0.5, -0.5, -0.5]

        Previously _loss_fn used even distribution (total_tokens //
        num_episodes), which misassigned credit at episode boundaries
        for variable-length episodes.
        """
        trainer = self._make_trainer()

        # Manually call _expand_advantages via the Trainer to verify
        # the public interface.  We test the expansion logic directly
        # since _loss_fn requires a full model forward pass.
        advantages = mx.array([0.5, -0.5])
        real_lengths = [4, 1]

        expanded = trainer._expand_advantages(advantages, real_lengths)
        mx.eval(expanded)

        expected = [0.5, 0.5, 0.5, 0.5, -0.5]
        assert expanded.tolist() == expected, (
            f"Expected {expected}, got {expanded.tolist()}. "
            f"Advantage expansion must use real episode lengths, not even distribution."
        )

    def test_even_distribution_fallback_without_episode_lengths(self):
        """When episode_lengths is absent, even distribution is used as fallback."""
        trainer = self._make_trainer()

        advantages = mx.array([0.5, -0.5])
        total_tokens = 6
        num_episodes = 2

        # Simulate the fallback path
        base_length = total_tokens // num_episodes
        remainder = total_tokens % num_episodes
        action_lengths = [base_length + (1 if i < remainder else 0) for i in range(num_episodes)]

        expanded = trainer._expand_advantages(advantages, action_lengths)
        mx.eval(expanded)

        # Even split: [3, 3] → [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
        expected = [0.5, 0.5, 0.5, -0.5, -0.5, -0.5]
        assert expanded.tolist() == expected

    def test_mismatched_episode_count_raises(self):
        """episode_lengths with wrong count raises clear ValueError."""
        trainer = self._make_trainer()

        # 3 episodes worth of rewards → 3 advantages, but episode_lengths has 2
        batch_data = {
            'obs': mx.zeros(7),
            'act': mx.zeros(7, dtype=mx.int32),
            'logprob': mx.zeros(7),
            'rewards': mx.array([1.0, 0.5, -0.5]),
            'episode_lengths': [4, 3],  # only 2 entries, but 3 episodes
        }
        with pytest.raises(ValueError, match="does not match num_episodes"):
            trainer._loss_fn(batch_data)

    def test_mismatched_total_tokens_raises(self):
        """episode_lengths that don't sum to total_tokens raises clear ValueError."""
        trainer = self._make_trainer()

        # episode_lengths sum to 6, but logprobs have 7 tokens
        batch_data = {
            'obs': mx.zeros(7),
            'act': mx.zeros(7, dtype=mx.int32),
            'logprob': mx.zeros(7),
            'rewards': mx.array([1.0, 0.5]),
            'episode_lengths': [3, 3],  # sum=6, but 7 tokens
        }
        with pytest.raises(ValueError, match="does not match total_tokens"):
            trainer._loss_fn(batch_data)


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
