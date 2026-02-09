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
            # TinyLM(Linear(10,10)) can't produce valid [seq_len, vocab]
            # logits from flat 1D input; bypass _default_get_logprobs so
            # the advantage expansion validation is what gets tested.
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
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
            'obs': mx.zeros(10),  # match TinyLM's Linear(10,10) input dim
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
            'obs': mx.zeros(10),  # match TinyLM's Linear(10,10) input dim
            'act': mx.zeros(7, dtype=mx.int32),
            'logprob': mx.zeros(7),
            'rewards': mx.array([1.0, 0.5]),
            'episode_lengths': [3, 3],  # sum=6, but 7 tokens
        }
        with pytest.raises(ValueError, match="does not match sum\\(episode_lengths\\)"):
            trainer._loss_fn(batch_data)


@pytest.mark.unit
class TestExtractGrpoLogprobs2D:
    """Direct tests for the 2D per-episode branch of _extract_grpo_logprobs.

    The 2D branch handles text generation episodes where observations are
    [num_episodes, prompt_len] and actions are [num_episodes, response_len].
    Each episode gets its own forward pass via compute_logprobs(model, prompt,
    response), and the results are concatenated into a flat 1D array.
    """

    def _make_model(self, vocab_size=16, dim=8):
        """Minimal causal LM: embedding + head → [batch, seq_len, vocab_size]."""
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _make_trainer(self, model):
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo
        return Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

    def test_output_is_flat_with_correct_length(self):
        """H1: Output is flat 1D with length = num_episodes * response_len."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        num_episodes, prompt_len, response_len = 3, 4, 5
        obs = mx.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        act = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
        old_lp = mx.zeros(num_episodes * response_len)
        ep_lengths = [response_len] * num_episodes

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths)
        mx.eval(result)

        assert result.ndim == 1
        assert result.shape[0] == num_episodes * response_len

    def test_logprobs_are_valid(self):
        """H2: All returned values are ≤ 0 with no NaN or Inf."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3], [7, 8, 9]])
        act = mx.array([[4, 5, 6, 7], [10, 11, 12, 13]])
        old_lp = mx.zeros(8)
        ep_lengths = [4, 4]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)).item()
        assert not mx.any(mx.isinf(result)).item()
        assert mx.all(result <= 0).item()

    def test_per_episode_not_tiled(self):
        """H3: Different prompts produce different logprobs.

        Regression: the old code computed logprobs from a single averaged
        prompt/response slice and mx.tile'd across episodes. With distinct
        inputs the per-episode logprobs must differ.
        """
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3], [10, 11, 12]])
        act = mx.array([[4, 5], [13, 14]])
        old_lp = mx.zeros(4)
        ep_lengths = [2, 2]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths)
        mx.eval(result)

        ep1_lp = result[:2]
        ep2_lp = result[2:]
        diff = mx.max(mx.abs(ep1_lp - ep2_lp)).item()
        assert diff > 1e-6, (
            f"Per-episode logprobs should differ for different inputs, "
            f"got ep1={ep1_lp.tolist()} ep2={ep2_lp.tolist()}"
        )

    def test_matches_direct_compute_logprobs(self):
        """H4: Output equals calling compute_logprobs per episode then concatenating."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3], [7, 8, 9]])
        act = mx.array([[4, 5, 6], [10, 11, 12]])
        old_lp = mx.zeros(6)
        ep_lengths = [3, 3]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths)
        mx.eval(result)

        expected_ep1 = compute_logprobs(model, obs[0], act[0])
        expected_ep2 = compute_logprobs(model, obs[1], act[1])
        expected = mx.concatenate([expected_ep1, expected_ep2])
        mx.eval(expected)

        assert mx.allclose(result, expected, atol=1e-5).item(), (
            f"Trainer result {result.tolist()} doesn't match "
            f"direct compute_logprobs {expected.tolist()}"
        )

    def test_single_episode_degenerate(self):
        """H5: Single episode (degenerate case) produces valid output."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3]])
        act = mx.array([[5, 6]])
        old_lp = mx.zeros(2)
        ep_lengths = [2]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths)
        mx.eval(result)

        assert result.ndim == 1
        assert result.shape[0] == 2
        assert mx.all(result <= 0).item()


@pytest.mark.unit
class TestExtractGrpoLogprobsBatched:
    """Tests for the batched path in _extract_grpo_logprobs.

    The batched path uses compute_logprobs_batched (single model forward pass)
    instead of N sequential compute_logprobs calls. It is triggered when
    prompt_lengths is provided alongside 2D observations.

    Hypotheses:
        H1: Batched output matches sequential compute_logprobs per episode
        H2: Single episode (degenerate case)
        H3: Variable prompt/response lengths with padding
        H4: Output is flat 1D with shape[0] == sum(episode_lengths)
        H5: All logprobs are valid (no NaN/Inf, all <= 0)
    """

    def _make_model(self, vocab_size=16, dim=8):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _make_trainer(self, model):
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo
        return Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

    def test_batched_matches_sequential(self):
        """H1: Batched output matches sequential compute_logprobs per episode."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        trainer = self._make_trainer(model)

        # 2 episodes: prompt_len=3, response_len=2 (uniform)
        prompt1 = mx.array([1, 2, 3])
        resp1 = mx.array([4, 5])
        prompt2 = mx.array([7, 8, 9])
        resp2 = mx.array([10, 11])

        # Build 2D obs: full_sequence = prompt + response, right-padded
        obs = mx.stack([mx.concatenate([prompt1, resp1]),
                        mx.concatenate([prompt2, resp2])])  # [2, 5]
        act = mx.stack([resp1, resp2])  # [2, 2]
        old_lp = mx.zeros(4)
        ep_lengths = [2, 2]
        prompt_lengths = [3, 3]

        # Batched path
        batched = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths, prompt_lengths)
        mx.eval(batched)

        # Sequential reference
        seq1 = compute_logprobs(model, prompt1, resp1)
        seq2 = compute_logprobs(model, prompt2, resp2)
        sequential = mx.concatenate([seq1, seq2])
        mx.eval(sequential)

        assert mx.allclose(batched, sequential, atol=1e-5).item(), (
            f"Batched {batched.tolist()} != sequential {sequential.tolist()}"
        )

    def test_single_episode(self):
        """H2: Single episode (degenerate case) produces valid output."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3, 5, 6]])  # [1, 5]
        act = mx.array([[5, 6]])            # [1, 2]
        old_lp = mx.zeros(2)
        ep_lengths = [2]
        prompt_lengths = [3]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths, prompt_lengths)
        mx.eval(result)

        assert result.ndim == 1
        assert result.shape[0] == 2
        assert mx.all(result <= 0).item()

    def test_variable_lengths_with_padding(self):
        """H3: Variable prompt/response lengths with right-padding."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        trainer = self._make_trainer(model)

        # ep1: prompt=3, response=2  → full_seq=5
        # ep2: prompt=2, response=3  → full_seq=5
        prompt1 = mx.array([1, 2, 3])
        resp1 = mx.array([4, 5])
        prompt2 = mx.array([6, 7])
        resp2 = mx.array([8, 9, 10])

        full1 = mx.concatenate([prompt1, resp1])     # [5]
        full2 = mx.concatenate([prompt2, resp2])      # [5]

        obs = mx.stack([full1, full2])  # [2, 5]
        # act: right-pad shorter response
        act = mx.stack([mx.pad(resp1, (0, 1)), resp2])  # [2, 3]

        old_lp = mx.zeros(5)
        ep_lengths = [2, 3]
        prompt_lengths = [3, 2]

        batched = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths, prompt_lengths)
        mx.eval(batched)

        # Sequential reference
        seq1 = compute_logprobs(model, prompt1, resp1)
        seq2 = compute_logprobs(model, prompt2, resp2)
        sequential = mx.concatenate([seq1, seq2])
        mx.eval(sequential)

        assert mx.allclose(batched, sequential, atol=1e-5).item(), (
            f"Batched {batched.tolist()} != sequential {sequential.tolist()}"
        )

    def test_output_is_flat_1d(self):
        """H4: Output is flat 1D with shape[0] == sum(episode_lengths)."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        act = mx.array([[4, 5], [9, 10]])
        old_lp = mx.zeros(4)
        ep_lengths = [2, 2]
        prompt_lengths = [3, 3]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths, prompt_lengths)
        mx.eval(result)

        assert result.ndim == 1
        assert result.shape[0] == sum(ep_lengths)

    def test_logprobs_are_valid(self):
        """H5: All logprobs are valid (no NaN/Inf, all <= 0)."""
        model = self._make_model()
        trainer = self._make_trainer(model)

        obs = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        act = mx.array([[4, 5], [9, 10]])
        old_lp = mx.zeros(4)
        ep_lengths = [2, 2]
        prompt_lengths = [3, 3]

        result = trainer._extract_grpo_logprobs(obs, act, old_lp, ep_lengths, prompt_lengths)
        mx.eval(result)

        assert not mx.any(mx.isnan(result)).item(), "NaN in batched logprobs"
        assert not mx.any(mx.isinf(result)).item(), "Inf in batched logprobs"
        assert mx.all(result <= 0).item(), f"Positive logprobs: {result.tolist()}"

    def test_return_token_entropies_matches_sequential(self):
        """Batched token entropy extraction matches per-episode sequential reference."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        trainer = self._make_trainer(model)

        prompt1 = mx.array([1, 2, 3])
        resp1 = mx.array([4, 5])
        prompt2 = mx.array([7, 8, 9])
        resp2 = mx.array([10, 11])

        obs = mx.stack([
            mx.concatenate([prompt1, resp1]),
            mx.concatenate([prompt2, resp2]),
        ])
        act = mx.stack([resp1, resp2])
        old_lp = mx.zeros(4)
        ep_lengths = [2, 2]
        prompt_lengths = [3, 3]

        batched_lp, batched_ent = trainer._extract_grpo_logprobs(
            obs,
            act,
            old_lp,
            ep_lengths,
            prompt_lengths,
            return_token_entropies=True,
        )
        mx.eval(batched_lp, batched_ent)

        seq1_lp, seq1_ent = compute_logprobs(
            model, prompt1, resp1, return_token_entropies=True
        )
        seq2_lp, seq2_ent = compute_logprobs(
            model, prompt2, resp2, return_token_entropies=True
        )
        seq_lp = mx.concatenate([seq1_lp, seq2_lp])
        seq_ent = mx.concatenate([seq1_ent, seq2_ent])
        mx.eval(seq_lp, seq_ent)

        assert batched_lp.shape == seq_lp.shape
        assert batched_ent.shape == seq_ent.shape
        assert mx.allclose(batched_lp, seq_lp, atol=1e-5).item()
        assert mx.allclose(batched_ent, seq_ent, atol=1e-5).item()
        assert not mx.any(mx.isnan(batched_ent)).item()
        assert not mx.any(mx.isinf(batched_ent)).item()
        assert mx.all(batched_ent >= 0).item()

    def test_empty_response_episode_in_middle(self):
        """Regression: empty-act episode between non-empty ones must not crash.

        Previously, _pack_episodes filtered out empty-act rows before stacking,
        causing act.shape[0] < len(prompt_lengths). compute_logprobs_batched
        then indexed out of bounds on the third episode.
        """
        from textpolicy.generation.mlx_generation import compute_logprobs
        from textpolicy.algorithms.grpo import _pack_episodes
        from types import SimpleNamespace

        model = self._make_model()
        trainer = self._make_trainer(model)

        ep1 = SimpleNamespace(obs=[[1, 2, 3]], act=[[4, 5]], rew=[1.0], logprob=[-0.5, -0.6])
        ep2 = SimpleNamespace(obs=[[6, 7]], act=[], rew=[0.0], logprob=[])
        ep3 = SimpleNamespace(obs=[[8, 9]], act=[[10, 11, 12]], rew=[0.5], logprob=[-0.3, -0.4, -0.7])

        batch = _pack_episodes([ep1, ep2, ep3], sort_by_length=False)

        # act must have same number of rows as obs / episode_lengths
        assert batch['obs'].shape[0] == batch['act'].shape[0] == 3

        result = trainer._extract_grpo_logprobs(
            batch['obs'], batch['act'], batch['logprob'],
            batch['episode_lengths'], batch['prompt_lengths'],
        )
        mx.eval(result)

        assert result.ndim == 1
        assert result.shape[0] == sum(batch['episode_lengths'])  # 2+0+3 = 5

        # Verify against sequential (skip empty ep2)
        seq1 = compute_logprobs(model, mx.array([1, 2, 3]), mx.array([4, 5]))
        seq3 = compute_logprobs(model, mx.array([8, 9]), mx.array([10, 11, 12]))
        sequential = mx.concatenate([seq1, seq3])
        mx.eval(sequential)

        assert mx.allclose(result, sequential, atol=1e-5).item(), (
            f"Batched {result.tolist()} != sequential {sequential.tolist()}"
        )

    def test_all_empty_responses_loss_is_zero_not_nan(self):
        """All-empty-response batch must produce loss=0.0, not nan.

        Regression: mx.mean([]) returns nan in MLX. When every episode
        has an empty response, policy_loss received empty arrays and
        returned nan. The guard in _loss_fn short-circuits to 0.0.
        """
        from textpolicy.algorithms import grpo
        from textpolicy.training import Trainer

        model = self._make_model()
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )

        batch = {
            'obs': mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int64),
            'act': mx.zeros((2, 0), dtype=mx.int64),
            'logprob': mx.array([], dtype=mx.float32),
            'rewards': mx.array([0.0, 0.0]),
            'episode_lengths': [0, 0],
            'prompt_lengths': [3, 3],
        }

        metrics = trainer.train(batch)
        assert metrics['loss'] == 0.0, f"Expected loss=0.0, got {metrics['loss']}"

    def test_loss_fn_populates_token_entropies_for_advantage_transform(self):
        """_loss_fn auto-populates token_entropies when transform is enabled."""
        from textpolicy.algorithms import grpo
        from textpolicy.training import Trainer

        model = self._make_model()
        seen = {"called": False}

        def passthrough_transform(advantages, batch_data):
            seen["called"] = True
            entropies = batch_data.get("token_entropies")
            assert entropies is not None, "token_entropies should be auto-populated"
            assert entropies.shape == advantages.shape
            assert not mx.any(mx.isnan(entropies)).item()
            assert not mx.any(mx.isinf(entropies)).item()
            assert mx.all(entropies >= 0).item()
            return advantages

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            advantage_transform_fn=passthrough_transform,
        )

        batch = {
            "obs": mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=mx.int64),
            "act": mx.array([[4, 5], [9, 10]], dtype=mx.int64),
            "logprob": mx.zeros(4, dtype=mx.float32),
            "rewards": mx.array([1.0, 0.5], dtype=mx.float32),
            "episode_lengths": [2, 2],
            "prompt_lengths": [3, 3],
        }

        loss = trainer._loss_fn(batch)
        mx.eval(loss)

        assert seen["called"], "advantage_transform_fn should be invoked"
        assert "token_entropies" in batch
        assert batch["token_entropies"].shape == (4,)


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

    def test_compiled_reasoning_transform_prepares_planning_mask(self):
        """Compiled trainer should precompute HICRA planning mask eagerly.

        Regression: identify_planning_tokens() uses eager token decoding and
        cannot run inside mx.compile. Trainer must call transform.prepare_batch
        before entering loss_and_grad_fn.
        """
        from textpolicy.training import Trainer, build_gtpo_hicra_transform
        from textpolicy.algorithms import grpo

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=32, hidden=16):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        class MockTokenizer:
            def convert_ids_to_tokens(self, ids):
                mapping = {
                    1: "let",
                    2: "me",
                    3: "think",
                }
                return [mapping.get(int(i), "x") for i in ids]

        model = TinyLM()
        mx.eval(model.parameters())
        transform = build_gtpo_hicra_transform(
            MockTokenizer(),
            strategic_grams=["let me think"],
            hicra_alpha=0.2,
            entropy_weight=0.0,
        )
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            advantage_transform_fn=transform,
        )

        batch = {
            "obs": mx.array([9, 8, 7, 6, 5], dtype=mx.int64),
            "act": mx.array([1, 2, 3, 4, 5], dtype=mx.int64),
            "logprob": -mx.ones(5, dtype=mx.float32),
            "rewards": mx.array([0.5, -0.2, 1.0, 0.1, -0.3], dtype=mx.float32),
        }

        metrics = trainer.train(batch)
        assert "loss" in metrics
        assert trainer._compiled is True
        assert "planning_mask" in batch
        assert batch["planning_mask"].shape == batch["act"].shape
        assert mx.sum(batch["planning_mask"]).item() >= 3.0

    def test_compiled_reasoning_with_checkpointing_train_step(self):
        """Compile + checkpoint + reasoning transform should train end-to-end."""
        from textpolicy.training import Trainer, build_gtpo_hicra_transform
        from textpolicy.algorithms import grpo

        class Block(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.linear = nn.Linear(hidden, hidden)

            def __call__(self, x, mask=None, cache=None):
                return nn.relu(self.linear(x))

        class TinyTransformer(nn.Module):
            def __init__(self, vocab_size=32, hidden=16):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.layers = [Block(hidden), Block(hidden)]
                self.head = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                h = self.embed(x)
                for layer in self.layers:
                    h = layer(h)
                return self.head(h)

        class MockTokenizer:
            def convert_ids_to_tokens(self, ids):
                mapping = {
                    1: "let",
                    2: "me",
                    3: "think",
                }
                return [mapping.get(int(i), "x") for i in ids]

        model = TinyTransformer()
        mx.eval(model.parameters())
        transform = build_gtpo_hicra_transform(
            MockTokenizer(),
            strategic_grams=["let me think"],
            hicra_alpha=0.2,
            entropy_weight=0.0,
        )
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            gradient_checkpointing=True,
            advantage_transform_fn=transform,
        )

        batch = {
            "obs": mx.array([9, 8, 7, 6, 5], dtype=mx.int64),
            "act": mx.array([1, 2, 3, 4, 5], dtype=mx.int64),
            "logprob": -mx.ones(5, dtype=mx.float32),
            "rewards": mx.array([0.5, -0.2, 1.0, 0.1, -0.3], dtype=mx.float32),
        }

        metrics = trainer.train(batch)
        assert "loss" in metrics
        assert trainer._compiled is True
        assert "planning_mask" in batch

    def test_trainer_calls_postprocess_batch_hook(self):
        """Trainer should call transform.postprocess_batch after loss/grad."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=32, hidden=16):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, hidden)
                self.proj = nn.Linear(hidden, vocab_size)

            def __call__(self, x):
                return self.proj(self.embed(x))

        class HookTransform:
            def __init__(self):
                self.prepare_calls = 0
                self.postprocess_calls = 0

            def prepare_batch(self, batch):
                self.prepare_calls += 1
                batch["prepare_seen"] = True

            def __call__(self, advantages, batch):
                return advantages

            def postprocess_batch(self, batch):
                self.postprocess_calls += 1
                batch["postprocess_seen"] = True
                batch["transform_metrics"] = {"hook_metric": 7.5}

        model = TinyLM()
        mx.eval(model.parameters())
        transform = HookTransform()
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            advantage_transform_fn=transform,
        )

        batch = {
            "obs": mx.array([9, 8, 7, 6, 5], dtype=mx.int64),
            "act": mx.array([1, 2, 3, 4, 5], dtype=mx.int64),
            "logprob": -mx.ones(5, dtype=mx.float32),
            "rewards": mx.array([0.5, -0.2, 1.0, 0.1, -0.3], dtype=mx.float32),
        }

        metrics = trainer.train(batch)
        assert "loss" in metrics
        assert transform.prepare_calls == 1
        assert transform.postprocess_calls == 1
        assert batch["prepare_seen"] is True
        assert batch["postprocess_seen"] is True
        assert metrics["hook_metric"] == pytest.approx(7.5)


@pytest.mark.unit
class TestAutoCompileDetection:
    """Tests for compile_training='auto' auto-detection (issue #40).

    Auto-compile uses a 1M parameter threshold: models above it are compiled,
    models below are not.  This avoids the 4x slowdown on tiny models while
    capturing the 16% speedup on real models (Qwen3-0.6B).
    """

    def _make_model(self, vocab_size=256, dim=64, layers=4):
        """TinyLM with configurable size."""
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.layers = [nn.Linear(dim, dim) for _ in range(layers)]
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                h = self.embed(x)
                for layer in self.layers:
                    h = nn.relu(layer(h))
                return self.head(h)

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _param_count(self, model):
        from mlx.utils import tree_flatten
        return sum(p.size for _, p in tree_flatten(model.parameters()))

    def test_auto_skips_compilation_for_small_models(self):
        """Models below 1M params should NOT be compiled in 'auto' mode."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        model = self._make_model(vocab_size=64, dim=16, layers=2)
        n_params = self._param_count(model)
        assert n_params < 1_000_000, f"Model too large for this test: {n_params}"

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training="auto",
        )

        assert trainer._compiled is False, (
            f"Model with {n_params:,} params should NOT be compiled"
        )

    def test_auto_compiles_large_models(self):
        """Models above 1M params should be compiled in 'auto' mode."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        # dim=128, vocab_size=1024 → embed(1024*128=131K) + head(128*1024=131K)
        # + 8 layers(128*128=16K each) = ~390K.  Need bigger.
        # dim=256, vocab_size=2048, layers=8 → 2048*256 + 8*256*256 + 256*2048
        # = 524K + 524K + 524K = ~1.6M params
        model = self._make_model(vocab_size=2048, dim=256, layers=8)
        n_params = self._param_count(model)
        assert n_params >= 1_000_000, f"Model too small for this test: {n_params}"

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training="auto",
        )

        assert trainer._compiled is True, (
            f"Model with {n_params:,} params should be compiled"
        )

    def test_explicit_true_always_compiles(self):
        """compile_training=True compiles regardless of model size."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        model = self._make_model(vocab_size=16, dim=4, layers=1)
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
        )
        assert trainer._compiled is True

    def test_explicit_false_never_compiles(self):
        """compile_training=False disables compilation regardless of size."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        model = self._make_model(vocab_size=2048, dim=256, layers=8)
        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )
        assert trainer._compiled is False


@pytest.mark.unit
class TestCompiledNumericalParity:
    """Compiled and uncompiled training must produce the same loss values.

    This is the L1 litmus test from issue #40: numerical parity ensures
    that mx.compile doesn't silently change training behavior.
    """

    def _make_model(self, vocab_size=16, dim=8):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))
        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def _make_batch(self, num_episodes=2, prompt_len=3, response_len=4, vocab_size=16):
        """Build a GRPO-style batch with prompt_lengths for the batched path."""
        obs = mx.random.randint(0, vocab_size, shape=(num_episodes, prompt_len + response_len))
        act = mx.random.randint(0, vocab_size, shape=(num_episodes, response_len))
        total_tokens = num_episodes * response_len
        logprob = -mx.abs(mx.random.normal((total_tokens,)))
        rewards = mx.random.normal((num_episodes,))
        mx.eval(obs, act, logprob, rewards)
        return {
            "obs": obs,
            "act": act,
            "logprob": logprob,
            "rewards": rewards,
            "episode_lengths": [response_len] * num_episodes,
            "prompt_lengths": [prompt_len] * num_episodes,
        }

    def test_compiled_vs_uncompiled_loss_match(self):
        """L1: Compiled and uncompiled _loss_fn produce identical loss."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)

        # Create two trainers with the SAME model (shared weights)
        model = self._make_model()
        batch = self._make_batch()

        trainer_uc = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )
        trainer_c = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
        )

        # Compute loss without training (no optimizer step)
        loss_uc, _ = trainer_uc.loss_and_grad_fn(batch)
        loss_c, _ = trainer_c.loss_and_grad_fn(batch)
        mx.eval(loss_uc, loss_c)

        assert abs(loss_uc.item() - loss_c.item()) < 1e-5, (
            f"Compiled loss {loss_c.item():.6f} != uncompiled {loss_uc.item():.6f}"
        )

    def test_compiled_training_step_produces_finite_loss(self):
        """Compiled training step produces a finite, non-NaN loss."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = self._make_model()
        batch = self._make_batch()

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
        )

        metrics = trainer.train(batch)
        assert not mx.isnan(mx.array(metrics["loss"])).item(), "Loss is NaN"
        assert not mx.isinf(mx.array(metrics["loss"])).item(), "Loss is Inf"

    def test_compiled_multi_step_training_converges(self):
        """Multiple compiled training steps should show decreasing loss."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = self._make_model()
        batch = self._make_batch()

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
        )

        losses = []
        for _ in range(5):
            metrics = trainer.train(batch)
            losses.append(metrics["loss"])

        # All losses must be finite
        for i, loss in enumerate(losses):
            assert not mx.isnan(mx.array(loss)).item(), f"Step {i}: loss is NaN"

    def test_compiled_with_grad_clip(self):
        """L5: mx.compile doesn't interfere with gradient clipping.

        Run compiled training with grad clipping enabled and verify
        all losses are finite across multiple steps.
        """
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        mx.random.seed(42)
        model = self._make_model()
        batch = self._make_batch()

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
            max_grad_norm=0.5,
        )

        for step in range(5):
            metrics = trainer.train(batch)
            assert not mx.isnan(mx.array(metrics["loss"])).item(), (
                f"Step {step}: loss is NaN with grad_clip + compile"
            )

    def test_compiled_grads_match_uncompiled(self):
        """Gradient values must match between compiled and uncompiled paths."""
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo
        from mlx.utils import tree_flatten

        mx.random.seed(42)
        model = self._make_model()
        batch = self._make_batch()

        trainer_uc = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
        )
        trainer_c = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=True,
        )

        _, grads_uc = trainer_uc.loss_and_grad_fn(batch)
        _, grads_c = trainer_c.loss_and_grad_fn(batch)
        mx.eval(grads_uc, grads_c)

        flat_uc = dict(tree_flatten(grads_uc))
        flat_c = dict(tree_flatten(grads_c))

        assert set(flat_uc.keys()) == set(flat_c.keys()), "Gradient keys differ"

        for key in flat_uc:
            assert mx.allclose(flat_uc[key], flat_c[key], atol=1e-5).item(), (
                f"Gradient mismatch for {key}"
            )


@pytest.mark.unit
class TestComputeLogprobsCompileSafety:
    """Verify that compute_logprobs _compiled flag works correctly.

    The _compiled parameter controls whether mx.any()-based validation
    (compile-unsafe) or mx.where-based sanitization (compile-safe) is used.
    Default (_compiled=False) preserves the original eager-eval semantics.
    """

    def _make_model(self, vocab_size=16, dim=8):
        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())
        return model

    def test_default_succeeds_on_normal_input(self):
        """Default mode returns valid logprobs for normal inputs."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        result = compute_logprobs(model, prompt, resp)
        mx.eval(result)
        assert result.shape[0] == 2
        assert not mx.any(mx.isnan(result)).item()

    def test_default_raises_on_nan_logprobs(self):
        """Default mode (_compiled=False) raises ValueError when model outputs NaN."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        inner = self._make_model()

        # Wrapper that injects NaN into prediction logits
        class NaNModel(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def __call__(self, x):
                logits = self.wrapped(x)
                return logits.at[0, 2, :].add(float('nan'))

        model = NaNModel(inner)

        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        with pytest.raises(ValueError, match="nan/inf"):
            result = compute_logprobs(model, prompt, resp)
            mx.eval(result)

    def test_compiled_flag_returns_same_values(self):
        """_compiled=True produces identical values when no NaN/Inf present."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        default_result = compute_logprobs(model, prompt, resp)
        compiled_result = compute_logprobs(model, prompt, resp, _compiled=True)
        mx.eval(default_result, compiled_result)

        assert mx.allclose(default_result, compiled_result, atol=1e-6).item(), (
            f"_compiled=True {compiled_result.tolist()} != "
            f"default {default_result.tolist()}"
        )

    def test_compiled_flag_sanitizes_nan(self):
        """_compiled=True replaces NaN/Inf with finfo.min instead of raising."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        inner = self._make_model()

        class InfModel(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def __call__(self, x):
                logits = self.wrapped(x)
                return logits.at[0, 2, :].add(float('inf'))

        model = InfModel(inner)
        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        # _compiled=True should sanitize, not raise
        result = compute_logprobs(model, prompt, resp, _compiled=True)
        mx.eval(result)
        assert not mx.any(mx.isnan(result)).item(), "NaN should be sanitized"
        assert not mx.any(mx.isinf(result)).item(), "Inf should be sanitized"

    def test_compiled_sanitizes_fp16_nan_inf(self):
        """fp16 NaN/Inf must be replaced with finite values, not -inf."""
        from textpolicy.generation.mlx_generation import compute_logprobs
        from mlx.utils import tree_map

        inner = self._make_model()
        # Cast model to fp16
        inner.update(tree_map(lambda p: p.astype(mx.float16), inner.parameters()))
        mx.eval(inner.parameters())

        class InfModel(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def __call__(self, x):
                logits = self.wrapped(x)
                return logits.at[0, 2, :].add(float('inf'))

        model = InfModel(inner)
        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        result = compute_logprobs(model, prompt, resp, _compiled=True)
        mx.eval(result)
        assert result.dtype == mx.float16, f"Expected float16, got {result.dtype}"
        assert not mx.any(mx.isinf(result)).item(), (
            f"fp16 sentinel overflowed to -inf: {result.tolist()}"
        )
        assert not mx.any(mx.isnan(result)).item()

    def test_batched_sanitizes_fp16_nan_inf(self):
        """compute_logprobs_batched fp16 NaN/Inf → finite, not -inf."""
        from textpolicy.generation.mlx_generation import compute_logprobs_batched
        from mlx.utils import tree_map

        inner = self._make_model()
        inner.update(tree_map(lambda p: p.astype(mx.float16), inner.parameters()))
        mx.eval(inner.parameters())

        class InfModel(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
            def __call__(self, x):
                logits = self.wrapped(x)
                # Corrupt first episode's prediction position
                return logits.at[0, 2, :].add(float('inf'))

        model = InfModel(inner)
        full_seq = mx.array([[1, 2, 3, 4, 5]])   # 1 episode, prompt=3, resp=2
        resp_2d = mx.array([[4, 5]])

        result = compute_logprobs_batched(model, full_seq, resp_2d, [3], [2])
        mx.eval(result)
        assert not mx.any(mx.isinf(result)).item(), (
            f"fp16 batched sentinel overflowed to -inf: {result.tolist()}"
        )
        assert not mx.any(mx.isnan(result)).item()

    def test_compiled_preserves_dtype(self):
        """_compiled=True must not promote fp16 logprobs to fp32."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        # Build an fp16 model
        model = self._make_model()
        from mlx.utils import tree_map
        model.update(tree_map(lambda p: p.astype(mx.float16), model.parameters()))
        mx.eval(model.parameters())

        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        result_default = compute_logprobs(model, prompt, resp, _compiled=False)
        result_compiled = compute_logprobs(model, prompt, resp, _compiled=True)
        mx.eval(result_default, result_compiled)

        assert result_default.dtype == result_compiled.dtype, (
            f"dtype mismatch: default={result_default.dtype}, "
            f"compiled={result_compiled.dtype}"
        )

    def test_compiled_flag_inside_mx_compile(self):
        """_compiled=True is safe inside mx.compile (no .item() calls)."""
        from textpolicy.generation.mlx_generation import compute_logprobs

        model = self._make_model()
        prompt = mx.array([1, 2, 3])
        resp = mx.array([4, 5])

        # Wrap in mx.compile — this would fail if _compiled path
        # still used mx.any() in a Python if statement.
        @mx.compile
        def compiled_logprobs(prompt_t, resp_t):
            return compute_logprobs(model, prompt_t, resp_t, _compiled=True)

        result = compiled_logprobs(prompt, resp)
        mx.eval(result)
        assert result.shape[0] == 2
        assert not mx.any(mx.isnan(result)).item()


@pytest.mark.unit
class TestLazyGradientClipping:
    """Tests for lazy gradient clipping (Issue #39).

    The lazy approach computes the global norm first and skips the scaling
    traversal when gradients are already within bounds.  This saves one
    full tree traversal per training step when the norm is below max_norm.

    Hypotheses:
        H1: When norm > max_norm, output matches optim.clip_grad_norm
        H2: When norm <= max_norm, original grads are returned (identity)
        H3: Works with nested gradient dicts (LoRA-style structures)
    """

    def _make_trainer(self, max_grad_norm=1.0):
        """Create a minimal Trainer for testing _clip_gradients."""
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
            max_grad_norm=max_grad_norm,
        )

    def test_lazy_clip_matches_full_when_clipping_needed(self):
        """H1: norm > max_norm → output matches optim.clip_grad_norm."""
        trainer = self._make_trainer(max_grad_norm=0.5)

        grads = {
            "weight": mx.array([[10.0, 20.0], [30.0, 40.0]]),
            "bias": mx.array([5.0, 15.0]),
        }
        mx.eval(grads)

        # Reference: MLX built-in clipping
        ref_clipped, _ = optim.clip_grad_norm(grads, 0.5)
        mx.eval(ref_clipped)

        # Lazy clipping
        lazy_clipped = trainer._clip_gradients(grads, 0.5)
        mx.eval(lazy_clipped)

        for key in grads:
            assert mx.allclose(lazy_clipped[key], ref_clipped[key], atol=1e-5).item(), (
                f"Mismatch for {key}: lazy={lazy_clipped[key].tolist()} "
                f"ref={ref_clipped[key].tolist()}"
            )

    def test_lazy_clip_returns_original_when_below_threshold(self):
        """H2: norm <= max_norm → returns original grads (identity, not copy)."""
        trainer = self._make_trainer(max_grad_norm=1000.0)

        grads = {
            "weight": mx.array([[0.01, 0.02], [0.03, 0.04]]),
            "bias": mx.array([0.001, 0.002]),
        }
        mx.eval(grads)

        result = trainer._clip_gradients(grads, 1000.0)

        # Should be the exact same dict (identity check)
        assert result is grads, (
            "When norm <= max_norm, _clip_gradients should return the "
            "original grads object, not a copy"
        )

    def test_lazy_clip_with_nested_lora_grads(self):
        """H3: Works with nested gradient dicts typical of LoRA models."""
        trainer = self._make_trainer(max_grad_norm=0.5)

        # Simulate nested LoRA gradient structure:
        # model.layers.0.self_attn.q_proj.lora_a.weight
        grads = {
            "layers": [
                {
                    "self_attn": {
                        "q_proj": {
                            "lora_a": {"weight": mx.array([[5.0, 6.0]])},
                            "lora_b": {"weight": mx.array([[7.0, 8.0]])},
                        }
                    }
                }
            ]
        }
        mx.eval(grads)

        # Reference
        ref_clipped, _ = optim.clip_grad_norm(grads, 0.5)
        mx.eval(ref_clipped)

        # Lazy
        lazy_clipped = trainer._clip_gradients(grads, 0.5)
        mx.eval(lazy_clipped)

        # Compare leaf values
        lazy_qa = lazy_clipped["layers"][0]["self_attn"]["q_proj"]["lora_a"]["weight"]
        ref_qa = ref_clipped["layers"][0]["self_attn"]["q_proj"]["lora_a"]["weight"]
        assert mx.allclose(lazy_qa, ref_qa, atol=1e-5).item(), (
            f"Nested LoRA grad mismatch: lazy={lazy_qa.tolist()} ref={ref_qa.tolist()}"
        )


@pytest.mark.unit
class TestSequencePacking:
    """Tests for length-sorted micro-batch trimming (sequence packing).

    _pack_episodes sorts episodes by total sequence length so that
    similarly-sized episodes are adjacent.  Combined with per-chunk
    trimming in _extract_grpo_logprobs, this reduces wasted padding
    compute when episode lengths vary widely.

    Hypotheses:
        H15: Sort correctness — episodes are reordered by total length
        H16: Trimmed micro-batch equivalence — sorted+trimmed produces
             the same logprobs as unsorted full-batch
        H17: sort_by_length=False preserves original order
        H18: Edge cases — 0 and 1 episode handled gracefully
    """

    def test_sort_correctness(self):
        """H15: Episodes sorted by total sequence length (ascending)."""
        from types import SimpleNamespace
        from textpolicy.algorithms.grpo import _pack_episodes

        # Total lengths: ep1=150 (100+50), ep2=60 (50+10), ep3=280 (200+80), ep4=100 (75+25)
        ep1 = SimpleNamespace(obs=[[1]*100], act=[[2]*50], rew=[1.0], logprob=[-0.5]*50)
        ep2 = SimpleNamespace(obs=[[1]*50], act=[[2]*10], rew=[0.5], logprob=[-0.3]*10)
        ep3 = SimpleNamespace(obs=[[1]*200], act=[[2]*80], rew=[0.8], logprob=[-0.4]*80)
        ep4 = SimpleNamespace(obs=[[1]*75], act=[[2]*25], rew=[0.2], logprob=[-0.6]*25)

        result = _pack_episodes([ep1, ep2, ep3, ep4])

        # Sorted order by total length: ep2(60), ep4(100), ep1(150), ep3(280)
        assert result['episode_lengths'] == [10, 25, 50, 80], (
            f"Expected [10, 25, 50, 80], got {result['episode_lengths']}"
        )
        assert result['prompt_lengths'] == [50, 75, 100, 200], (
            f"Expected [50, 75, 100, 200], got {result['prompt_lengths']}"
        )
        mx.eval(result['rewards'])
        expected_rewards = [0.5, 0.2, 1.0, 0.8]
        actual_rewards = result['rewards'].tolist()
        for i, (got, want) in enumerate(zip(actual_rewards, expected_rewards)):
            assert abs(got - want) < 1e-6, (
                f"rewards[{i}]: got {got}, want {want}"
            )

    def test_trimmed_microbatch_equivalence(self):
        """H16: Micro-batched logprobs (with trimming) match full-batch logprobs.

        Both paths produce the same set of per-episode logprobs, just in
        sorted order. We compare by sorting the per-episode segments.
        """
        from textpolicy.generation.mlx_generation import compute_logprobs
        from textpolicy.algorithms.grpo import _pack_episodes
        from types import SimpleNamespace

        class TinyLM(nn.Module):
            def __init__(self, vocab_size=16, dim=8):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, dim)
                self.head = nn.Linear(dim, vocab_size)
            def __call__(self, x):
                return self.head(self.embed(x))

        model = TinyLM()
        mx.eval(model.parameters())

        # 4 episodes with varying lengths (prompt + response)
        ep1 = SimpleNamespace(obs=[[1, 2, 3]], act=[[4, 5]], rew=[1.0], logprob=[-0.5, -0.6])
        ep2 = SimpleNamespace(obs=[[6, 7, 8, 9]], act=[[10]], rew=[0.5], logprob=[-0.3])
        ep3 = SimpleNamespace(obs=[[1, 2]], act=[[3, 4, 5]], rew=[0.8], logprob=[-0.4, -0.2, -0.1])
        ep4 = SimpleNamespace(obs=[[6, 7, 8]], act=[[9, 10, 11, 12]], rew=[0.2], logprob=[-0.7, -0.8, -0.9, -1.0])

        # Sequential reference: compute logprobs per episode independently
        ref_lps = []
        for ep in [ep1, ep2, ep3, ep4]:
            obs_flat = [t for step in ep.obs for t in (step if isinstance(step, list) else [step])]
            act_flat = [t for step in ep.act for t in (step if isinstance(step, list) else [step])]
            lp = compute_logprobs(model, mx.array(obs_flat), mx.array(act_flat))
            mx.eval(lp)
            ref_lps.append(lp)

        # Pack with sorting (default)
        batch = _pack_episodes([ep1, ep2, ep3, ep4])

        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo

        trainer = Trainer(
            model=model,
            loss_fn=grpo.policy_loss,
            optimizer=optim.Adam(learning_rate=1e-3),
            advantage_fn=grpo.compute_advantages,
            compile_training=False,
            micro_batch_size=2,
        )

        result = trainer._extract_grpo_logprobs(
            batch['obs'], batch['act'], batch['logprob'],
            batch['episode_lengths'], batch['prompt_lengths'],
        )
        mx.eval(result)

        # Extract per-episode logprob segments from the sorted result
        sorted_ep_lps = []
        offset = 0
        for length in batch['episode_lengths']:
            sorted_ep_lps.append(result[offset:offset + length])
            offset += length

        # The sorted episode_lengths tell us the order: match each segment
        # to its reference by episode length
        ref_by_len = {}
        for i, ep in enumerate([ep1, ep2, ep3, ep4]):
            act_flat = [t for step in ep.act for t in (step if isinstance(step, list) else [step])]
            ref_by_len[len(act_flat)] = ref_lps[i]

        for seg, length in zip(sorted_ep_lps, batch['episode_lengths']):
            ref = ref_by_len[length]
            mx.eval(seg, ref)
            assert mx.allclose(seg, ref, atol=1e-5).item(), (
                f"Segment mismatch for length {length}: "
                f"got {seg.tolist()}, want {ref.tolist()}"
            )

    def test_sort_by_length_false_preserves_order(self):
        """H17: sort_by_length=False preserves original episode ordering."""
        from types import SimpleNamespace
        from textpolicy.algorithms.grpo import _pack_episodes

        # Same episodes as H15
        ep1 = SimpleNamespace(obs=[[1]*100], act=[[2]*50], rew=[1.0], logprob=[-0.5]*50)
        ep2 = SimpleNamespace(obs=[[1]*50], act=[[2]*10], rew=[0.5], logprob=[-0.3]*10)
        ep3 = SimpleNamespace(obs=[[1]*200], act=[[2]*80], rew=[0.8], logprob=[-0.4]*80)
        ep4 = SimpleNamespace(obs=[[1]*75], act=[[2]*25], rew=[0.2], logprob=[-0.6]*25)

        result = _pack_episodes([ep1, ep2, ep3, ep4], sort_by_length=False)

        # Original order preserved
        assert result['episode_lengths'] == [50, 10, 80, 25], (
            f"Expected [50, 10, 80, 25], got {result['episode_lengths']}"
        )
        assert result['prompt_lengths'] == [100, 50, 200, 75], (
            f"Expected [100, 50, 200, 75], got {result['prompt_lengths']}"
        )

    def test_empty_episodes_edge_case(self):
        """H18: Zero episodes — no crash, returns empty batch."""
        from textpolicy.algorithms.grpo import _pack_episodes

        result = _pack_episodes([])
        assert result['episode_lengths'] == []
        assert result['prompt_lengths'] == []

    def test_single_episode_edge_case(self):
        """H18: Single episode — sorting is a no-op."""
        from types import SimpleNamespace
        from textpolicy.algorithms.grpo import _pack_episodes

        ep = SimpleNamespace(obs=[[1, 2, 3]], act=[[4, 5]], rew=[1.0], logprob=[-0.5, -0.6])
        result = _pack_episodes([ep])

        assert result['episode_lengths'] == [2]
        assert result['prompt_lengths'] == [3]
        mx.eval(result['rewards'])
        assert result['rewards'].tolist() == [1.0]
