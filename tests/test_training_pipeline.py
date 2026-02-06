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

        batch = _pack_episodes([ep1, ep2, ep3])

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
