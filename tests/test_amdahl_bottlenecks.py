"""
Tests for Amdahl's Law bottleneck fixes.

Validates that batched MLX evaluation produces identical results to the
original per-element .item() approach, and measures the performance
improvement from reduced GPU-CPU synchronization barriers.

Bottlenecks addressed:
1. _precompute_episode_rewards: N .item() calls → 1 mx.eval()
2. _pack_episodes: N .item() calls → 1 mx.eval()
3. _prepare_batch_from_buffer: N .item() calls → 1 mx.eval()
4. Trainer.train() duplicate forward pass → metrics_interval
"""

import time
import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Dict, Any
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers: reference (original) implementations for correctness comparison
# ---------------------------------------------------------------------------

def _precompute_episode_rewards_reference(episodes):
    """Original per-episode .item() implementation for correctness baseline."""
    rewards = []
    for ep in episodes:
        if hasattr(ep, 'rew'):
            reward = float(mx.sum(mx.array(ep.rew)).item())
        else:
            rew = ep.get('rew', ep.get('reward', [0.0]))
            if isinstance(rew, (int, float)):
                reward = float(rew)
            else:
                reward = float(mx.sum(mx.array(rew)).item())
        rewards.append(reward)
    return rewards


def _pack_episodes_episode_rewards_reference(episodes):
    """Original per-episode .item() for episode_rewards in _pack_episodes."""
    episode_rewards = []
    for episode in episodes:
        if hasattr(episode, 'rew'):
            episode_reward = mx.sum(mx.array(episode.rew)).item()
        else:
            episode_reward = mx.sum(mx.array(episode['rew'])).item()
        episode_rewards.append(episode_reward)
    return episode_rewards


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dict_episodes(n: int, reward_len: int = 5) -> List[Dict]:
    """Create n synthetic dict-format episodes with known rewards."""
    episodes = []
    for i in range(n):
        rew = [float(j + i * 0.1) for j in range(reward_len)]
        episodes.append({
            'obs': [[1, 2, 3]],
            'act': [[4, 5]],
            'rew': rew,
            'next_obs': [[6, 7, 8]],
            'done': [True],
            'timeout': [False],
            'logprob': [[-0.5, -0.3]],
        })
    return episodes


class _FakeEpisode:
    """Minimal Episode-like object for testing."""

    def __init__(self, rew, obs=None, act=None, logprob=None):
        self.rew = rew
        self.obs = obs or [[1, 2, 3]]
        self.act = act or [[4, 5]]
        self.logprob = logprob or [[-0.5, -0.3]]


def _make_object_episodes(n: int, reward_len: int = 5) -> list:
    """Create n synthetic Episode-like objects with known rewards."""
    episodes = []
    for i in range(n):
        rew = [float(j + i * 0.1) for j in range(reward_len)]
        episodes.append(_FakeEpisode(rew=rew))
    return episodes


# ---------------------------------------------------------------------------
# 1. Correctness: _precompute_episode_rewards
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrecomputeEpisodeRewards:
    """Verify batched version produces identical results to reference."""

    def test_dict_episodes_match_reference(self):
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = _make_dict_episodes(20)
        expected = _precompute_episode_rewards_reference(episodes)
        actual = _precompute_episode_rewards(episodes)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-5, f"Mismatch: {a} vs {e}"

    def test_object_episodes_match_reference(self):
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = _make_object_episodes(20)
        expected = _precompute_episode_rewards_reference(episodes)
        actual = _precompute_episode_rewards(episodes)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-5, f"Mismatch: {a} vs {e}"

    def test_scalar_rewards_no_mlx_needed(self):
        """Scalar float rewards should bypass MLX entirely."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [{'rew': 1.5}, {'rew': 2.0}, {'rew': -0.3}]
        result = _precompute_episode_rewards(episodes)
        assert result == [1.5, 2.0, -0.3]

    def test_mixed_scalar_and_array_rewards(self):
        """Mix of scalar and array rewards should all resolve correctly."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [
            {'rew': 3.0},                          # scalar
            {'rew': [1.0, 2.0]},                    # array → sum = 3.0
            _FakeEpisode(rew=[0.5, 0.5, 0.5]),     # object → sum = 1.5
            {'rew': 0},                             # scalar zero
        ]
        expected = _precompute_episode_rewards_reference(episodes)
        actual = _precompute_episode_rewards(episodes)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-5

    def test_empty_episodes(self):
        from textpolicy.algorithms.grpo import _precompute_episode_rewards
        assert _precompute_episode_rewards([]) == []

    def test_single_episode(self):
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [{'rew': [10.0]}]
        assert _precompute_episode_rewards(episodes) == [10.0]


# ---------------------------------------------------------------------------
# 2. Correctness: _pack_episodes reward extraction
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPackEpisodesRewards:
    """Verify _pack_episodes produces correct episode_rewards with batched eval."""

    def test_dict_episodes_rewards(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = _make_dict_episodes(10)
        expected_rewards = _pack_episodes_episode_rewards_reference(episodes)
        result = _pack_episodes(episodes)

        actual_rewards = result['rewards'].tolist()
        assert len(actual_rewards) == len(expected_rewards)
        for a, e in zip(actual_rewards, expected_rewards):
            assert abs(a - e) < 1e-5, f"Mismatch: {a} vs {e}"

    def test_object_episodes_rewards(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = _make_object_episodes(10)
        expected_rewards = _pack_episodes_episode_rewards_reference(episodes)
        result = _pack_episodes(episodes)

        actual_rewards = result['rewards'].tolist()
        assert len(actual_rewards) == len(expected_rewards)
        for a, e in zip(actual_rewards, expected_rewards):
            assert abs(a - e) < 1e-5

    def test_episode_lengths_preserved(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = _make_dict_episodes(5)
        result = _pack_episodes(episodes)

        # Each episode has act=[[4,5]], flattened length = 2
        assert result['episode_lengths'] == [2] * 5

    def test_logprobs_flattened_for_single_step_token_arrays(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = [{
            'obs': [[101, 102, 103]],
            'act': [[201, 202, 203, 204]],
            'rew': [1.0],
            'next_obs': [[0]],
            'done': [True],
            'timeout': [False],
            'logprob': [mx.array([-0.5, -0.4, -0.3, -0.2])],
        }]
        result = _pack_episodes(episodes)

        assert result['episode_lengths'] == [4]
        assert result['logprob'].shape == (4,)
        assert np.allclose(result['logprob'].tolist(), [-0.5, -0.4, -0.3, -0.2], atol=1e-6)

    def test_logprobs_flattened_when_dict_logprob_is_direct_array(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = [{
            'obs': [[11, 12]],
            'act': [[21, 22, 23]],
            'rew': [1.0],
            'next_obs': [[0]],
            'done': [True],
            'timeout': [False],
            'logprob': mx.array([-0.7, -0.6, -0.5]),
        }]
        result = _pack_episodes(episodes)

        assert result['episode_lengths'] == [3]
        assert result['logprob'].shape == (3,)
        assert np.allclose(result['logprob'].tolist(), [-0.7, -0.6, -0.5], atol=1e-6)

    def test_empty_episodes(self):
        from textpolicy.algorithms.grpo import _pack_episodes

        result = _pack_episodes([])
        assert result['rewards'].size == 0
        assert result['episode_lengths'] == []


# ---------------------------------------------------------------------------
# 3. Correctness: _prepare_batch_from_buffer
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPrepareBatchFromBuffer:
    """Verify batched reward extraction in _prepare_batch_from_buffer."""

    def test_rewards_match_sequential(self):
        """Batched mx.eval should produce same rewards as sequential .item()."""
        from textpolicy.buffer import Buffer

        # Build buffer with known episodes using Buffer.add() API
        buf = Buffer(max_episodes=10)
        expected_rewards = []
        for i in range(5):
            rew_vals = [float(j + i) for j in range(3)]
            for j in range(3):
                buf.add(
                    obs=mx.array([1, 2]),
                    act=mx.array([3]),
                    rew=rew_vals[j],
                    next_obs=mx.array([4, 5]),
                    done=(j == 2),
                    logprob=mx.array([-0.5]),
                )
            expected_rewards.append(sum(rew_vals))

        # Use a minimal Trainer to call _prepare_batch_from_buffer
        from textpolicy.training.trainer import Trainer
        import mlx.nn as nn
        import mlx.optimizers as optim

        model = nn.Linear(2, 2)
        optimizer = optim.Adam(learning_rate=0.01)
        trainer = Trainer(
            model=model,
            advantage_fn=lambda r: r,
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optimizer,
            compile_training=False,
        )

        batch = trainer._prepare_batch_from_buffer(buf)
        actual_rewards = batch['rewards'].tolist()

        assert len(actual_rewards) == len(expected_rewards)
        for a, e in zip(actual_rewards, expected_rewards):
            assert abs(a - e) < 1e-5, f"Mismatch: {a} vs {e}"


# ---------------------------------------------------------------------------
# 4. Correctness: metrics_interval
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMetricsInterval:
    """Verify that metrics_interval controls when detailed metrics are computed."""

    def _make_trainer(self, metrics_interval=1):
        import mlx.nn as nn
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        model = nn.Linear(4, 4)
        optimizer = optim.Adam(learning_rate=0.01)

        call_count = {'n': 0}

        def counting_metrics_fn(old_lp, new_lp, adv):
            call_count['n'] += 1
            return {'dummy_metric': 0.0}

        trainer = Trainer(
            model=model,
            advantage_fn=lambda r: r - mx.mean(r),
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optimizer,
            metrics_fn=counting_metrics_fn,
            compile_training=False,
            metrics_interval=metrics_interval,
            # Linear(4,4) can't produce valid sequence logits; bypass
            # _default_get_logprobs since this test is about metrics_interval
            # scheduling, not logprob correctness.
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
        )
        return trainer, call_count

    def _make_batch(self):
        """Create a batch compatible with the Linear(4,4) test model."""
        return {
            'obs': mx.random.normal((4,)),
            'act': mx.array([1, 2, 3]),
            'logprob': mx.array([-1.0, -1.0, -1.0]),
            'rewards': mx.array([1.0, 0.5]),
            'episode_lengths': [2, 1],
        }

    def test_interval_1_computes_every_step(self):
        trainer, call_count = self._make_trainer(metrics_interval=1)

        for _ in range(5):
            trainer.train(self._make_batch())

        assert call_count['n'] == 5, f"Expected 5 metrics calls, got {call_count['n']}"

    def test_interval_3_skips_steps(self):
        trainer, call_count = self._make_trainer(metrics_interval=3)

        for _ in range(9):
            trainer.train(self._make_batch())

        # Steps 0,3,6 → 3 calls out of 9
        assert call_count['n'] == 3, f"Expected 3 metrics calls, got {call_count['n']}"

    def test_interval_always_returns_loss(self):
        """Even when metrics are skipped, loss should always be present."""
        trainer, _ = self._make_trainer(metrics_interval=100)

        # Step 0: 0 % 100 == 0, so metrics WILL be computed on first step
        # Step 1: 1 % 100 != 0, so metrics will be skipped
        trainer.train(self._make_batch())  # step 0 - metrics computed
        metrics = trainer.train(self._make_batch())  # step 1 - metrics skipped

        assert 'loss' in metrics
        assert 'step' in metrics
        # The dummy_metric should NOT be in step 1's metrics
        assert 'dummy_metric' not in metrics

    def test_default_interval_is_10(self):
        """Default metrics_interval should be 10 to avoid redundant forward passes."""
        import mlx.nn as nn
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        trainer = Trainer(
            model=nn.Linear(2, 2),
            advantage_fn=lambda r: r,
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optim.Adam(learning_rate=0.01),
            compile_training=False,
        )
        assert trainer.metrics_interval == 10


# ---------------------------------------------------------------------------
# 5. Performance benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchedEvalPerformance:
    """Measure sync barrier reduction from batched evaluation.

    These tests verify that the batched implementation uses fewer
    mx.eval() calls than the reference. We proxy this by counting
    actual wall-clock time — fewer sync barriers = faster execution.
    """

    @staticmethod
    def _time_fn(fn, *args, warmup=2, iterations=10):
        """Time a function with warmup. Returns median time in seconds."""
        for _ in range(warmup):
            fn(*args)
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        times.sort()
        return times[len(times) // 2]  # median

    def test_precompute_rewards_no_slower(self):
        """Batched implementation should not be slower than reference."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = _make_dict_episodes(100, reward_len=20)

        t_ref = self._time_fn(_precompute_episode_rewards_reference, episodes)
        t_new = self._time_fn(_precompute_episode_rewards, episodes)

        # Allow 20% tolerance — the batched version should be equal or faster
        assert t_new < t_ref * 1.2, (
            f"Batched ({t_new:.4f}s) is more than 20% slower than "
            f"reference ({t_ref:.4f}s)"
        )

    def test_pack_episodes_no_slower(self):
        """Batched _pack_episodes should not be slower than sequential baseline."""
        from textpolicy.algorithms.grpo import _pack_episodes

        episodes = _make_dict_episodes(100, reward_len=20)

        # Baseline: time _pack_episodes (which is now batched)
        t_new = self._time_fn(_pack_episodes, episodes)

        # Reference: time the sequential reward extraction
        t_ref = self._time_fn(_pack_episodes_episode_rewards_reference, episodes)

        # The full _pack_episodes does more work (flattening, padding) than
        # just reward extraction, so we only check it's in the same ballpark
        # The point is the reward portion shouldn't be a bottleneck
        assert t_new < t_ref * 5.0, (
            f"_pack_episodes ({t_new:.4f}s) is unexpectedly slow vs "
            f"reward-only reference ({t_ref:.4f}s)"
        )

    def test_batched_eval_fewer_syncs(self):
        """Verify batched approach uses 1 mx.eval vs N .item() calls.

        We measure this indirectly: building N lazy sums and evaluating
        them once should be faster than evaluating them one at a time.
        """
        n = 200
        arrays = [mx.array([float(i), float(i + 1), float(i + 2)]) for i in range(n)]

        def sequential():
            return [mx.sum(a).item() for a in arrays]

        def batched():
            lazy = [mx.sum(a) for a in arrays]
            stacked = mx.stack(lazy)
            mx.eval(stacked)
            return stacked.tolist()

        t_seq = self._time_fn(sequential, warmup=3, iterations=20)
        t_bat = self._time_fn(batched, warmup=3, iterations=20)

        # Batched should be at least not worse
        # (On Apple Silicon with unified memory, the improvement may be modest
        # for small N but should never be slower)
        assert t_bat < t_seq * 1.5, (
            f"Batched ({t_bat:.6f}s) unexpectedly slower than "
            f"sequential ({t_seq:.6f}s) — suggests batching adds overhead "
            f"without reducing sync barriers"
        )


# ---------------------------------------------------------------------------
# 6. Regression: full pipeline still works
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPipelineRegression:
    """Ensure optimized functions integrate correctly with the rest of the system."""

    def test_filter_informative_prompts_with_batched_rewards(self):
        """filter_informative_prompts uses _precompute_episode_rewards internally."""
        from textpolicy.algorithms.grpo import filter_informative_prompts

        # Create episodes with varied rewards (some informative, some not)
        episodes = []
        # Group 1: same prompt, varied rewards (informative)
        for r in [0.0, 0.5, 1.0]:
            episodes.append({
                'obs': [[10, 20]],
                'act': [[30]],
                'rew': [r],
                'next_obs': [[40]],
                'done': [True],
                'timeout': [False],
            })
        # Group 2: same prompt, identical rewards (uninformative)
        for _ in range(3):
            episodes.append({
                'obs': [[50, 60]],
                'act': [[70]],
                'rew': [1.0],
                'next_obs': [[80]],
                'done': [True],
                'timeout': [False],
            })

        filtered, stats = filter_informative_prompts(episodes, min_variance=0.01)

        # Group 1 should be kept (variance > 0), group 2 dropped
        assert stats['prompts_kept'] >= 1
        assert stats['episodes_kept'] >= 3

    def test_select_all_data_with_batched_pack(self):
        """select_all_data uses _pack_episodes internally."""
        from textpolicy.algorithms.grpo import select_all_data
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=10)
        for i in range(3):
            buf.add(
                obs=[1, 2, 3],
                act=[4, 5],
                rew=float(i),
                next_obs=[6, 7, 8],
                done=True,
                logprob=[-0.5, -0.3],
            )

        batch = select_all_data(buf)

        assert 'rewards' in batch
        assert 'episode_lengths' in batch
        assert batch['rewards'].shape[0] == 3

    def test_compute_prompt_group_stats_with_batched_rewards(self):
        """compute_prompt_group_stats uses _precompute_episode_rewards internally."""
        from textpolicy.algorithms.grpo import compute_prompt_group_stats

        episodes = _make_dict_episodes(10)
        stats = compute_prompt_group_stats(episodes)

        assert 'num_prompts' in stats
        assert 'num_episodes' in stats
        assert stats['num_episodes'] == 10

    def test_trainer_train_returns_metrics(self):
        """Trainer.train() should work with optimized _prepare_batch_from_buffer."""
        import mlx.nn as nn
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        model = nn.Linear(4, 4)
        optimizer = optim.Adam(learning_rate=0.01)

        trainer = Trainer(
            model=model,
            advantage_fn=lambda r: r - mx.mean(r),
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optimizer,
            compile_training=False,
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
        )

        batch = {
            'obs': mx.random.normal((4,)),
            'act': mx.array([1, 2, 3]),
            'logprob': mx.array([-1.0, -1.0, -1.0]),
            'rewards': mx.array([1.0, 0.5, -0.2]),
        }
        metrics = trainer.train(batch)

        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        assert 'step' in metrics


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEdgeCases:
    """Edge cases that could break batched evaluation."""

    def test_single_episode_batched(self):
        """Single episode should not break mx.stack with 1 element."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [{'rew': [1.0, 2.0, 3.0]}]
        result = _precompute_episode_rewards(episodes)
        assert len(result) == 1
        assert abs(result[0] - 6.0) < 1e-5

    def test_large_batch_batched(self):
        """Large batch should work without OOM or stack issues."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = _make_dict_episodes(500, reward_len=10)
        expected = _precompute_episode_rewards_reference(episodes)
        actual = _precompute_episode_rewards(episodes)

        assert len(actual) == 500
        for a, e in zip(actual, expected):
            assert abs(a - e) < 1e-4

    def test_zero_rewards(self):
        """All-zero rewards should produce all-zero results."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [{'rew': [0.0, 0.0]} for _ in range(5)]
        result = _precompute_episode_rewards(episodes)
        assert all(r == 0.0 for r in result)

    def test_negative_rewards(self):
        """Negative rewards should be preserved correctly."""
        from textpolicy.algorithms.grpo import _precompute_episode_rewards

        episodes = [{'rew': [-1.0, -2.0]}, {'rew': [-0.5]}]
        result = _precompute_episode_rewards(episodes)
        assert abs(result[0] - (-3.0)) < 1e-5
        assert abs(result[1] - (-0.5)) < 1e-5

    def test_metrics_interval_zero_clamps_to_1(self):
        """metrics_interval=0 should be clamped to 1."""
        import mlx.nn as nn
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        trainer = Trainer(
            model=nn.Linear(2, 2),
            advantage_fn=lambda r: r,
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optim.Adam(learning_rate=0.01),
            compile_training=False,
            metrics_interval=0,
        )
        assert trainer.metrics_interval == 1

    def test_metrics_interval_negative_clamps_to_1(self):
        """Negative metrics_interval should be clamped to 1."""
        import mlx.nn as nn
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        trainer = Trainer(
            model=nn.Linear(2, 2),
            advantage_fn=lambda r: r,
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optim.Adam(learning_rate=0.01),
            compile_training=False,
            metrics_interval=-5,
        )
        assert trainer.metrics_interval == 1


# ---------------------------------------------------------------------------
# 8. Inline logprob capture
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Minimal causal LM for testing: embedding + linear head → (batch, seq, vocab)."""

    def __init__(self, vocab_size: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x):
        return self.head(self.embed(x))


@pytest.mark.unit
class TestInlineLogprobCapture:
    """Verify that inline logprob capture matches compute_logprobs output."""

    def test_simple_generate_inline_matches_teacher_forcing(self):
        """Inline logprobs from _simple_generate should match compute_logprobs."""
        from textpolicy.generation.mlx_generation import _simple_generate, compute_logprobs

        model = _TinyLM(vocab_size=16, dim=8)
        mx.eval(model.parameters())

        prompt = mx.array([1, 2, 3, 4])

        # Generate with inline logprobs (temperature=0 for deterministic)
        response_tokens, info = _simple_generate(model, prompt, max_tokens=5, temperature=0.0)
        inline_logprobs = info['logprob']

        # Recompute via teacher-forcing forward pass
        tf_logprobs = compute_logprobs(model, prompt, response_tokens)

        # They should match within numerical precision
        mx.eval(inline_logprobs, tf_logprobs)
        for i in range(len(response_tokens)):
            diff = abs(float(inline_logprobs[i]) - float(tf_logprobs[i]))
            assert diff < 1e-4, (
                f"Token {i}: inline={float(inline_logprobs[i]):.6f} vs "
                f"teacher-forcing={float(tf_logprobs[i]):.6f} (diff={diff:.6f})"
            )

    def test_generate_tokens_fallback_when_logprobs_none(self):
        """generate_tokens should fall back to compute_logprobs when
        segment.logprobs is None."""
        from unittest.mock import patch, MagicMock
        from textpolicy.generation.mlx_generation import generate_tokens, compute_logprobs

        model = _TinyLM(vocab_size=16, dim=8)
        mx.eval(model.parameters())
        prompt = mx.array([1, 2, 3, 4])

        # Create fake segments with logprobs=None
        fake_segment = MagicMock()
        fake_segment.token = 5
        fake_segment.logprobs = None
        fake_segment.finish_reason = "stop"

        # Patch stream_generate at the import site inside generate_tokens
        with patch.dict('sys.modules', {
            'mlx_lm': MagicMock(
                stream_generate=MagicMock(return_value=[fake_segment]),
            ),
        }):
            with patch(
                'textpolicy.generation.mlx_generation.compute_logprobs',
                wraps=compute_logprobs,
            ) as mock_compute:
                response_tokens, info = generate_tokens(
                    model, MagicMock(), prompt, max_tokens=3,
                )
                # Since logprobs was None, it should have called compute_logprobs
                assert mock_compute.called, "compute_logprobs should be called as fallback"

    def test_inline_logprobs_are_negative(self):
        """All inline logprobs should be <= 0 (valid log-probabilities)."""
        from textpolicy.generation.mlx_generation import _simple_generate

        model = _TinyLM(vocab_size=16, dim=8)
        mx.eval(model.parameters())
        prompt = mx.array([1, 2, 3, 4])

        _, info = _simple_generate(model, prompt, max_tokens=5, temperature=0.5)
        logprobs = info['logprob']
        mx.eval(logprobs)

        for i in range(len(logprobs)):
            assert float(logprobs[i]) <= 0.0, (
                f"Token {i}: logprob={float(logprobs[i]):.6f} is positive"
            )


# ---------------------------------------------------------------------------
# 9. Batched logprob recomputation (Issue #27)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchedLogprobRecomputation:
    """Verify batched compute_logprobs_batched matches sequential compute_logprobs.

    This tests the core optimization from Issue #27: converting N sequential
    model forward passes into a single batched forward pass.
    """

    def _make_model(self, vocab_size=16, dim=8):
        model = _TinyLM(vocab_size=vocab_size, dim=dim)
        mx.eval(model.parameters())
        return model

    def test_batched_matches_sequential(self):
        """Batched output matches N sequential compute_logprobs calls."""
        from textpolicy.generation.mlx_generation import compute_logprobs, compute_logprobs_batched

        model = self._make_model()
        n_episodes = 4
        prompt_len = 3
        resp_len = 2

        # Build per-episode data
        prompts = [mx.array([1 + i * 3, 2 + i * 3, 3 + i * 3]) for i in range(n_episodes)]
        responses = [mx.array([10 + i * 2, 11 + i * 2]) for i in range(n_episodes)]

        # Sequential reference
        sequential_parts = []
        for i in range(n_episodes):
            lp = compute_logprobs(model, prompts[i], responses[i])
            sequential_parts.append(lp)
        sequential = mx.concatenate(sequential_parts)
        mx.eval(sequential)

        # Batched: build 2D arrays
        full_seqs = mx.stack([mx.concatenate([p, r]) for p, r in zip(prompts, responses)])
        resp_2d = mx.stack(responses)
        prompt_lengths = [prompt_len] * n_episodes
        response_lengths = [resp_len] * n_episodes

        batched = compute_logprobs_batched(model, full_seqs, resp_2d, prompt_lengths, response_lengths)
        mx.eval(batched)

        assert mx.allclose(sequential, batched, atol=1e-5).item(), (
            f"Batched {batched.tolist()} != sequential {sequential.tolist()}"
        )

    def test_variable_lengths(self):
        """Variable prompt/response lengths produce correct flat 1D output."""
        from textpolicy.generation.mlx_generation import compute_logprobs, compute_logprobs_batched

        model = self._make_model()

        prompt1 = mx.array([1, 2, 3])     # len=3
        resp1 = mx.array([4, 5])          # len=2
        prompt2 = mx.array([6, 7])        # len=2
        resp2 = mx.array([8, 9, 10])      # len=3

        # Sequential
        seq1 = compute_logprobs(model, prompt1, resp1)
        seq2 = compute_logprobs(model, prompt2, resp2)
        sequential = mx.concatenate([seq1, seq2])
        mx.eval(sequential)

        # Batched: pad to max lengths
        full1 = mx.concatenate([prompt1, resp1])          # [5]
        full2 = mx.concatenate([prompt2, resp2])           # [5]
        full_seqs = mx.stack([full1, full2])               # [2, 5]

        resp_2d = mx.stack([mx.pad(resp1, (0, 1)), resp2])  # [2, 3]

        batched = compute_logprobs_batched(
            model, full_seqs, resp_2d, [3, 2], [2, 3]
        )
        mx.eval(batched)

        assert batched.shape[0] == 5  # sum(response_lengths) = 2+3
        assert mx.allclose(sequential, batched, atol=1e-5).item()

    def test_fewer_model_calls(self):
        """Batched path uses 1 model call vs N sequential calls.

        We verify by wrapping the model in a counting proxy that intercepts
        the forward pass.
        """
        from textpolicy.generation.mlx_generation import compute_logprobs, compute_logprobs_batched

        inner_model = self._make_model()
        call_count = {'n': 0}

        class CountingWrapper(nn.Module):
            """Thin wrapper that counts forward-pass invocations."""
            def __init__(self, wrapped):
                super().__init__()
                self._wrapped = wrapped
            def __call__(self, x):
                call_count['n'] += 1
                return self._wrapped(x)

        model = CountingWrapper(inner_model)

        n_episodes = 8
        prompts = [mx.array([1, 2, 3]) for _ in range(n_episodes)]
        responses = [mx.array([4, 5]) for _ in range(n_episodes)]

        # Sequential: N calls
        call_count['n'] = 0
        for i in range(n_episodes):
            compute_logprobs(model, prompts[i], responses[i])
        n_sequential = call_count['n']

        # Batched: 1 call
        call_count['n'] = 0
        full_seqs = mx.stack([mx.concatenate([p, r]) for p, r in zip(prompts, responses)])
        resp_2d = mx.stack(responses)
        compute_logprobs_batched(model, full_seqs, resp_2d, [3] * n_episodes, [2] * n_episodes)
        n_batched = call_count['n']

        assert n_sequential == n_episodes, f"Sequential should be {n_episodes} calls, got {n_sequential}"
        assert n_batched == 1, f"Batched should be 1 call, got {n_batched}"


# ---------------------------------------------------------------------------
# 10. On-policy buffer management (Issue #35)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOnPolicyBufferManagement:
    """Verify that on_policy prevents unbounded buffer growth.

    Issue #35: Without clearing the buffer, training time grows exponentially
    because the Trainer does a forward pass over ALL buffered episodes.  For
    on-policy algorithms (GRPO, GTPO), the buffer should be cleared after each
    train() call so only the latest rollouts are used.

    The default is on_policy=True — matching the dominant use case.
    """

    def _make_trainer(self, buffer, on_policy=None):
        """Create a minimal Trainer with a linked buffer.

        Args:
            on_policy: Pass True/False to set explicitly, or None to use
                the Trainer default (True).
        """
        import mlx.optimizers as optim
        from textpolicy.training.trainer import Trainer

        model = nn.Linear(4, 4)
        optimizer = optim.Adam(learning_rate=0.01)
        kwargs = dict(
            model=model,
            advantage_fn=lambda r: r - mx.mean(r),
            loss_fn=lambda o, n, a: mx.mean(n),
            optimizer=optimizer,
            compile_training=False,
            buffer=buffer,
            data_selector_fn=lambda buf: self._make_batch(),
            # Bypass default logprob extraction — Linear(4,4) can't
            # produce valid sequence logits; this test is about buffer
            # management, not logprob correctness.
            get_logprobs_fn=lambda model_out, acts: -mx.ones(acts.shape),
        )
        if on_policy is not None:
            kwargs['on_policy'] = on_policy
        return Trainer(**kwargs)

    @staticmethod
    def _make_batch():
        """Create a minimal batch compatible with Linear(4,4)."""
        return {
            'obs': mx.random.normal((4,)),
            'act': mx.array([1, 2, 3]),
            'logprob': mx.array([-1.0, -1.0, -1.0]),
            'rewards': mx.array([1.0, 0.5]),
            'episode_lengths': [2, 1],
        }

    @staticmethod
    def _add_episode(buffer, reward=1.0):
        """Add a single complete episode to the buffer."""
        buffer.add(
            obs=mx.array([1, 2]),
            act=mx.array([3]),
            rew=reward,
            next_obs=mx.array([4, 5]),
            done=True,
            logprob=mx.array([-0.5]),
        )

    def test_default_clears_buffer(self):
        """H1: The default (on_policy=True) clears the buffer after train()."""
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        self._add_episode(buf)
        self._add_episode(buf)
        assert buf.episode_count == 2

        trainer = self._make_trainer(buf)  # uses default on_policy=True
        trainer.train()

        assert buf.episode_count == 0, (
            f"Default on_policy should clear buffer, got {buf.episode_count}"
        )

    def test_on_policy_explicit_clears_buffer(self):
        """H2: Explicit on_policy=True clears the linked buffer after train()."""
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        self._add_episode(buf)
        self._add_episode(buf)

        trainer = self._make_trainer(buf, on_policy=True)
        trainer.train()

        assert buf.episode_count == 0, (
            f"Expected empty buffer after on_policy train(), got {buf.episode_count}"
        )

    def test_off_policy_preserves_buffer(self):
        """H3: on_policy=False does not clear the buffer."""
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        self._add_episode(buf)
        self._add_episode(buf)

        trainer = self._make_trainer(buf, on_policy=False)
        trainer.train()

        assert buf.episode_count == 2, (
            f"off-policy should preserve buffer, got {buf.episode_count}"
        )

    def test_manual_rollout_data_does_not_clear_buffer(self):
        """H4: Passing explicit rollout_data should never clear the buffer,
        even when on_policy=True."""
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        self._add_episode(buf)
        self._add_episode(buf)

        trainer = self._make_trainer(buf, on_policy=True)
        trainer.train(rollout_data=self._make_batch())  # manual mode

        assert buf.episode_count == 2, (
            f"Manual rollout_data should not clear buffer, got {buf.episode_count}"
        )

    def test_on_policy_buffer_stays_empty_across_steps(self):
        """H5: Multiple train() calls with on_policy=True keep buffer empty."""
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        trainer = self._make_trainer(buf, on_policy=True)

        for step in range(5):
            # Simulate rollout: add fresh episodes before each train()
            self._add_episode(buf, reward=float(step))
            self._add_episode(buf, reward=float(step) + 0.5)
            assert buf.episode_count == 2, f"Step {step}: expected 2 episodes before train()"

            trainer.train()

            assert buf.episode_count == 0, (
                f"Step {step}: expected empty buffer after train(), "
                f"got {buf.episode_count}"
            )

    def test_buffer_growth_warning_immediate(self, caplog):
        """H6: Warning fires on first detected growth, not before."""
        import logging
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        trainer = self._make_trainer(buf, on_policy=False)

        # Step 0: add 2 episodes, train — no growth yet (baseline)
        self._add_episode(buf)
        self._add_episode(buf)
        with caplog.at_level(logging.WARNING, logger="textpolicy.training.trainer"):
            trainer.train()

        # No warning should fire on the first step (no prior baseline to compare)
        assert not any("Buffer grew" in r.message for r in caplog.records), (
            "Warning should not fire on first step (no growth yet)"
        )
        caplog.clear()

        # Step 1: add 2 more (buffer grows from 2 → 4), train
        self._add_episode(buf)
        self._add_episode(buf)
        with caplog.at_level(logging.WARNING, logger="textpolicy.training.trainer"):
            trainer.train()

        warnings = [r for r in caplog.records if "Buffer grew" in r.message]
        assert len(warnings) == 1, (
            f"Expected exactly 1 warning on first growth, got {len(warnings)}"
        )

    def test_buffer_growth_warning_late_start(self, caplog):
        """H7: Warning fires even when growth starts at a later step."""
        import logging
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        trainer = self._make_trainer(buf, on_policy=False)

        # Steps 0-4: train with manual rollout_data (buffer stays empty,
        # no growth detected).
        for _ in range(5):
            trainer.train(rollout_data=self._make_batch())

        # Step 5: first time using linked buffer — establishes baseline
        self._add_episode(buf)
        self._add_episode(buf)
        with caplog.at_level(logging.WARNING, logger="textpolicy.training.trainer"):
            trainer.train()

        # Step 6: buffer grows (2 → 4), warning should fire
        self._add_episode(buf)
        self._add_episode(buf)
        with caplog.at_level(logging.WARNING, logger="textpolicy.training.trainer"):
            trainer.train()

        assert any("Buffer grew" in record.message for record in caplog.records), (
            f"Expected late-start growth warning, got: {[r.message for r in caplog.records]}"
        )

    def test_buffer_growth_warning_fires_only_once(self, caplog):
        """H8: Warning fires only once, not on every subsequent step."""
        import logging
        from textpolicy.buffer import Buffer

        buf = Buffer(max_episodes=100)
        trainer = self._make_trainer(buf, on_policy=False)

        with caplog.at_level(logging.WARNING, logger="textpolicy.training.trainer"):
            for _ in range(5):
                self._add_episode(buf)
                trainer.train()

        warning_count = sum(1 for r in caplog.records if "Buffer grew" in r.message)
        assert warning_count == 1, (
            f"Expected exactly 1 warning, got {warning_count}"
        )


# ---------------------------------------------------------------------------
# 11. Integration: on_policy with real select_recent_data (Issue #35, P3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOnPolicyWithSelectRecentData:
    """Verify on_policy works through the real grpo.select_recent_data path.

    Unlike the unit tests above (which inject a lambda data_selector_fn),
    this exercises the full pipeline: Buffer → select_recent_data →
    _pack_episodes → Trainer._loss_fn.
    """

    @staticmethod
    def _add_text_episode(buffer, prompt_tokens, response_tokens, reward, logprobs):
        """Add a text-generation episode via add_episode_from_dict.

        Builds the dict format expected by _pack_episodes:
        obs is a list of prompt token lists (one per step),
        act is a list of response token lists (one per step).
        """
        n_steps = len(response_tokens)
        buffer.add_episode_from_dict({
            'obs': [prompt_tokens] * n_steps,
            'act': [[tok] for tok in response_tokens],
            'rew': [0.0] * (n_steps - 1) + [reward],
            'next_obs': [prompt_tokens] * n_steps,
            'done': [False] * (n_steps - 1) + [True],
            'timeout': [False] * n_steps,
            'logprob': [[lp] for lp in logprobs],
        })

    def test_on_policy_clears_buffer_through_real_pipeline(self):
        """Full pipeline: select_recent_data → _pack_episodes → train → clear."""
        import mlx.optimizers as optim
        from textpolicy.buffer import Buffer
        from textpolicy.training.trainer import Trainer
        from textpolicy.algorithms import grpo

        model = _TinyLM(vocab_size=16, dim=8)
        mx.eval(model.parameters())
        optimizer = optim.Adam(learning_rate=0.01)

        buf = Buffer(max_episodes=100)

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=grpo.policy_loss,
            optimizer=optimizer,
            buffer=buf,
            data_selector_fn=grpo.select_recent_data,
            compile_training=False,
            on_policy=True,
        )

        # Add episodes with tokens within model vocab (0-15)
        self._add_text_episode(buf, [1, 2, 3], [4, 5], reward=1.0, logprobs=[-0.5, -0.3])
        self._add_text_episode(buf, [1, 2, 3], [6, 7], reward=0.0, logprobs=[-0.8, -0.4])
        self._add_text_episode(buf, [1, 2, 3], [8, 9], reward=0.5, logprobs=[-0.6, -0.2])
        assert buf.episode_count == 3

        # Train — this goes through select_recent_data → _pack_episodes → _loss_fn
        metrics = trainer.train()

        assert 'loss' in metrics
        assert buf.episode_count == 0, (
            f"on_policy should clear buffer after real pipeline, got {buf.episode_count}"
        )

    def test_select_recent_data_caps_episodes(self):
        """select_recent_data with on_policy keeps only latest batch each step."""
        import functools
        import mlx.optimizers as optim
        from textpolicy.buffer import Buffer
        from textpolicy.training.trainer import Trainer
        from textpolicy.algorithms import grpo

        model = _TinyLM(vocab_size=16, dim=8)
        mx.eval(model.parameters())
        optimizer = optim.Adam(learning_rate=0.01)

        buf = Buffer(max_episodes=100)

        # Cap select_recent_data to 2 episodes max
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=grpo.policy_loss,
            optimizer=optimizer,
            buffer=buf,
            data_selector_fn=functools.partial(grpo.select_recent_data, max_episodes=2),
            compile_training=False,
            on_policy=True,
        )

        # Simulate two training steps
        for step in range(2):
            self._add_text_episode(buf, [1, 2], [3, 4], reward=float(step), logprobs=[-0.5, -0.3])
            self._add_text_episode(buf, [1, 2], [5, 6], reward=float(step) + 0.5, logprobs=[-0.4, -0.2])

            metrics = trainer.train()
            assert 'loss' in metrics
            assert buf.episode_count == 0, f"Step {step}: buffer should be cleared"
