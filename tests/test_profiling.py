"""
Tests for per-phase training profiling (Issue #25).

TestTimerFix — verifies the Timer per-name start/stop fix and format_breakdown.
TestTrainerProfiling — verifies timing/* keys appear when profile=True and are
absent when profile=False, and that the loss value is unaffected by profiling.
"""

import time
import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.buffer import Buffer
from textpolicy.utils.timing import Timer
from textpolicy.training.trainer import Trainer
from textpolicy.algorithms import grpo


# ---------------------------------------------------------------------------
# Timer tests
# ---------------------------------------------------------------------------

class TestTimerFix:
    """Verify per-name start tracking and format_breakdown."""

    @pytest.mark.unit
    def test_overlapping_timers(self):
        """H1: Concurrent named timers don't overwrite each other."""
        timer = Timer()
        timer.start("outer")
        time.sleep(0.01)
        timer.start("inner")
        time.sleep(0.01)
        inner_dur = timer.stop("inner")
        time.sleep(0.01)
        outer_dur = timer.stop("outer")

        assert inner_dur > 0
        assert outer_dur > inner_dur, "outer must include inner + extra sleep"
        assert timer.get_stats("inner")["count"] == 1
        assert timer.get_stats("outer")["count"] == 1

    @pytest.mark.unit
    def test_stop_without_start_raises(self):
        """H2: Stopping an unstarted timer raises RuntimeError."""
        timer = Timer()
        with pytest.raises(RuntimeError, match="was not started"):
            timer.stop("never_started")

    @pytest.mark.unit
    def test_context_manager_per_name(self):
        """H3: Context manager works independently for different names."""
        timer = Timer()
        with timer.time("a"):
            time.sleep(0.005)
        with timer.time("b"):
            time.sleep(0.005)

        assert timer.get_stats("a")["count"] == 1
        assert timer.get_stats("b")["count"] == 1

    @pytest.mark.unit
    def test_format_breakdown_percentages(self):
        """H4: format_breakdown computes correct percentages."""
        timer = Timer()
        # Manually inject known durations for deterministic testing
        timer.times["total"] = [1.0]
        timer.times["phase_a"] = [0.6]
        timer.times["phase_b"] = [0.3]

        breakdown = timer.format_breakdown("total")
        assert "phase_a" in breakdown
        assert "phase_b" in breakdown
        assert "total" not in breakdown

        assert abs(breakdown["phase_a"]["percent"] - 60.0) < 1e-6
        assert abs(breakdown["phase_b"]["percent"] - 30.0) < 1e-6
        assert abs(breakdown["phase_a"]["seconds"] - 0.6) < 1e-6

    @pytest.mark.unit
    def test_format_breakdown_empty_total(self):
        """H5: format_breakdown returns empty dict when total has no data."""
        timer = Timer()
        assert timer.format_breakdown("total") == {}

    @pytest.mark.unit
    def test_reset_clears_running_timers(self):
        """H6: reset() clears active starts and prevents KeyError on stop."""
        timer = Timer()
        timer.start("phase")
        timer.reset()
        with pytest.raises(RuntimeError, match="was not started"):
            timer.stop("phase")


# ---------------------------------------------------------------------------
# Trainer profiling tests
# ---------------------------------------------------------------------------

_DIM = 8  # hidden dim for the tiny model


def _make_tiny_model():
    """Create a minimal MLX model for profiling tests."""
    return nn.Linear(_DIM, _DIM)


def _make_batch(num_episodes: int = 3, tokens_per_ep: int = 5):
    """
    Create a synthetic flat-1D batch matching Trainer expectations.

    obs is shaped [total_tokens, _DIM] so the linear model can process it.
    act uses integer indices for logprob extraction.
    No ``episode_lengths`` key → the fallback (non-GRPO) path is used inside
    ``_loss_fn``, which calls ``get_logprobs_fn`` directly.
    """
    total = num_episodes * tokens_per_ep
    return {
        "obs": mx.random.normal((total, _DIM)),
        "act": mx.zeros((total,), dtype=mx.int32),   # index 0 for gather
        "logprob": -mx.abs(mx.random.normal((total,))),
        "rewards": mx.random.normal((num_episodes,)),
        # episode_lengths intentionally omitted — keeps the test off the GRPO
        # path which requires real tokenization infrastructure.
    }


def _dummy_get_logprobs(model_output, actions):
    """
    Extract logprobs from model output for the test model.

    model_output: [batch, seq_len, _DIM]  (raw logits from nn.Linear)
    actions: [seq_len] int indices
    """
    logits = model_output
    if logits.ndim == 3:
        logits = logits[0]  # [seq_len, _DIM]
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    seq_len = actions.shape[0]
    return log_probs[mx.arange(seq_len), actions]


def _simple_advantage(rewards):
    """Normalise rewards to zero-mean unit-variance."""
    return (rewards - mx.mean(rewards)) / (mx.std(rewards) + 1e-8)


def _simple_loss(old_lp, new_lp, adv):
    """Minimal surrogate loss for testing."""
    ratio = mx.exp(new_lp - old_lp)
    return -mx.mean(ratio * adv)


class TestTrainerProfiling:
    """Verify that profile=True emits timing/* keys and profile=False does not."""

    @pytest.mark.unit
    def test_timing_keys_present_when_profiled(self):
        """When profile=True, returned metrics must contain timing/* keys."""
        model = _make_tiny_model()
        trainer = Trainer(
            model=model,
            advantage_fn=_simple_advantage,
            loss_fn=_simple_loss,
            optimizer=optim.Adam(learning_rate=1e-4),
            get_logprobs_fn=_dummy_get_logprobs,
            compile_training=False,
            profile=True,
        )

        batch = _make_batch()
        metrics = trainer.train(batch)

        timing_keys = [k for k in metrics if k.startswith("timing/")]
        assert len(timing_keys) > 0, "Expected timing/* keys when profile=True"
        assert "timing/total_s" in metrics
        assert "timing/loss_and_grad_s" in metrics
        assert "timing/loss_and_grad_pct" in metrics

    @pytest.mark.unit
    def test_no_timing_keys_when_not_profiled(self):
        """When profile=False (default), no timing/* keys should appear."""
        model = _make_tiny_model()
        trainer = Trainer(
            model=model,
            advantage_fn=_simple_advantage,
            loss_fn=_simple_loss,
            optimizer=optim.Adam(learning_rate=1e-4),
            get_logprobs_fn=_dummy_get_logprobs,
            compile_training=False,
            profile=False,
        )

        batch = _make_batch()
        metrics = trainer.train(batch)

        timing_keys = [k for k in metrics if k.startswith("timing/")]
        assert len(timing_keys) == 0, f"Unexpected timing keys: {timing_keys}"

    @pytest.mark.unit
    def test_loss_unaffected_by_profiling(self):
        """Profiling must not change the computed loss value."""
        mx.random.seed(42)
        batch = _make_batch()

        losses = {}
        for label, do_profile in [("off", False), ("on", True)]:
            mx.random.seed(0)
            model = _make_tiny_model()
            trainer = Trainer(
                model=model,
                advantage_fn=_simple_advantage,
                loss_fn=_simple_loss,
                optimizer=optim.Adam(learning_rate=1e-4),
                get_logprobs_fn=_dummy_get_logprobs,
                compile_training=False,
                profile=do_profile,
            )
            metrics = trainer.train(batch)
            losses[label] = metrics["loss"]

        assert abs(losses["on"] - losses["off"]) < 1e-5, (
            f"Loss diverged: profile=off → {losses['off']}, "
            f"profile=on → {losses['on']}"
        )

    @pytest.mark.unit
    def test_timing_values_are_positive(self):
        """All timing values must be non-negative floats."""
        model = _make_tiny_model()
        trainer = Trainer(
            model=model,
            advantage_fn=_simple_advantage,
            loss_fn=_simple_loss,
            optimizer=optim.Adam(learning_rate=1e-4),
            get_logprobs_fn=_dummy_get_logprobs,
            compile_training=False,
            profile=True,
        )

        batch = _make_batch()
        metrics = trainer.train(batch)

        for k, v in metrics.items():
            if k.startswith("timing/") and k.endswith("_s"):
                assert v >= 0.0, f"{k} has negative value: {v}"
            if k.startswith("timing/") and k.endswith("_pct"):
                assert 0.0 <= v <= 100.0 + 1e-6, f"{k} out of range: {v}"

    @pytest.mark.unit
    def test_profiled_timer_does_not_accumulate_unbounded_history(self):
        """Trainer should reset timing history per step."""
        model = _make_tiny_model()
        trainer = Trainer(
            model=model,
            advantage_fn=_simple_advantage,
            loss_fn=_simple_loss,
            optimizer=optim.Adam(learning_rate=1e-4),
            compile_training=False,
            profile=True,
        )

        batch = _make_batch()
        trainer.train(batch)
        assert trainer._timer is not None
        assert trainer._timer.get_stats("total")["count"] == 1

        trainer.train(batch)
        assert trainer._timer.get_stats("total")["count"] == 1

    @pytest.mark.unit
    def test_buffer_without_logprob_raises_clear_error(self):
        """Training from a buffer with missing logprob should fail clearly."""
        model = _make_tiny_model()
        trainer = Trainer(
            model=model,
            advantage_fn=_simple_advantage,
            loss_fn=_simple_loss,
            optimizer=optim.Adam(learning_rate=1e-4),
            get_logprobs_fn=_dummy_get_logprobs,
            compile_training=False,
        )

        buf = Buffer(max_episodes=10)
        # Complete one episode without logprob values.
        for j in range(3):
            buf.add(
                obs=mx.array([1.0] * _DIM),
                act=mx.array([0], dtype=mx.int32),
                rew=1.0,
                next_obs=mx.array([1.0] * _DIM),
                done=(j == 2),
            )

        with pytest.raises(ValueError, match="missing logprob"):
            trainer.train(buf)


# ---------------------------------------------------------------------------
# Rollout profiling tests
# ---------------------------------------------------------------------------

from textpolicy.rollout.runner import RolloutRunner
from textpolicy.rollout.strategy import create_strategy


class _SingleTurnEnv:
    """Minimal single-turn env that terminates every episode."""

    def __init__(self):
        self._idx = 0

    def reset(self):
        self._idx += 1
        return mx.array([float(self._idx)]), {}

    def step(self, action):
        return {
            "observation": mx.array([0.0]),
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "info": {},
        }


def _dummy_policy(obs, deterministic=False):
    return mx.array(0), {"logprob": mx.array([-0.5])}


def _dummy_batched_policy(obs_list):
    return [
        (mx.array([1], dtype=mx.int32), {"logprob": mx.array([-0.5])})
        for _ in obs_list
    ]


class TestRolloutProfiling:
    """Verify rollout sub-phase timing via RolloutRunner.profile."""

    @pytest.mark.unit
    def test_rollout_timing_keys_present_when_profiled(self):
        """When profile=True, get_timing() must contain expected phase keys."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=3, profile=True,
        )
        runner.collect()
        timing = runner.get_timing()

        expected_keys = {"total", "generation", "env_step", "buffer_store", "env_reset"}
        assert expected_keys.issubset(timing.keys()), (
            f"Missing keys: {expected_keys - timing.keys()}"
        )

    @pytest.mark.unit
    def test_no_timing_when_not_profiled(self):
        """When profile=False (default), get_timing() returns empty dict."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=2, profile=False,
        )
        runner.collect()
        assert runner.get_timing() == {}

    @pytest.mark.unit
    def test_timing_values_positive(self):
        """All timing values must be > 0 after a collect call."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=3, profile=True,
        )
        runner.collect()
        timing = runner.get_timing()

        for phase, secs in timing.items():
            assert secs > 0.0, f"{phase} has non-positive timing: {secs}"

    @pytest.mark.unit
    def test_batched_timing_keys_present(self):
        """collect_batched path should also produce expected timing phases."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=4, profile=True,
        )
        runner.collect_batched(_dummy_batched_policy, batch_size=2)
        timing = runner.get_timing()

        expected_keys = {"total", "generation", "env_step", "buffer_store", "env_reset"}
        assert expected_keys.issubset(timing.keys()), (
            f"Missing keys: {expected_keys - timing.keys()}"
        )
        for phase, secs in timing.items():
            assert secs > 0.0, f"{phase} has non-positive timing: {secs}"

    @pytest.mark.unit
    def test_reset_timing_clears_data(self):
        """reset_timing() should clear all accumulated timing data."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=2, profile=True,
        )
        runner.collect()
        assert runner.get_timing() != {}

        runner.reset_timing()
        assert runner.get_timing() == {}

    @pytest.mark.unit
    def test_timer_is_none_when_not_profiled(self):
        """When profile=False, _timer must be None (zero overhead)."""
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env, policy=_dummy_policy, strategy=create_strategy("grpo"),
            max_steps=1,
        )
        assert runner._timer is None
