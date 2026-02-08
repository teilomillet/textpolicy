"""
Tests for training metrics collection and analysis (textpolicy/training/metrics.py).

This module was previously untested. These tests verify:
  - TrainingMetrics: rolling average, update, summary, reset
  - RolloutMetrics: episode tracking, summary stats, reset
  - compute_explained_variance: perfect, zero, partial, edge cases
  - compute_policy_metrics: ratio stats, clip fraction, KL divergence
  - Public API: all functions importable from textpolicy.training
"""

import pytest
import mlx.core as mx

from textpolicy.training.metrics import (
    TrainingMetrics,
    RolloutMetrics,
    compute_explained_variance,
    compute_policy_metrics,
    log_metrics,
)


# ---------------------------------------------------------------------------
# TrainingMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingMetrics:
    """Validate the lightweight rolling-window metrics collector."""

    def test_update_and_get_latest(self):
        """Latest value is the most recently added."""
        m = TrainingMetrics()
        m.update({"loss": 1.0})
        m.update({"loss": 0.5})
        assert m.get_latest("loss") == 0.5

    def test_get_latest_missing_key(self):
        """Querying a key that was never recorded returns None."""
        m = TrainingMetrics()
        assert m.get_latest("nonexistent") is None

    def test_get_mean_all_values(self):
        """Mean across all values in the window."""
        m = TrainingMetrics()
        for v in [1.0, 2.0, 3.0]:
            m.update({"loss": v})
        assert m.get_mean("loss") == pytest.approx(2.0)

    def test_get_mean_last_n(self):
        """Mean over the last N values only."""
        m = TrainingMetrics()
        for v in [10.0, 1.0, 2.0, 3.0]:
            m.update({"loss": v})
        # Last 2: [2.0, 3.0] → mean=2.5
        assert m.get_mean("loss", last_n=2) == pytest.approx(2.5)

    def test_history_length_caps_window(self):
        """Old values are evicted when the history window is exceeded."""
        m = TrainingMetrics(history_length=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            m.update({"loss": v})
        # Window keeps last 3: [3.0, 4.0, 5.0]
        assert m.get_mean("loss") == pytest.approx(4.0)
        assert m.get_latest("loss") == 5.0

    def test_get_summary(self):
        """Summary returns latest, mean, min, max, count."""
        m = TrainingMetrics()
        for v in [3.0, 1.0, 4.0]:
            m.update({"loss": v})
        summary = m.get_summary()
        loss_stats = summary["metrics"]["loss"]
        assert loss_stats["latest"] == 4.0
        assert loss_stats["min"] == 1.0
        assert loss_stats["max"] == 4.0
        assert loss_stats["count"] == 3
        assert loss_stats["mean"] == pytest.approx(8.0 / 3)

    def test_tracks_total_steps(self):
        """Step count is updated when 'step' key is present."""
        m = TrainingMetrics()
        m.update({"step": 10, "loss": 0.5})
        assert m.total_steps == 10
        m.update({"step": 20, "loss": 0.3})
        assert m.total_steps == 20

    def test_reset_clears_everything(self):
        """Reset empties all metrics and resets step counter."""
        m = TrainingMetrics()
        m.update({"loss": 1.0, "step": 5})
        m.reset()
        assert m.get_latest("loss") is None
        assert m.total_steps == 0
        assert len(m) == 0

    def test_len_counts_tracked_metrics(self):
        """__len__ returns the number of distinct metric keys."""
        m = TrainingMetrics()
        m.update({"loss": 1.0, "reward": 2.0, "lr": 0.001})
        assert len(m) == 3


# ---------------------------------------------------------------------------
# RolloutMetrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRolloutMetrics:
    """Validate rollout phase metrics tracking."""

    def test_add_episodes_and_summary(self):
        """Adding episodes updates all tracked statistics."""
        rm = RolloutMetrics()
        rm.add_episode(reward=1.0, length=10)
        rm.add_episode(reward=0.0, length=5)
        rm.add_episode(reward=0.5, length=8)

        s = rm.get_summary()
        assert s["episodes_collected"] == 3
        assert s["total_reward"] == pytest.approx(1.5)
        assert s["mean_reward"] == pytest.approx(0.5)
        assert s["mean_length"] == pytest.approx(23.0 / 3)
        assert s["min_reward"] == 0.0
        assert s["max_reward"] == 1.0
        assert s["min_length"] == 5
        assert s["max_length"] == 10

    def test_empty_summary(self):
        """Summary before any episodes returns sensible zeros."""
        rm = RolloutMetrics()
        s = rm.get_summary()
        assert s["episodes_collected"] == 0
        assert s["mean_reward"] == 0.0
        assert s["mean_length"] == 0.0

    def test_reset(self):
        """Reset returns metrics to initial state."""
        rm = RolloutMetrics()
        rm.add_episode(reward=1.0, length=10)
        rm.reset()
        assert rm.episodes_collected == 0
        assert rm.get_summary()["episodes_collected"] == 0


# ---------------------------------------------------------------------------
# compute_explained_variance
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeExplainedVariance:
    """Validate explained variance: 1 - Var(residual) / Var(target)."""

    def test_perfect_prediction(self):
        """Predicted == targets → explained variance = 1.0."""
        targets = mx.array([1.0, 2.0, 3.0, 4.0])
        predicted = mx.array([1.0, 2.0, 3.0, 4.0])
        ev = compute_explained_variance(predicted, targets)
        assert ev == pytest.approx(1.0, abs=1e-5)

    def test_mean_prediction(self):
        """Predicting the mean → explained variance = 0.0."""
        targets = mx.array([1.0, 2.0, 3.0, 4.0])
        mean_val = 2.5
        predicted = mx.full((4,), mean_val)
        ev = compute_explained_variance(predicted, targets)
        assert ev == pytest.approx(0.0, abs=1e-5)

    def test_partial_prediction(self):
        """Explained variance between 0 and 1 for partial fit."""
        targets = mx.array([1.0, 2.0, 3.0, 4.0])
        predicted = mx.array([1.5, 2.5, 2.5, 3.5])
        ev = compute_explained_variance(predicted, targets)
        assert 0.0 < ev < 1.0

    def test_constant_targets_returns_zero(self):
        """Var(targets) = 0 → returns 0.0 (avoid division by zero)."""
        targets = mx.array([5.0, 5.0, 5.0])
        predicted = mx.array([3.0, 4.0, 5.0])
        ev = compute_explained_variance(predicted, targets)
        assert ev == 0.0

    def test_worse_than_mean(self):
        """Predictions worse than the mean → negative explained variance."""
        targets = mx.array([1.0, 2.0, 3.0])
        # Predictions anti-correlated
        predicted = mx.array([3.0, 2.0, 1.0])
        ev = compute_explained_variance(predicted, targets)
        assert ev < 0.0


# ---------------------------------------------------------------------------
# compute_policy_metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputePolicyMetrics:
    """Validate standard PPO/GRPO policy diagnostics."""

    def test_identical_logprobs(self):
        """Same old/new logprobs → ratio=1, clip_fraction=0, KL=0."""
        lp = mx.array([-1.0, -2.0, -1.5, -0.8])
        metrics = compute_policy_metrics(lp, lp)

        assert metrics["policy/ratio_mean"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["policy/clip_fraction"] == pytest.approx(0.0, abs=1e-5)
        assert metrics["policy/kl_divergence"] == pytest.approx(0.0, abs=1e-5)

    def test_shifted_logprobs_nonzero_kl(self):
        """Different logprobs → nonzero KL divergence and ratio != 1."""
        old_lp = mx.array([-1.0, -1.0, -1.0, -1.0])
        new_lp = mx.array([-0.5, -0.5, -0.5, -0.5])  # policy moved toward higher prob
        metrics = compute_policy_metrics(old_lp, new_lp)

        # Ratio = exp(-0.5 - (-1.0)) = exp(0.5) ≈ 1.6487
        assert metrics["policy/ratio_mean"] == pytest.approx(1.6487, abs=1e-3)
        # KL = mean(old - new) = mean(-1 - (-0.5)) = -0.5
        assert metrics["policy/kl_divergence"] == pytest.approx(-0.5, abs=1e-5)

    def test_clip_fraction_detects_clipping(self):
        """Large policy change → tokens get clipped."""
        old_lp = mx.array([-1.0, -1.0, -1.0, -1.0])
        # Huge shift: ratio = exp(2.0) ≈ 7.4 — well above 1+0.2
        new_lp = mx.array([1.0, 1.0, 1.0, 1.0])
        metrics = compute_policy_metrics(old_lp, new_lp, clip_ratio=0.2)
        assert metrics["policy/clip_fraction"] == pytest.approx(1.0, abs=1e-5)

    def test_returns_all_expected_keys(self):
        """Output dictionary must contain the standard metric keys."""
        lp = mx.array([-1.0, -2.0])
        metrics = compute_policy_metrics(lp, lp)
        expected_keys = {
            "policy/ratio_mean",
            "policy/ratio_std",
            "policy/clip_fraction",
            "policy/kl_divergence",
            "policy/entropy",
        }
        assert set(metrics.keys()) == expected_keys

    def test_entropy_sign(self):
        """Entropy should be positive (it's -E[log(p)])."""
        # logprobs are negative → -mean(logprobs) is positive
        lp = mx.array([-1.0, -2.0, -1.5])
        metrics = compute_policy_metrics(lp, lp)
        assert metrics["policy/entropy"] > 0


# ---------------------------------------------------------------------------
# log_metrics (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogMetrics:
    """Verify log_metrics doesn't crash and calls external logger."""

    def test_console_output(self, capsys):
        """log_metrics prints to console."""
        log_metrics({"loss": 0.5, "reward": 1.0}, step=10)
        captured = capsys.readouterr()
        assert "Step 10" in captured.out
        assert "loss" in captured.out

    def test_external_logger_called(self):
        """External logger's .log() method is invoked."""
        calls = []

        class FakeLogger:
            def log(self, metrics, step):
                calls.append((metrics, step))

        log_metrics({"loss": 0.5}, step=5, logger=FakeLogger(), prefix="train/")
        assert len(calls) == 1
        assert calls[0][1] == 5
        assert "train/loss" in calls[0][0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTrainingMetricsPublicAPI:
    """Verify all metrics are importable from the training package."""

    def test_importable_from_training_package(self):
        """All metrics must be importable from textpolicy.training."""
        from textpolicy.training import (
            TrainingMetrics,
            RolloutMetrics,
            log_metrics,
            compute_explained_variance,
            compute_policy_metrics,
        )
        assert callable(compute_explained_variance)
        assert callable(compute_policy_metrics)
        assert callable(log_metrics)
        assert callable(TrainingMetrics)
        assert callable(RolloutMetrics)

    def test_listed_in_all(self):
        """All metrics names must appear in __all__."""
        import textpolicy.training as training
        for name in [
            "TrainingMetrics",
            "RolloutMetrics",
            "log_metrics",
            "compute_explained_variance",
            "compute_policy_metrics",
        ]:
            assert name in training.__all__, f"{name} missing from training.__all__"
