"""
Exact-output regression tests for optimization safety (Issue #24).

These tests lock down the exact numerical output of all hot-path functions
*before* any optimization work begins (Issue #23). Every expected value was
captured from the current implementation with fixed inputs; if a refactor
silently changes semantics, these tests catch it.

Design:
    - Pure functions → hardcoded expected values (deterministic on same HW).
    - Compiled variants → side-by-side comparison with non-compiled.
    - Trainer tests → structural invariants only (TinyLM has random weights).
    - Tolerance: atol=1e-6 for pure functions, atol=1e-5 for compiled.

Hypotheses per class:
    H1: Exact values match frozen constants
    H2: Compiled variants match non-compiled
    H3: Flat 1D invariant is preserved
    H4: Gradient flow is correct (where applicable)
"""

from types import SimpleNamespace

import pytest
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.algorithms import grpo, gspo
from textpolicy.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Minimal model for Trainer structural tests (16-vocab, 8-dim)."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def __call__(self, x):
        return self.head(self.embed(x))


def _has_nonzero_grads(grads) -> bool:
    """Recursively check that at least one gradient array has a nonzero element."""
    if isinstance(grads, mx.array):
        return bool(mx.any(grads != 0).item())
    if isinstance(grads, dict):
        return any(_has_nonzero_grads(v) for v in grads.values())
    if isinstance(grads, (list, tuple)):
        return any(_has_nonzero_grads(v) for v in grads)
    return False


def _all_grads_finite(grads) -> bool:
    """Recursively check that all gradient arrays are finite (no nan/inf)."""
    if isinstance(grads, mx.array):
        return bool((~mx.isnan(grads)).all().item() and (~mx.isinf(grads)).all().item())
    if isinstance(grads, dict):
        return all(_all_grads_finite(v) for v in grads.values())
    if isinstance(grads, (list, tuple)):
        return all(_all_grads_finite(v) for v in grads)
    return True


# ---------------------------------------------------------------------------
# Frozen inputs reused across classes
# ---------------------------------------------------------------------------

# GRPO advantage inputs
_REWARDS_4 = [1.0, 0.5, 0.0, -0.5]

# Policy loss inputs (8 tokens, 4 positive + 4 negative advantages)
_OLD_LP_8 = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9, -1.3, -0.7, -1.0], dtype=mx.float32)
_NEW_LP_8 = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0, -1.2, -0.8, -0.9], dtype=mx.float32)
_ADV_8 = mx.array([0.75, 0.75, 0.25, 0.25, -0.25, -0.25, -0.75, -0.75], dtype=mx.float32)

# Entropy weighting inputs
_ADV_5 = mx.array([0.5, -0.3, 0.2, -0.1, 0.4], dtype=mx.float32)
_ENTROPIES_5 = mx.array([1.0, 5.0, 0.5, 3.0, 2.0], dtype=mx.float32)

# GSPO inputs
_OLD_LP_5 = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9], dtype=mx.float32)
_NEW_LP_5 = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0], dtype=mx.float32)
_SEQ_LENS_2 = [2, 3]
_ADV_GSPO_2 = mx.array([0.5, -0.3], dtype=mx.float32)


# ===========================================================================
# 1. GRPO Advantages
# ===========================================================================

@pytest.mark.unit
class TestRegressionGRPOAdvantages:
    """Freeze exact output of grpo.compute_advantages."""

    # Frozen: compute_advantages([1.0, 0.5, 0.0, -0.5]) → [0.75, 0.25, -0.25, -0.75]
    EXPECTED = [0.75, 0.25, -0.25, -0.75]

    def test_exact_advantages(self):
        """H1: Exact values match frozen constants."""
        adv = grpo.compute_advantages(_REWARDS_4)
        mx.eval(adv)
        for i, (got, want) in enumerate(zip(adv.tolist(), self.EXPECTED)):
            assert abs(got - want) < 1e-6, f"advantages[{i}]: got {got}, want {want}"

    def test_compiled_matches_non_compiled(self):
        """H2: compute_advantages_compiled matches compute_advantages."""
        rewards_mx = mx.array(_REWARDS_4, dtype=mx.float32)
        adv = grpo.compute_advantages(_REWARDS_4)
        adv_c = grpo.compute_advantages_compiled(rewards_mx)
        mx.eval(adv, adv_c)
        assert mx.allclose(adv, adv_c, atol=1e-5), (
            f"Compiled mismatch: {adv.tolist()} vs {adv_c.tolist()}"
        )

    def test_flat_1d_output(self):
        """H3: Output is flat 1D array."""
        adv = grpo.compute_advantages(_REWARDS_4)
        mx.eval(adv)
        assert adv.ndim == 1, f"Expected 1D, got {adv.ndim}D"
        assert adv.shape[0] == len(_REWARDS_4)

    def test_mean_is_zero(self):
        """Advantages are zero-centered (group-relative)."""
        adv = grpo.compute_advantages(_REWARDS_4)
        mx.eval(adv)
        assert abs(mx.mean(adv).item()) < 1e-6, "Advantages should be zero-mean"


# ===========================================================================
# 2. GRPO Policy Loss
# ===========================================================================

@pytest.mark.unit
class TestRegressionGRPOPolicyLoss:
    """Freeze exact output of grpo.policy_loss under 3 normalization modes."""

    # Frozen values from current implementation
    EXPECTED_DAPO = -0.010896742343902588
    EXPECTED_SYMMETRIC = -0.008890241384506226
    EXPECTED_CONSTANT_NORM = -0.021793484687805176

    def test_exact_dapo_loss(self):
        """H1: DAPO defaults (asymmetric clip) produce frozen loss."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8)
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_DAPO) < 1e-6, (
            f"DAPO loss: got {loss.item()}, want {self.EXPECTED_DAPO}"
        )

    def test_exact_symmetric_loss(self):
        """H1: Symmetric clip_ratio=0.2 produces frozen loss."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8, clip_ratio=0.2)
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_SYMMETRIC) < 1e-6, (
            f"Symmetric loss: got {loss.item()}, want {self.EXPECTED_SYMMETRIC}"
        )

    def test_exact_constant_norm_loss(self):
        """H1: normalize_constant=4 produces frozen loss."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8, normalize_constant=4)
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_CONSTANT_NORM) < 1e-6, (
            f"Constant-norm loss: got {loss.item()}, want {self.EXPECTED_CONSTANT_NORM}"
        )

    def test_compiled_mean_matches(self):
        """H2: policy_loss_compiled matches policy_loss (DAPO defaults)."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8)
        loss_c = grpo.policy_loss_compiled(_OLD_LP_8, _NEW_LP_8, _ADV_8)
        mx.eval(loss, loss_c)
        assert abs(loss.item() - loss_c.item()) < 1e-5, (
            f"Compiled mean mismatch: {loss.item()} vs {loss_c.item()}"
        )

    def test_compiled_constant_norm_matches(self):
        """H2: policy_loss_compiled_constant_norm matches policy_loss."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8, normalize_constant=4)
        loss_c = grpo.policy_loss_compiled_constant_norm(
            _OLD_LP_8, _NEW_LP_8, _ADV_8, normalize_constant=4.0
        )
        mx.eval(loss, loss_c)
        assert abs(loss.item() - loss_c.item()) < 1e-5, (
            f"Compiled constant-norm mismatch: {loss.item()} vs {loss_c.item()}"
        )

    def test_loss_is_scalar(self):
        """Loss output is a scalar (0-D array)."""
        loss = grpo.policy_loss(_OLD_LP_8, _NEW_LP_8, _ADV_8)
        mx.eval(loss)
        assert loss.ndim == 0, f"Expected scalar, got {loss.ndim}D"

    def test_gradient_flows(self):
        """H4: Gradients exist and are finite through policy_loss."""
        def loss_fn(new_lp):
            return grpo.policy_loss(_OLD_LP_8, new_lp, _ADV_8)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(_NEW_LP_8)
        mx.eval(grads)
        assert grads.shape == _NEW_LP_8.shape, "Grad shape mismatch"
        assert not mx.any(mx.isnan(grads)).item(), "NaN in gradients"
        assert mx.any(grads != 0).item(), "All-zero gradients"


# ===========================================================================
# 3. Entropy Weighting (GTPO)
# ===========================================================================

@pytest.mark.unit
class TestRegressionEntropyWeighting:
    """Freeze exact output of grpo.apply_entropy_weighting."""

    # Frozen: apply_entropy_weighting([0.5, -0.3, 0.2, -0.1, 0.4], [1, 5, 0.5, 3, 2], β=0.1)
    EXPECTED = [
        0.47173914313316345,
        -0.3352174162864685,
        0.18434782326221466,
        -0.10304348915815353,
        0.3947826325893402,
    ]

    def test_exact_weighted_advantages(self):
        """H1: Exact values match frozen constants."""
        weighted = grpo.apply_entropy_weighting(_ADV_5, _ENTROPIES_5, entropy_weight=0.1)
        mx.eval(weighted)
        for i, (got, want) in enumerate(zip(weighted.tolist(), self.EXPECTED)):
            assert abs(got - want) < 1e-6, f"weighted[{i}]: got {got}, want {want}"

    def test_stop_gradient_verified(self):
        """H4: mx.stop_gradient prevents gradient flow through entropy weights.

        Gradients w.r.t. token_entropies must be zero — the model must not
        learn to game entropy to amplify its own advantage signal (GTPO Remark 2.5).
        """
        def loss_fn(entropies):
            weighted = grpo.apply_entropy_weighting(_ADV_5, entropies, entropy_weight=0.1)
            return mx.sum(weighted)

        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(_ENTROPIES_5)
        mx.eval(grads)
        assert mx.allclose(grads, mx.zeros_like(grads), atol=1e-7), (
            f"Entropy gradients should be zero, got {grads.tolist()}"
        )

    def test_flat_1d_output(self):
        """H3: Output is flat 1D and same shape as input."""
        weighted = grpo.apply_entropy_weighting(_ADV_5, _ENTROPIES_5, entropy_weight=0.1)
        mx.eval(weighted)
        assert weighted.ndim == 1, f"Expected 1D, got {weighted.ndim}D"
        assert weighted.shape == _ADV_5.shape

    def test_compute_advantages_gtpo_exact(self):
        """H1: compute_advantages_gtpo with episode_lengths matches frozen output."""
        expected = [
            0.4653846025466919,
            0.48076921701431274,
            0.4961538314819336,
            0.0, 0.0, 0.0, 0.0,
            -0.5038461685180664,
            -0.5192307829856873,
            -0.5346153974533081,
        ]
        rewards = [1.0, 0.5, 0.0]
        entropies = mx.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5], dtype=mx.float32
        )
        episode_lengths = [3, 4, 3]

        result = grpo.compute_advantages_gtpo(
            rewards, entropies, entropy_weight=0.1, episode_lengths=episode_lengths
        )
        mx.eval(result)
        assert result.ndim == 1, f"Expected 1D, got {result.ndim}D"
        assert result.shape[0] == sum(episode_lengths)
        for i, (got, want) in enumerate(zip(result.tolist(), expected)):
            assert abs(got - want) < 1e-6, f"gtpo[{i}]: got {got}, want {want}"


# ===========================================================================
# 4. _pack_episodes
# ===========================================================================

@pytest.mark.unit
class TestRegressionPackEpisodes:
    """Freeze exact output of grpo._pack_episodes."""

    @staticmethod
    def _make_episodes():
        ep1 = SimpleNamespace(obs=[[1, 2, 3]], act=[[4, 5]], rew=[1.0], logprob=[-0.5, -0.6])
        ep2 = SimpleNamespace(obs=[[6, 7]], act=[[8, 9, 10]], rew=[0.5], logprob=[-0.3, -0.4, -0.7])
        return [ep1, ep2]

    def test_exact_rewards(self):
        """H1: Episode rewards are correctly summed."""
        result = grpo._pack_episodes(self._make_episodes())
        mx.eval(result['rewards'])
        assert result['rewards'].tolist() == [1.0, 0.5]

    def test_exact_episode_lengths(self):
        """H1: Episode lengths match flattened action counts."""
        result = grpo._pack_episodes(self._make_episodes())
        assert result['episode_lengths'] == [2, 3]

    def test_flat_1d_logprobs(self):
        """H3: Logprobs are flat 1D after packing."""
        result = grpo._pack_episodes(self._make_episodes())
        mx.eval(result['logprob'])
        assert result['logprob'].ndim == 1, f"Expected 1D, got {result['logprob'].ndim}D"

    def test_boundary_invariant(self):
        """Episode lengths sum to total logprob tokens (after padding)."""
        result = grpo._pack_episodes(self._make_episodes())
        mx.eval(result['logprob'])
        # Each episode's logprobs are padded to max length, then concatenated
        # ep1: 2 logprobs padded to 3, ep2: 3 logprobs → total 6
        total_logprob_tokens = result['logprob'].shape[0]
        max_len = max(result['episode_lengths'])
        expected_total = max_len * len(result['episode_lengths'])
        assert total_logprob_tokens == expected_total, (
            f"Logprob tokens {total_logprob_tokens} != padded total {expected_total}"
        )

    def test_empty_episodes(self):
        """Empty input returns empty arrays with correct dtypes."""
        result = grpo._pack_episodes([])
        mx.eval(result['rewards'], result['obs'], result['logprob'])
        assert result['rewards'].shape == (0,)
        assert result['episode_lengths'] == []
        assert result['obs'].shape == (0,)


# ===========================================================================
# 5. GSPO
# ===========================================================================

@pytest.mark.unit
class TestRegressionGSPO:
    """Freeze exact output of gspo functions."""

    # Frozen values
    EXPECTED_SEQ_WEIGHTS = [1.0512712001800537, 0.9672160744667053]
    EXPECTED_HYBRID_WEIGHTS = [
        0.9753099679946899,
        1.1331485509872437,
        0.9355069994926453,
        1.0338951349258423,
        0.9355069994926453,
    ]
    EXPECTED_LOSS_SEQUENCE = -0.11773538589477539
    EXPECTED_LOSS_HYBRID = -0.03655129671096802
    EXPECTED_LOSS_TOKEN = -0.03559298440814018

    def test_exact_sequence_weights(self):
        """H1: compute_sequence_importance_weights matches frozen output."""
        weights = gspo.compute_sequence_importance_weights(
            _OLD_LP_5, _NEW_LP_5, _SEQ_LENS_2, clip_ratio=0.2
        )
        mx.eval(weights)
        for i, (got, want) in enumerate(zip(weights.tolist(), self.EXPECTED_SEQ_WEIGHTS)):
            assert abs(got - want) < 1e-6, f"seq_weight[{i}]: got {got}, want {want}"

    def test_exact_hybrid_weights(self):
        """H1: compute_hybrid_importance_weights matches frozen output."""
        weights = gspo.compute_hybrid_importance_weights(
            _OLD_LP_5, _NEW_LP_5, _SEQ_LENS_2, alpha=0.5, beta=0.5
        )
        mx.eval(weights)
        for i, (got, want) in enumerate(zip(weights.tolist(), self.EXPECTED_HYBRID_WEIGHTS)):
            assert abs(got - want) < 1e-5, f"hybrid_weight[{i}]: got {got}, want {want}"

    def test_exact_loss_sequence(self):
        """H1: gspo_policy_loss 'sequence' variant matches frozen output."""
        loss = gspo.gspo_policy_loss(
            _OLD_LP_5, _NEW_LP_5, _ADV_GSPO_2, _SEQ_LENS_2, variant="sequence"
        )
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_LOSS_SEQUENCE) < 1e-6, (
            f"GSPO sequence loss: got {loss.item()}, want {self.EXPECTED_LOSS_SEQUENCE}"
        )

    def test_exact_loss_hybrid(self):
        """H1: gspo_policy_loss 'hybrid' variant matches frozen output."""
        loss = gspo.gspo_policy_loss(
            _OLD_LP_5, _NEW_LP_5, _ADV_GSPO_2, _SEQ_LENS_2, variant="hybrid"
        )
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_LOSS_HYBRID) < 1e-6, (
            f"GSPO hybrid loss: got {loss.item()}, want {self.EXPECTED_LOSS_HYBRID}"
        )

    def test_exact_loss_token(self):
        """H1: gspo_policy_loss 'token' variant matches frozen output."""
        loss = gspo.gspo_policy_loss(
            _OLD_LP_5, _NEW_LP_5, _ADV_GSPO_2, _SEQ_LENS_2, variant="token"
        )
        mx.eval(loss)
        assert abs(loss.item() - self.EXPECTED_LOSS_TOKEN) < 1e-6, (
            f"GSPO token loss: got {loss.item()}, want {self.EXPECTED_LOSS_TOKEN}"
        )

    def test_sequence_weights_within_clip_bounds(self):
        """Sequence weights should respect clip bounds [1-ε, 1+ε]."""
        clip_ratio = 0.2
        weights = gspo.compute_sequence_importance_weights(
            _OLD_LP_5, _NEW_LP_5, _SEQ_LENS_2, clip_ratio=clip_ratio
        )
        mx.eval(weights)
        for w in weights.tolist():
            assert 1.0 - clip_ratio - 1e-6 <= w <= 1.0 + clip_ratio + 1e-6, (
                f"Weight {w} outside clip bounds [{1-clip_ratio}, {1+clip_ratio}]"
            )


# ===========================================================================
# 6. _expand_advantages
# ===========================================================================

@pytest.mark.unit
class TestRegressionExpandAdvantages:
    """Freeze exact output of Trainer._expand_advantages."""

    @staticmethod
    def _make_trainer():
        model = _TinyLM()
        opt = optim.Adam(learning_rate=1e-3)
        return Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=grpo.policy_loss,
            optimizer=opt,
            compile_training=False,
        )

    def test_exact_uniform_expansion(self):
        """H1: Uniform episode lengths produce correct repeated pattern."""
        trainer = self._make_trainer()
        adv = mx.array([0.5, -0.3, 0.2], dtype=mx.float32)
        expanded = trainer._expand_advantages(adv, [3, 3, 3])
        mx.eval(expanded)

        expected = [0.5, 0.5, 0.5, -0.3, -0.3, -0.3, 0.2, 0.2, 0.2]
        for i, (got, want) in enumerate(zip(expanded.tolist(), expected)):
            assert abs(got - want) < 1e-5, f"uniform[{i}]: got {got}, want {want}"

    def test_exact_variable_expansion(self):
        """H1: Variable episode lengths produce correct repeated pattern."""
        trainer = self._make_trainer()
        adv = mx.array([0.5, -0.3, 0.2], dtype=mx.float32)
        expanded = trainer._expand_advantages(adv, [2, 4, 3])
        mx.eval(expanded)

        expected = [0.5, 0.5, -0.3, -0.3, -0.3, -0.3, 0.2, 0.2, 0.2]
        for i, (got, want) in enumerate(zip(expanded.tolist(), expected)):
            assert abs(got - want) < 1e-5, f"variable[{i}]: got {got}, want {want}"

    def test_flat_1d_output(self):
        """H3: Expanded advantages are flat 1D."""
        trainer = self._make_trainer()
        adv = mx.array([0.5, -0.3, 0.2], dtype=mx.float32)
        expanded = trainer._expand_advantages(adv, [2, 4, 3])
        mx.eval(expanded)
        assert expanded.ndim == 1, f"Expected 1D, got {expanded.ndim}D"
        assert expanded.shape[0] == 2 + 4 + 3

    def test_sum_matches_weighted_total(self):
        """Expanded sum equals Σ(advantage_i × length_i)."""
        trainer = self._make_trainer()
        adv = mx.array([0.5, -0.3, 0.2], dtype=mx.float32)
        lengths = [2, 4, 3]
        expanded = trainer._expand_advantages(adv, lengths)
        mx.eval(expanded)

        expected_sum = sum(a * l for a, l in zip(adv.tolist(), lengths))
        actual_sum = mx.sum(expanded).item()
        assert abs(actual_sum - expected_sum) < 1e-4, (
            f"Sum mismatch: got {actual_sum}, want {expected_sum}"
        )


# ===========================================================================
# 7. Trainer._loss_fn (structural)
# ===========================================================================

@pytest.mark.unit
class TestRegressionTrainerLossFn:
    """Structural invariants for Trainer._loss_fn (random weights → no frozen values)."""

    @staticmethod
    def _make_trainer_and_batch():
        model = _TinyLM()
        opt = optim.Adam(learning_rate=1e-3)
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=grpo.policy_loss,
            optimizer=opt,
            compile_training=False,
        )
        # Minimal batch: 2 episodes, 3 tokens each
        batch = {
            'obs': mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32),
            'act': mx.array([7, 8, 9, 10, 11, 12], dtype=mx.int32),
            'logprob': mx.array([-0.5, -0.6, -0.7, -0.3, -0.4, -0.8], dtype=mx.float32),
            'rewards': mx.array([1.0, 0.0], dtype=mx.float32),
            'episode_lengths': [3, 3],
        }
        return trainer, batch

    def test_loss_is_finite_scalar(self):
        """Loss from _loss_fn is a finite scalar."""
        trainer, batch = self._make_trainer_and_batch()
        loss = trainer._loss_fn(batch)
        mx.eval(loss)
        assert loss.ndim == 0, f"Expected scalar, got {loss.ndim}D"
        assert not mx.isnan(loss).item(), "Loss is NaN"
        assert not mx.isinf(loss).item(), "Loss is Inf"

    def test_gradients_are_nonzero_and_finite(self):
        """Gradients through _loss_fn are nonzero and finite."""
        trainer, batch = self._make_trainer_and_batch()
        loss_and_grad_fn = nn.value_and_grad(trainer.model, trainer._loss_fn)
        loss, grads = loss_and_grad_fn(batch)
        mx.eval(loss, grads)
        assert _has_nonzero_grads(grads), "All gradients are zero"
        assert _all_grads_finite(grads), "Non-finite gradients found"

    def test_train_returns_metrics_dict(self):
        """Trainer.train() returns dict with 'loss' and 'step' keys."""
        trainer, batch = self._make_trainer_and_batch()
        metrics = trainer.train(rollout_data=batch)
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        assert 'loss' in metrics, "Missing 'loss' key"
        assert 'step' in metrics, "Missing 'step' key"
        assert isinstance(metrics['loss'], float), f"Loss not float: {type(metrics['loss'])}"
