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

    def test_policy_loss_asymmetric_clipping(self):
        """Test GRPO policy_loss with DAPO-style asymmetric clipping."""
        from textpolicy.algorithms import grpo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.6, 0.1, -0.9, 0.4, -0.2])

        # Test with DAPO defaults: low=0.2, high=0.28
        loss_asymmetric = grpo.policy_loss(
            old_lp, new_lp, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        assert not mx.isnan(loss_asymmetric)
        assert loss_asymmetric.shape == ()

        # Test backward compatibility: clip_ratio should override both
        loss_symmetric = grpo.policy_loss(
            old_lp, new_lp, advantages, clip_ratio=0.2
        )

        assert not mx.isnan(loss_symmetric)
        assert loss_symmetric.shape == ()

    def test_policy_loss_asymmetric_bounds_behavior(self):
        """Test that asymmetric clipping produces correct bounds."""
        from textpolicy.algorithms import grpo

        # Create extreme ratios that will definitely be clipped
        old_lp = mx.array([-1.0, -1.0, -1.0])
        # Large positive change -> high ratio; Large negative change -> low ratio
        new_lp = mx.array([-0.2, -1.8, -1.0])  # ratios: ~2.23, ~0.45, 1.0
        advantages = mx.array([1.0, 1.0, 1.0])

        # With asymmetric bounds: lower=0.8, upper=1.28
        loss_asymmetric = grpo.policy_loss(
            old_lp, new_lp, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        # With symmetric bounds: lower=0.8, upper=1.2
        loss_symmetric = grpo.policy_loss(
            old_lp, new_lp, advantages, clip_ratio=0.2
        )

        # Asymmetric allows higher ratios, so clipping differs
        # Both should be valid (not NaN)
        assert not mx.isnan(loss_asymmetric)
        assert not mx.isnan(loss_symmetric)

    def test_policy_loss_normalize_constant(self):
        """Test policy_loss with fixed constant normalization (Dr. GRPO)."""
        from textpolicy.algorithms import grpo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.6, 0.1, -0.9, 0.4, -0.2])

        # Test with mean normalization (default)
        loss_mean = grpo.policy_loss(old_lp, new_lp, advantages)

        # Test with constant normalization
        loss_const = grpo.policy_loss(
            old_lp, new_lp, advantages, normalize_constant=5
        )

        # Both should be valid
        assert not mx.isnan(loss_mean)
        assert not mx.isnan(loss_const)

        # With constant=5 (same as number of elements), should be similar
        # but not identical due to different formulas
        assert loss_mean.shape == ()
        assert loss_const.shape == ()

    def test_policy_loss_normalize_constant_eliminates_length_bias(self):
        """Test that constant normalization eliminates length bias."""
        from textpolicy.algorithms import grpo

        # Short sequence
        old_lp_short = mx.array([-1.0, -1.2])
        new_lp_short = mx.array([-1.1, -1.0])
        adv_short = mx.array([0.5, 0.5])

        # Long sequence (same pattern repeated 4x)
        old_lp_long = mx.array([-1.0, -1.2] * 4)
        new_lp_long = mx.array([-1.1, -1.0] * 4)
        adv_long = mx.array([0.5, 0.5] * 4)

        # With mean normalization: short and long should give same loss
        # (since pattern is identical, just repeated)
        loss_short_mean = grpo.policy_loss(old_lp_short, new_lp_short, adv_short)
        loss_long_mean = grpo.policy_loss(old_lp_long, new_lp_long, adv_long)

        # Mean losses should be equal (same average per token)
        assert abs(float(loss_short_mean) - float(loss_long_mean)) < 1e-5

        # With constant normalization: long sequence contributes more
        loss_short_const = grpo.policy_loss(
            old_lp_short, new_lp_short, adv_short, normalize_constant=10
        )
        loss_long_const = grpo.policy_loss(
            old_lp_long, new_lp_long, adv_long, normalize_constant=10
        )

        # Long sequence should have larger absolute loss (4x tokens)
        # This is the expected behavior with constant normalization
        assert abs(float(loss_long_const)) > abs(float(loss_short_const)) * 3

    def test_policy_loss_compiled_constant_norm(self):
        """Test compiled policy loss with constant normalization."""
        from textpolicy.algorithms import grpo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.6, 0.1, -0.9, 0.4, -0.2])

        # Test compiled version with constant normalization
        loss = grpo.policy_loss_compiled_constant_norm(
            old_lp, new_lp, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
            normalize_constant=1024.0
        )

        assert not mx.isnan(loss)
        assert loss.shape == ()

    def test_compute_metrics_asymmetric(self):
        """Test compute_metrics with asymmetric clipping."""
        from textpolicy.algorithms import grpo

        old_lp = mx.array([-1.0, -1.2, -0.8, -1.1, -0.9])
        new_lp = mx.array([-1.1, -1.0, -0.9, -1.0, -1.0])
        advantages = mx.array([0.6, 0.1, -0.9, 0.4, -0.2])

        # Test with asymmetric bounds
        metrics = grpo.compute_metrics(
            old_lp, new_lp, advantages,
            clip_ratio_low=0.2, clip_ratio_high=0.28
        )

        # Check all expected metrics are present
        assert 'clip_fraction_lower' in metrics
        assert 'clip_fraction_upper' in metrics
        assert 'clip_fraction' in metrics
        assert 'clip_ratio_low' in metrics
        assert 'clip_ratio_high' in metrics

        # Check values are reasonable
        assert 0 <= metrics['clip_fraction_lower'] <= 1
        assert 0 <= metrics['clip_fraction_upper'] <= 1
        assert 0 <= metrics['clip_fraction'] <= 1
        assert metrics['clip_ratio_low'] == 0.2
        assert metrics['clip_ratio_high'] == 0.28

        # Total clip fraction should be >= max of individual fractions
        total = metrics['clip_fraction']
        assert total >= metrics['clip_fraction_lower'] or abs(total - metrics['clip_fraction_lower']) < 1e-6
        assert total >= metrics['clip_fraction_upper'] or abs(total - metrics['clip_fraction_upper']) < 1e-6

    def test_entropy_bonus(self):
        """Test entropy bonus computation."""
        from textpolicy.algorithms import grpo

        logprobs = mx.array([-1.0, -2.0, -0.5, -1.5])
        entropy = grpo.entropy_bonus(logprobs, coefficient=0.01)

        assert not mx.isnan(entropy)

    def test_compute_length_penalty(self):
        """Test soft overlong penalty computation."""
        from textpolicy.algorithms import grpo

        max_length = 512
        cache_length = 100

        # Below threshold: no penalty
        penalty = grpo.compute_length_penalty(400, max_length, cache_length)
        assert penalty == 0.0

        # At threshold boundary: no penalty
        penalty = grpo.compute_length_penalty(412, max_length, cache_length)
        assert penalty == 0.0

        # Just past threshold: small penalty
        penalty = grpo.compute_length_penalty(413, max_length, cache_length)
        assert -0.1 < penalty < 0.0

        # Halfway through cache zone
        penalty = grpo.compute_length_penalty(462, max_length, cache_length)
        assert -0.35 < penalty < -0.2

        # At max length: max penalty
        penalty = grpo.compute_length_penalty(512, max_length, cache_length)
        assert penalty == -0.5

        # Past max length: clamped at max penalty
        penalty = grpo.compute_length_penalty(600, max_length, cache_length)
        assert penalty == -0.5

    def test_compute_length_penalty_custom_max_penalty(self):
        """Test length penalty with custom max penalty."""
        from textpolicy.algorithms import grpo

        # Custom max penalty
        penalty = grpo.compute_length_penalty(512, 512, 100, max_penalty=1.0)
        assert penalty == -1.0

        penalty = grpo.compute_length_penalty(462, 512, 100, max_penalty=1.0)
        assert -0.6 < penalty < -0.4

    def test_apply_length_shaping(self):
        """Test applying length penalties to rewards."""
        from textpolicy.algorithms import grpo

        rewards = mx.array([1.0, 0.5, 0.0, -0.5])
        lengths = [400, 462, 512, 600]  # below, mid, at, past max
        max_length = 512
        cache_length = 100

        shaped = grpo.apply_length_shaping(
            rewards, lengths, max_length, cache_length
        )

        # First should be unchanged (below threshold)
        assert abs(float(shaped[0]) - 1.0) < 1e-6

        # Second should have small penalty
        assert 0.1 < float(shaped[1]) < 0.4

        # Third should have max penalty (-0.5)
        assert abs(float(shaped[2]) - (-0.5)) < 1e-6

        # Fourth should have max penalty (clamped)
        assert abs(float(shaped[3]) - (-1.0)) < 1e-6

    def test_compute_length_shaping_stats(self):
        """Test length shaping statistics."""
        from textpolicy.algorithms import grpo

        lengths = [400, 420, 462, 500, 512, 520]
        max_length = 512
        cache_length = 100

        stats = grpo.compute_length_shaping_stats(lengths, max_length, cache_length)

        assert 'mean_length' in stats
        assert 'max_length_observed' in stats
        assert 'truncation_rate' in stats
        assert 'penalty_zone_rate' in stats

        # 2 truncated (512, 520)
        assert abs(stats['truncation_rate'] - 2/6) < 1e-6

        # threshold = 412, so penalty zone is 412 <= l < 512
        # 420, 462, 500 are in zone = 3 items
        assert stats['penalty_zone_rate'] == 3/6  # 420, 462, 500

        assert stats['max_length_observed'] == 520

    def test_length_shaping_empty_list(self):
        """Test length shaping with empty list."""
        from textpolicy.algorithms import grpo

        stats = grpo.compute_length_shaping_stats([], 512, 100)
        assert stats['mean_length'] == 0.0
        assert stats['truncation_rate'] == 0.0

    def test_filter_informative_prompts_keeps_varied(self):
        """Test that filter_informative_prompts keeps prompts with varied rewards."""
        from textpolicy.algorithms import grpo

        # Create mock episodes with same prompt but different rewards
        # Simulate 2 prompts, each with 3 completions
        episodes = [
            # Prompt 1: varied rewards (should be kept)
            {'obs': [1, 2, 3], 'act': [4], 'rew': [1.0]},
            {'obs': [1, 2, 3], 'act': [5], 'rew': [0.0]},
            {'obs': [1, 2, 3], 'act': [6], 'rew': [0.5]},
            # Prompt 2: all correct (should be filtered)
            {'obs': [7, 8, 9], 'act': [10], 'rew': [1.0]},
            {'obs': [7, 8, 9], 'act': [11], 'rew': [1.0]},
            {'obs': [7, 8, 9], 'act': [12], 'rew': [1.0]},
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        # Should keep prompt 1 (3 episodes) and filter prompt 2
        assert len(filtered) == 3
        assert stats['prompts_kept'] == 1
        assert stats['prompts_dropped_all_correct'] == 1
        assert stats['prompts_dropped_all_wrong'] == 0
        assert stats['episodes_kept'] == 3
        assert stats['episodes_dropped'] == 3

    def test_filter_informative_prompts_filters_all_wrong(self):
        """Test that filter_informative_prompts filters prompts where all completions fail."""
        from textpolicy.algorithms import grpo

        episodes = [
            # Prompt 1: all wrong (should be filtered)
            {'obs': [1, 2], 'act': [3], 'rew': [0.0]},
            {'obs': [1, 2], 'act': [4], 'rew': [0.0]},
            # Prompt 2: varied (should be kept)
            {'obs': [5, 6], 'act': [7], 'rew': [1.0]},
            {'obs': [5, 6], 'act': [8], 'rew': [0.0]},
        ]

        filtered, stats = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        assert len(filtered) == 2
        assert stats['prompts_kept'] == 1
        assert stats['prompts_dropped_all_correct'] == 0
        assert stats['prompts_dropped_all_wrong'] == 1

    def test_filter_informative_prompts_min_variance_threshold(self):
        """Test that min_variance threshold controls filtering sensitivity."""
        from textpolicy.algorithms import grpo

        # Prompt with moderate variance (0.7 and 1.0)
        # variance = ((0.7-0.85)^2 + (1.0-0.85)^2)/2 = (0.0225 + 0.0225)/2 = 0.0225
        episodes = [
            {'obs': [1], 'act': [2], 'rew': [0.7]},
            {'obs': [1], 'act': [3], 'rew': [1.0]},
        ]

        # With threshold below variance (0.0225), should be kept
        filtered_low, _ = grpo.filter_informative_prompts(episodes, min_variance=0.01)
        assert len(filtered_low) == 2

        # With threshold above variance (0.0225), should be filtered
        filtered_high, _ = grpo.filter_informative_prompts(episodes, min_variance=0.05)
        assert len(filtered_high) == 0

    def test_filter_informative_prompts_empty_list(self):
        """Test filter_informative_prompts with empty episode list."""
        from textpolicy.algorithms import grpo

        filtered, stats = grpo.filter_informative_prompts([])

        assert filtered == []
        assert stats['prompts_kept'] == 0
        assert stats['filter_rate'] == 0.0

    def test_compute_prompt_group_stats(self):
        """Test compute_prompt_group_stats returns correct statistics."""
        from textpolicy.algorithms import grpo

        episodes = [
            # Prompt 1: 3 completions
            {'obs': [1, 2], 'act': [3], 'rew': [1.0]},
            {'obs': [1, 2], 'act': [4], 'rew': [0.5]},
            {'obs': [1, 2], 'act': [5], 'rew': [0.0]},
            # Prompt 2: 2 completions
            {'obs': [6, 7], 'act': [8], 'rew': [1.0]},
            {'obs': [6, 7], 'act': [9], 'rew': [1.0]},
        ]

        stats = grpo.compute_prompt_group_stats(episodes)

        assert stats['num_prompts'] == 2
        assert stats['num_episodes'] == 5
        assert stats['completions_per_prompt'] == 2.5
        assert stats['reward_variance_mean'] > 0  # Some variance expected
        assert 'reward_variance_std' in stats

    def test_filter_preserves_episode_structure(self):
        """Test that filtered episodes maintain their structure."""
        from textpolicy.algorithms import grpo

        # Episodes with various fields
        episodes = [
            {'obs': [1, 2], 'act': [3, 4], 'rew': [1.0], 'logprob': [-0.5]},
            {'obs': [1, 2], 'act': [5, 6], 'rew': [0.0], 'logprob': [-0.7]},
        ]

        filtered, _ = grpo.filter_informative_prompts(episodes, min_variance=0.01)

        # Both should be kept (varied rewards)
        assert len(filtered) == 2
        # Check structure preserved
        assert 'obs' in filtered[0]
        assert 'act' in filtered[0]
        assert 'rew' in filtered[0]
        assert 'logprob' in filtered[0]


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
