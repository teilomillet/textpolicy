"""
Smoke tests for experiments/countdown_reasoning_lora.py.
"""

import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


class TestImports:
    def test_import_config(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig

        assert ReasoningConfig is not None

    def test_import_runner(self):
        from experiments.countdown_reasoning_lora import run_experiment

        assert callable(run_experiment)


class TestReasoningConfig:
    def test_defaults(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig

        cfg = ReasoningConfig()
        assert cfg.model_id == "arcee-ai/Trinity-Nano-Preview"
        assert cfg.lora_rank == 2
        assert cfg.lora_layers == 4
        assert cfg.entropy_weight == 0.1
        assert cfg.hicra_alpha == 0.2
        assert cfg.episodes_per_step == 8
        assert cfg.batch_size == 8
        assert cfg.output_dir == "results/countdown_reasoning_lora"

    def test_roundtrip_dict(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig

        cfg = ReasoningConfig(max_steps=12, entropy_weight=0.2, hicra_alpha=0.3)
        restored = ReasoningConfig(**asdict(cfg))
        assert asdict(restored) == asdict(cfg)

    def test_allows_episodes_per_step_above_legacy_runner_limit(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig, run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("experiments.countdown_reasoning_lora.load_model") as mock_load_model,
                patch(
                    "experiments.countdown_reasoning_lora.create_tinylora_reasoning_setup"
                ) as mock_create_setup,
                patch("experiments.countdown_reasoning_lora.create_policy"),
                patch(
                    "experiments.countdown_reasoning_lora.generate_countdown_problems"
                ) as mock_generate_problems,
                patch(
                    "experiments.countdown_reasoning_lora.RolloutCoordinator"
                ) as mock_rollout_cls,
                patch(
                    "experiments.countdown_reasoning_lora.EmergenceLogger"
                ) as mock_emergence_cls,
                patch("experiments.countdown_reasoning_lora.save_checkpoint"),
            ):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load_model.return_value = (mock_model, mock_tokenizer)

                mock_trainer = MagicMock()
                mock_trainer.model = mock_model
                mock_create_setup.return_value = (
                    mock_trainer,
                    {"memory_savings_percent": 95.0},
                )

                mock_generate_problems.return_value = [{"target": 15, "numbers": [10, 5, 3]}]
                mock_rollout_cls.return_value = MagicMock()
                mock_emergence_cls.return_value = MagicMock()

                cfg = ReasoningConfig(
                    max_steps=0,
                    output_dir=tmpdir,
                    num_problems=1,
                    episodes_per_step=11,
                    batch_size=8,
                )
                run_experiment(cfg)

                rollout_kwargs = mock_rollout_cls.call_args.kwargs
                assert rollout_kwargs["max_steps"] == 11
                assert rollout_kwargs["max_episodes"] == 11
                assert rollout_kwargs["batch_size"] == 8

                # No training loop work should run when max_steps=0.
                mock_trainer.train.assert_not_called()
                assert (Path(tmpdir) / "config.json").exists()

    def test_rejects_batch_size_exceeding_episodes_per_step(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig, run_experiment

        cfg = ReasoningConfig(episodes_per_step=4, batch_size=5)
        with pytest.raises(ValueError, match="batch_size"):
            run_experiment(cfg)


class TestRunExperiment:
    def test_smoke_uses_batched_rollout_config(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig, run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("experiments.countdown_reasoning_lora.load_model") as mock_load_model,
                patch(
                    "experiments.countdown_reasoning_lora.create_tinylora_reasoning_setup"
                ) as mock_create_setup,
                patch("experiments.countdown_reasoning_lora.create_policy") as mock_create_policy,
                patch(
                    "experiments.countdown_reasoning_lora.generate_countdown_problems"
                ) as mock_generate_problems,
                patch(
                    "experiments.countdown_reasoning_lora.RolloutCoordinator"
                ) as mock_rollout_cls,
                patch(
                    "experiments.countdown_reasoning_lora.EmergenceLogger"
                ) as mock_emergence_cls,
                patch("experiments.countdown_reasoning_lora.save_checkpoint"),
            ):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load_model.return_value = (mock_model, mock_tokenizer)

                mock_trainer = MagicMock()
                mock_trainer.model = mock_model
                mock_trainer.train.return_value = {"loss": 0.25}
                mock_create_setup.return_value = (
                    mock_trainer,
                    {"memory_savings_percent": 95.0},
                )

                mock_create_policy.return_value = MagicMock()
                mock_generate_problems.return_value = [{"target": 15, "numbers": [10, 5, 3]}]

                mock_episode = MagicMock()
                mock_episode.to_dict.return_value = {
                    "obs": [mx.array([1, 2])],
                    "act": [mx.array([3, 4])],
                    "rew": [0.5],
                    "next_obs": [mx.array([5, 6])],
                    "done": [True],
                    "timeout": [False],
                    "logprob": [mx.array([-0.2, -0.1])],
                }
                mock_rollout = MagicMock()
                mock_rollout.collect.return_value = MagicMock(episodes=[mock_episode])
                mock_rollout_cls.return_value = mock_rollout

                mock_emergence = MagicMock()
                mock_emergence.log_step.return_value = {
                    "mean_reward": 0.5,
                    "correct_count": 1,
                    "total_count": 1,
                    "planning_token_ratio": 0.0,
                }
                mock_emergence_cls.return_value = mock_emergence

                cfg = ReasoningConfig(
                    max_steps=1,
                    output_dir=tmpdir,
                    num_problems=1,
                    episodes_per_step=4,
                    batch_size=4,
                )
                run_experiment(cfg)

                assert (Path(tmpdir) / "config.json").exists()
                mock_trainer.train.assert_called_once()
                mock_rollout.collect.assert_called_once()
                mock_rollout.close.assert_called_once()

                rollout_kwargs = mock_rollout_cls.call_args.kwargs
                assert rollout_kwargs["batch_size"] == 4
                assert rollout_kwargs["model"] is mock_model
                assert rollout_kwargs["tokenizer"] is mock_tokenizer
                assert rollout_kwargs["generation_params"]["max_tokens"] == cfg.max_completion_tokens
                assert rollout_kwargs["generation_params"]["temperature"] == cfg.temperature
                assert rollout_kwargs["generation_params"]["top_p"] == cfg.top_p
                assert (
                    rollout_kwargs["generation_params"]["repetition_penalty"]
                    == cfg.repetition_penalty
                )
