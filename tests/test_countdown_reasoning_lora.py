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
        assert cfg.alpha_1 == 1.0
        assert cfg.alpha_2 == 0.1
        assert cfg.hicra_gamma == 0.3
        assert cfg.episodes_per_step == 8
        assert cfg.batch_size == 8
        assert cfg.output_dir == "results/countdown_reasoning_lora"

    def test_roundtrip_dict(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig

        cfg = ReasoningConfig(max_steps=12, alpha_1=0.9, alpha_2=0.2)
        restored = ReasoningConfig(**asdict(cfg))
        assert asdict(restored) == asdict(cfg)

    def test_allows_episodes_per_step_above_legacy_runner_limit(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig, run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("experiments.countdown_reasoning_lora.load_model") as mock_load_model,
                patch(
                    "experiments.countdown_reasoning_lora.create_lora_setup"
                ) as mock_lora_setup,
                patch(
                    "experiments.countdown_reasoning_lora.Trainer"
                ) as mock_trainer_cls,
                patch("experiments.countdown_reasoning_lora.build_gtpo_transform"),
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

                mock_lora_setup.return_value = (
                    mock_model,
                    {"memory_savings_percent": 95.0},
                )

                mock_trainer = MagicMock()
                mock_trainer.model = mock_model
                mock_trainer_cls.return_value = mock_trainer

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
    def test_log_wandb_step_uses_explicit_step_kwarg(self):
        import experiments.countdown_reasoning_lora as exp

        mock_wandb = MagicMock()
        with patch.object(exp, "wandb", mock_wandb, create=True):
            cfg = exp.ReasoningConfig()
            exp.log_wandb_step(
                step=7,
                step_stats={
                    "entropy_mean": 0.2,
                    "entropy_std": 0.05,
                    "mean_reward": 0.4,
                    "std_reward": 0.1,
                    "planning_token_ratio": 0.3,
                    "total_count": 2,
                    "correct_count": 1,
                    "mean_completion_length": 12.0,
                },
                train_metrics={"loss": 0.25},
                episode_stats={},
                config=cfg,
                use_wandb=True,
            )

        assert mock_wandb.log.call_count == 1
        args, kwargs = mock_wandb.log.call_args
        assert kwargs["step"] == 7
        assert args[0]["step"] == 7

    def test_log_wandb_completions_uses_explicit_step_kwarg(self):
        import experiments.countdown_reasoning_lora as exp

        mock_wandb = MagicMock()
        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda tokens: " ".join(str(t) for t in tokens)
        episodes = [
            {
                "obs": [[101, 102]],
                "act": [[201, 202, 203]],
                "rew": [1.0],
            }
        ]

        with patch.object(exp, "wandb", mock_wandb, create=True):
            exp.log_wandb_completions(
                step=10,
                episodes=episodes,
                tokenizer=tokenizer,
                use_wandb=True,
            )

        assert mock_wandb.log.call_count == 1
        args, kwargs = mock_wandb.log.call_args
        assert kwargs["step"] == 10
        assert "completions/samples" in args[0]

    def test_log_wandb_completions_skips_non_interval_steps(self):
        import experiments.countdown_reasoning_lora as exp

        mock_wandb = MagicMock()
        tokenizer = MagicMock()

        with patch.object(exp, "wandb", mock_wandb, create=True):
            exp.log_wandb_completions(
                step=9,
                episodes=[],
                tokenizer=tokenizer,
                use_wandb=True,
            )

        mock_wandb.log.assert_not_called()

    def test_wandb_project_without_wandb_does_not_force_metrics_interval_1(self):
        import experiments.countdown_reasoning_lora as exp

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(exp, "HAS_WANDB", False),
                patch.object(exp, "load_model") as mock_load_model,
                patch.object(exp, "create_lora_setup") as mock_lora_setup,
                patch.object(exp, "Trainer") as mock_trainer_cls,
                patch.object(exp, "build_gtpo_transform"),
                patch.object(exp, "create_policy"),
                patch.object(exp, "generate_countdown_problems") as mock_generate_problems,
                patch.object(exp, "RolloutCoordinator") as mock_rollout_cls,
                patch.object(exp, "EmergenceLogger") as mock_emergence_cls,
                patch.object(exp, "save_checkpoint"),
            ):
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load_model.return_value = (mock_model, mock_tokenizer)

                mock_lora_setup.return_value = (
                    mock_model,
                    {"memory_savings_percent": 95.0},
                )

                mock_trainer = MagicMock()
                mock_trainer.model = mock_model
                mock_trainer_cls.return_value = mock_trainer

                mock_generate_problems.return_value = [{"target": 15, "numbers": [10, 5, 3]}]
                mock_rollout_cls.return_value = MagicMock()
                mock_emergence_cls.return_value = MagicMock()

                cfg = exp.ReasoningConfig(
                    max_steps=0,
                    output_dir=tmpdir,
                    num_problems=1,
                    episodes_per_step=4,
                    batch_size=4,
                    wandb_project="example-project",
                )
                exp.run_experiment(cfg)

                trainer_kwargs = mock_trainer_cls.call_args.kwargs
                assert "metrics_fn" not in trainer_kwargs
                assert "metrics_interval" not in trainer_kwargs

    def test_smoke_uses_batched_rollout_config(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig, run_experiment

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("experiments.countdown_reasoning_lora.load_model") as mock_load_model,
                patch(
                    "experiments.countdown_reasoning_lora.create_lora_setup"
                ) as mock_lora_setup,
                patch(
                    "experiments.countdown_reasoning_lora.Trainer"
                ) as mock_trainer_cls,
                patch("experiments.countdown_reasoning_lora.build_gtpo_transform"),
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

                mock_lora_setup.return_value = (
                    mock_model,
                    {"memory_savings_percent": 95.0},
                )

                mock_trainer = MagicMock()
                mock_trainer.model = mock_model
                mock_trainer.train.return_value = {"loss": 0.25}
                mock_trainer_cls.return_value = mock_trainer

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
