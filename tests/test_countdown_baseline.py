"""
Tests for experiments/countdown_baseline.py

All tests use mocks — no GPU or model download required.
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Smoke test: import succeeds
# ---------------------------------------------------------------------------
class TestImports:
    def test_import_baseline_config(self):
        from experiments.countdown_baseline import BaselineConfig

        assert BaselineConfig is not None

    def test_import_run_baseline(self):
        from experiments.countdown_baseline import run_baseline

        assert callable(run_baseline)

    def test_import_helpers(self):
        from experiments.countdown_baseline import (
            print_summary,
            save_checkpoint,
            save_config,
        )

        assert callable(save_config)
        assert callable(save_checkpoint)
        assert callable(print_summary)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------
class TestBaselineConfig:
    def test_default_values(self):
        from experiments.countdown_baseline import BaselineConfig

        config = BaselineConfig()
        assert config.model_id == "Qwen/Qwen3-0.6B"
        assert config.lora_rank == 8
        assert config.lora_layers == 8
        assert config.learning_rate == 5e-6
        assert config.max_steps == 500
        assert config.episodes_per_step == 8
        assert config.num_problems == 50
        assert config.temperature == 0.7
        assert config.clip_ratio_low == 0.2
        assert config.clip_ratio_high == 0.28

    def test_custom_values(self):
        from experiments.countdown_baseline import BaselineConfig

        config = BaselineConfig(model_id="test/model", max_steps=10, learning_rate=1e-4)
        assert config.model_id == "test/model"
        assert config.max_steps == 10
        assert config.learning_rate == 1e-4

    def test_config_serialization_roundtrip(self):
        from experiments.countdown_baseline import BaselineConfig

        config = BaselineConfig(max_steps=42, learning_rate=1e-3)
        d = asdict(config)
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        assert restored["max_steps"] == 42
        assert restored["learning_rate"] == 1e-3
        config2 = BaselineConfig(**restored)
        assert asdict(config2) == asdict(config)

    def test_episodes_per_step_is_multiple_of_group_size(self):
        from experiments.countdown_baseline import BaselineConfig

        config = BaselineConfig()
        assert config.episodes_per_step % config.num_generations_per_prompt == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------
class TestSaveConfig:
    def test_saves_json(self):
        from experiments.countdown_baseline import BaselineConfig, save_config

        config = BaselineConfig(max_steps=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_config(config, Path(tmpdir))
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()
            loaded = json.loads(config_path.read_text())
            assert loaded["max_steps"] == 7
            assert loaded["model_id"] == "Qwen/Qwen3-0.6B"


class TestPrintSummary:
    def test_no_file(self, capsys):
        from experiments.countdown_baseline import print_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            print_summary(Path(tmpdir))
            out = capsys.readouterr().out
            assert "nothing to summarize" in out.lower()

    def test_empty_file(self, capsys):
        from experiments.countdown_baseline import print_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            emergence_dir = Path(tmpdir) / "emergence"
            emergence_dir.mkdir()
            (emergence_dir / "steps.jsonl").write_text("")
            print_summary(Path(tmpdir))
            out = capsys.readouterr().out
            assert "no step records" in out.lower()

    def test_prints_summary(self, capsys):
        from experiments.countdown_baseline import print_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            emergence_dir = Path(tmpdir) / "emergence"
            emergence_dir.mkdir()
            records = [
                {
                    "step": 0,
                    "mean_reward": 0.1,
                    "correct_count": 1,
                    "total_count": 8,
                    "planning_token_ratio": 0.0,
                },
                {
                    "step": 9,
                    "mean_reward": 0.5,
                    "correct_count": 4,
                    "total_count": 8,
                    "planning_token_ratio": 0.02,
                },
            ]
            lines = [json.dumps(r) for r in records]
            (emergence_dir / "steps.jsonl").write_text("\n".join(lines) + "\n")

            print_summary(Path(tmpdir))
            out = capsys.readouterr().out
            assert "EXPERIMENT SUMMARY" in out
            assert "0.100" in out  # first step reward
            assert "0.500" in out  # final step reward
            assert "50.0%" in out  # best accuracy = 4/8


# ---------------------------------------------------------------------------
# Environment integration (mocked model, real env)
# ---------------------------------------------------------------------------
class TestEnvironmentIntegration:
    def test_create_env_with_countdown_prompts(self):
        from textpolicy.environment.text_generation import TextGenerationEnv
        from textpolicy.tasks.countdown import (
            countdown_reward,
            format_countdown_prompt,
            generate_countdown_problems,
        )

        problems = generate_countdown_problems(5, seed=42)
        prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]

        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=countdown_reward,
            max_tokens=64,
            tokenizer=mock_tokenizer,
            examples=problems,
        )
        assert env is not None
        assert len(prompts) == 5

    def test_prompts_cycle_through(self):
        from textpolicy.tasks.countdown import (
            format_countdown_prompt,
            generate_countdown_problems,
        )

        problems = generate_countdown_problems(3, seed=99)
        prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

        # Verify prompts contain target and numbers
        for prompt, prob in zip(prompts, problems):
            assert str(prob["target"]) in prompt
            for n in prob["numbers"]:
                assert str(n) in prompt


# ---------------------------------------------------------------------------
# Reward function integration
# ---------------------------------------------------------------------------
class TestRewardIntegration:
    def test_correct_expression(self):
        from textpolicy.tasks.countdown import countdown_reward

        reward = countdown_reward(
            prompt="ignored",
            completion="10 + 5",
            example={"target": 15, "numbers": [10, 5, 3]},
        )
        assert reward == 1.0

    def test_wrong_answer(self):
        from textpolicy.tasks.countdown import countdown_reward

        reward = countdown_reward(
            prompt="ignored",
            completion="10 + 5",
            example={"target": 20, "numbers": [10, 5, 3]},
        )
        assert reward == 0.0

    def test_syntax_error(self):
        from textpolicy.tasks.countdown import countdown_reward

        reward = countdown_reward(
            prompt="ignored",
            completion="not a valid expression!!!",
            example={"target": 15, "numbers": [10, 5, 3]},
        )
        assert reward == -0.5

    def test_empty_completion(self):
        from textpolicy.tasks.countdown import countdown_reward

        reward = countdown_reward(
            prompt="ignored",
            completion="",
            example={"target": 15, "numbers": [10, 5, 3]},
        )
        assert reward == -0.5


# ---------------------------------------------------------------------------
# EmergenceLogger integration (no GPU)
# ---------------------------------------------------------------------------
class TestEmergenceLoggerIntegration:
    def test_logger_produces_output_files(self):
        from textpolicy.analysis import EmergenceLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EmergenceLogger(output_dir=tmpdir)

            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "hello world"

            # Create a minimal episode-like dict
            episode = MagicMock()
            episode.obs = [mx.array([1, 2, 3])]
            episode.act = [mx.array([4, 5, 6])]
            episode.rew = [1.0]
            episode.logprob = [mx.array([-0.5, -0.3, -0.1])]

            step_stats = logger.log_step(
                step=0,
                episodes=[episode],
                tokenizer=mock_tokenizer,
                examples=[{"target": 15, "numbers": [10, 5]}],
            )

            logger.finish()

            assert (Path(tmpdir) / "generations.jsonl").exists()
            assert (Path(tmpdir) / "steps.jsonl").exists()
            assert "mean_reward" in step_stats
            assert "correct_count" in step_stats
            assert "planning_token_ratio" in step_stats

    def test_logger_step_stats_values(self):
        from textpolicy.analysis import EmergenceLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EmergenceLogger(output_dir=tmpdir)

            mock_tokenizer = MagicMock()
            mock_tokenizer.decode.return_value = "10 + 5"

            ep1 = MagicMock()
            ep1.obs = [mx.array([1, 2])]
            ep1.act = [mx.array([3, 4])]
            ep1.rew = [1.0]
            ep1.logprob = [mx.array([-0.2, -0.3])]

            ep2 = MagicMock()
            ep2.obs = [mx.array([5, 6])]
            ep2.act = [mx.array([7, 8])]
            ep2.rew = [0.0]
            ep2.logprob = [mx.array([-0.4, -0.5])]

            stats = logger.log_step(
                step=0,
                episodes=[ep1, ep2],
                tokenizer=mock_tokenizer,
                examples=[
                    {"target": 15, "numbers": [10, 5]},
                    {"target": 20, "numbers": [10, 5]},
                ],
            )
            logger.finish()

            assert stats["total_count"] == 2
            assert stats["mean_reward"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Full run_baseline integration (fully mocked)
# ---------------------------------------------------------------------------
class TestRunBaseline:
    @patch("experiments.countdown_baseline.RolloutCoordinator")
    @patch("experiments.countdown_baseline.Trainer")
    @patch("experiments.countdown_baseline.create_policy")
    @patch("experiments.countdown_baseline.create_lora_setup")
    @patch("experiments.countdown_baseline.load_model")
    def test_run_baseline_smoke(
        self,
        mock_load_model,
        mock_create_lora,
        mock_create_policy,
        mock_trainer_cls,
        mock_rollout_cls,
    ):
        from experiments.countdown_baseline import BaselineConfig, run_baseline

        # Mock model + tokenizer
        mock_model = MagicMock()
        mock_model.named_parameters.return_value = []
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "10 + 5"
        mock_load_model.return_value = (mock_model, mock_tokenizer)

        # Mock LoRA setup
        mock_create_lora.return_value = (
            mock_model,
            {"memory_savings_percent": 95.0},
        )

        # Mock policy
        mock_create_policy.return_value = MagicMock()

        # Mock rollout coordinator
        mock_rollout = MagicMock()
        mock_ep = MagicMock()
        mock_ep.obs = [mx.array([1, 2, 3])]
        mock_ep.act = [mx.array([4, 5, 6])]
        mock_ep.rew = [0.5]
        mock_ep.logprob = [mx.array([-0.3, -0.2, -0.1])]
        mock_ep.to_dict.return_value = {
            "obs": [mx.array([1, 2, 3])],
            "act": [mx.array([4, 5, 6])],
            "rew": [0.5],
            "next_obs": [mx.array([7, 8, 9])],
            "done": [True],
            "timeout": [False],
            "logprob": [mx.array([-0.3, -0.2, -0.1])],
        }
        mock_buffer = MagicMock()
        mock_buffer.episodes = [mock_ep]
        mock_rollout.collect.return_value = mock_buffer
        mock_rollout_cls.return_value = mock_rollout

        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"loss": 0.5, "step": 0}
        mock_trainer_cls.return_value = mock_trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            config = BaselineConfig(
                max_steps=2,
                output_dir=tmpdir,
                num_problems=3,
            )
            run_baseline(config)

            # Verify config was saved
            assert (Path(tmpdir) / "config.json").exists()

            # Verify emergence logs exist
            assert (Path(tmpdir) / "emergence" / "generations.jsonl").exists()
            assert (Path(tmpdir) / "emergence" / "steps.jsonl").exists()

            # Verify checkpoint was saved (final step)
            mock_model.named_parameters.assert_called()

            # Verify training happened
            assert mock_trainer.train.call_count == 2
            assert mock_rollout.collect.call_count == 2
            mock_rollout.close.assert_called_once()
