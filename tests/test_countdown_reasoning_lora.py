"""
Smoke tests for experiments/countdown_reasoning_lora.py.
"""

from dataclasses import asdict


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
        assert cfg.model_id == "Qwen/Qwen3-0.6B"
        assert cfg.lora_rank == 2
        assert cfg.lora_layers == 4
        assert cfg.entropy_weight == 0.1
        assert cfg.hicra_alpha == 0.2
        assert cfg.output_dir == "results/countdown_reasoning_lora"

    def test_roundtrip_dict(self):
        from experiments.countdown_reasoning_lora import ReasoningConfig

        cfg = ReasoningConfig(max_steps=12, entropy_weight=0.2, hicra_alpha=0.3)
        restored = ReasoningConfig(**asdict(cfg))
        assert asdict(restored) == asdict(cfg)
