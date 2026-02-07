#!/usr/bin/env python3
"""
LoRA reasoning run on the Countdown task (GTPO + HICRA).

This experiment composes:
- TinyLoRA-style PEFT defaults
- GTPO entropy-weighted token credit
- HICRA planning-token amplification

Usage:
    uv run python experiments/countdown_reasoning_lora.py --steps 500
    uv run python experiments/countdown_reasoning_lora.py --model arcee-ai/Trinity-Nano-Preview --steps 10 --output results/test_reasoning
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.analysis import EmergenceLogger, load_strategic_grams
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.generation.reload import save_adapters
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.training import create_tinylora_reasoning_setup


@dataclass
class ReasoningConfig:
    """Configuration for LoRA reasoning experiment."""

    # Model
    model_id: str = "arcee-ai/Trinity-Nano-Preview"

    # TinyLoRA-style defaults
    lora_layers: int = 4
    lora_rank: int = 2
    lora_scale: float = 8.0
    lora_dropout: float = 0.0

    # Reasoning shaping
    entropy_weight: float = 0.1
    hicra_alpha: float = 0.2
    strategic_grams_path: Optional[str] = None

    # Training
    learning_rate: float = 5e-6
    max_steps: int = 500
    max_grad_norm: float = 0.5
    compile_training: Union[bool, str] = False
    profile_training: bool = False

    # Generation
    max_completion_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Dataset
    num_problems: int = 50
    dataset_seed: int = 42

    # Rollout
    episodes_per_step: int = 8
    batch_size: int = 8

    # Output
    output_dir: str = "results/countdown_reasoning_lora"


_RUNNER_INTERNAL_MAX_EPISODES = 10  # hardcoded in RolloutRunner.__init__


def save_config(config: ReasoningConfig, output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")


def save_checkpoint(model, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(checkpoint_dir / "lora_adapters.safetensors")
    save_adapters(model, adapter_path)


def print_summary(output_dir: Path) -> None:
    steps_path = output_dir / "emergence" / "steps.jsonl"
    if not steps_path.exists():
        print("No steps.jsonl found — nothing to summarize.")
        return

    steps = []
    with open(steps_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    if not steps:
        print("No step records found.")
        return

    first = steps[0]
    last = steps[-1]
    best_accuracy = 0.0
    best_step = 0
    for s in steps:
        total = s.get("total_count", 0)
        if total > 0:
            acc = s.get("correct_count", 0) / total
            if acc > best_accuracy:
                best_accuracy = acc
                best_step = s["step"]

    print("\n" + "=" * 50)
    print("REASONING EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"  Steps completed:  {last['step'] + 1}")
    print(f"  First step reward: {first['mean_reward']:.3f}")
    print(f"  Final step reward: {last['mean_reward']:.3f}")
    print(f"  Best accuracy:     {best_accuracy:.1%} (step {best_step})")
    print(f"  Final planning ratio: {last['planning_token_ratio']:.4f}")
    print("=" * 50)


def print_amdahl_summary(
    phase_totals: Dict[str, float],
    trainer_phase_totals: Dict[str, float],
) -> None:
    total = sum(phase_totals.values())
    if total <= 0.0:
        return

    print("\n" + "=" * 50)
    print("AMDAHL BOTTLENECK SUMMARY (END-TO-END)")
    print("=" * 50)
    ranked = sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True)
    for phase, seconds in ranked:
        fraction = seconds / total
        max_speedup = float("inf") if fraction >= 0.999999 else 1.0 / (1.0 - fraction)
        name = phase.replace("_s", "")
        speedup_str = "inf" if max_speedup == float("inf") else f"{max_speedup:.2f}x"
        print(
            f"  {name:16s} {seconds:8.2f}s  "
            f"({fraction * 100:5.1f}%)  Amdahl limit if perfect: {speedup_str}"
        )

    if trainer_phase_totals:
        trainer_total = trainer_phase_totals.get("total", 0.0)
        if trainer_total > 0.0:
            print("\n  Trainer-internal split (for train phase):")
            t_ranked = sorted(
                (
                    (phase, secs)
                    for phase, secs in trainer_phase_totals.items()
                    if phase != "total"
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for phase, seconds in t_ranked:
                pct = (seconds / trainer_total) * 100.0
                print(f"    {phase:16s} {seconds:8.2f}s  ({pct:5.1f}%)")
    print("=" * 50)


def run_experiment(config: ReasoningConfig) -> None:
    if config.episodes_per_step > _RUNNER_INTERNAL_MAX_EPISODES:
        raise ValueError(
            f"episodes_per_step ({config.episodes_per_step}) exceeds "
            f"RolloutRunner's internal buffer capacity "
            f"({_RUNNER_INTERNAL_MAX_EPISODES}). Episodes would be "
            f"silently evicted before reaching the training buffer."
        )

    if config.batch_size > config.episodes_per_step:
        raise ValueError(
            f"batch_size ({config.batch_size}) exceeds episodes_per_step "
            f"({config.episodes_per_step}). batch_size should be <= episodes_per_step."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    print(f"Loading model: {config.model_id}")
    base_model, tokenizer = load_model(config.model_id)

    strategic_grams = None
    if config.strategic_grams_path:
        strategic_grams = load_strategic_grams(config.strategic_grams_path)
        print(
            f"Loaded {len(strategic_grams)} strategic grams from "
            f"{config.strategic_grams_path}"
        )

    optimizer = optim.Adam(learning_rate=config.learning_rate)
    trainer, memory_stats = create_tinylora_reasoning_setup(
        model=base_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lora_config={
            "lora_layers": config.lora_layers,
            "lora_rank": config.lora_rank,
            "lora_scale": config.lora_scale,
            "lora_dropout": config.lora_dropout,
        },
        strategic_grams=strategic_grams,
        hicra_alpha=config.hicra_alpha,
        entropy_weight=config.entropy_weight,
        compile_training=config.compile_training,
        profile=config.profile_training,
        max_grad_norm=config.max_grad_norm,
        adapter_save_path=str(output_dir / "lora_adapters.safetensors"),
    )
    model = trainer.model

    print(
        "Reasoning stack ready: "
        f"entropy_weight={config.entropy_weight}, hicra_alpha={config.hicra_alpha}"
    )
    print(
        "  Memory savings: "
        f"{memory_stats.get('memory_savings_percent', 0.0):.1f}%"
    )

    print(f"Generating {config.num_problems} countdown problems (seed={config.dataset_seed})...")
    problems = generate_countdown_problems(config.num_problems, seed=config.dataset_seed)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    def create_env():
        return TextGenerationEnv(
            prompts=prompts,
            reward_fn=countdown_reward,
            max_tokens=config.max_completion_tokens,
            tokenizer=tokenizer,
            examples=problems,
        )

    policy_fn = create_policy(
        model,
        tokenizer,
        generation_params={
            "max_tokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
    )

    rollout = RolloutCoordinator(
        env_fn=create_env,
        policy_fn=lambda: policy_fn,
        algorithm="grpo",
        num_workers=0,
        max_steps=config.episodes_per_step,
        max_episodes=config.episodes_per_step,
        batch_size=config.batch_size,
        model=model,
        tokenizer=tokenizer,
        generation_params={
            "max_tokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
    )

    buffer = Buffer(max_episodes=config.episodes_per_step)
    trainer.link_buffer(buffer, data_selector_fn=grpo.select_recent_data)
    emergence = EmergenceLogger(output_dir=output_dir / "emergence")

    print(f"\nStarting reasoning training for {config.max_steps} steps...")
    global_episode_count = 0
    phase_totals: Dict[str, float] = {
        "rollout_collect_s": 0.0,
        "emergence_log_s": 0.0,
        "train_s": 0.0,
        "checkpoint_s": 0.0,
    }
    trainer_phase_totals: Dict[str, float] = {}
    for step in range(config.max_steps):
        step_start = time.perf_counter()

        buffer.clear()
        rollout_start = time.perf_counter()
        rollout_buffer = rollout.collect()
        for ep in rollout_buffer.episodes:
            buffer.add_episode_from_dict(ep.to_dict())
        phase_totals["rollout_collect_s"] += time.perf_counter() - rollout_start

        new_episode = rollout_buffer.episodes[-1]
        example = problems[global_episode_count % len(problems)]
        global_episode_count += 1

        emergence_start = time.perf_counter()
        step_stats = emergence.log_step(
            step=step,
            episodes=[new_episode],
            tokenizer=tokenizer,
            examples=[example],
        )
        phase_totals["emergence_log_s"] += time.perf_counter() - emergence_start

        train_start = time.perf_counter()
        metrics = trainer.train()
        phase_totals["train_s"] += time.perf_counter() - train_start
        for key, value in metrics.items():
            if key.startswith("timing/") and key.endswith("_s"):
                phase = key[len("timing/") : -2]
                trainer_phase_totals[phase] = trainer_phase_totals.get(phase, 0.0) + float(value)

        if step % 10 == 0:
            cumulative_total = sum(phase_totals.values()) or 1e-9
            rollout_pct = phase_totals["rollout_collect_s"] / cumulative_total * 100
            train_pct = phase_totals["train_s"] / cumulative_total * 100
            print(
                f"Step {step}: loss={metrics['loss']:.4f} "
                f"reward={step_stats['mean_reward']:.3f} "
                f"correct={step_stats['correct_count']}/{step_stats['total_count']} "
                f"planning_ratio={step_stats['planning_token_ratio']:.4f} "
                f"step_time={time.perf_counter() - step_start:.2f}s "
                f"[rollout {rollout_pct:.0f}% | train {train_pct:.0f}%]"
            )

        if step % 100 == 0 and step > 0:
            checkpoint_start = time.perf_counter()
            save_checkpoint(model, output_dir, step)
            phase_totals["checkpoint_s"] += time.perf_counter() - checkpoint_start

    emergence.finish()
    rollout.close()
    checkpoint_start = time.perf_counter()
    save_checkpoint(model, output_dir, config.max_steps)
    phase_totals["checkpoint_s"] += time.perf_counter() - checkpoint_start
    print_summary(output_dir)
    print_amdahl_summary(phase_totals, trainer_phase_totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TinyLoRA + GTPO + HICRA on the Countdown task"
    )
    parser.add_argument("--model", default="arcee-ai/Trinity-Nano-Preview", help="Model ID")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--output", default="results/countdown_reasoning_lora", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-problems", type=int, default=50, help="Number of countdown problems")
    parser.add_argument("--episodes-per-step", type=int, default=8, help="Episodes per training step")
    parser.add_argument("--batch-size", type=int, default=8, help="Batched generation across episodes")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed")
    parser.add_argument("--lora-rank", type=int, default=2, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=4, help="LoRA layers")
    parser.add_argument("--lora-scale", type=float, default=8.0, help="LoRA scale")
    parser.add_argument("--entropy-weight", type=float, default=0.1, help="GTPO entropy weight")
    parser.add_argument("--hicra-alpha", type=float, default=0.2, help="HICRA amplification alpha")
    parser.add_argument(
        "--compile-training",
        choices=["false", "true", "auto"],
        default="false",
        help="Training compilation mode (HICRA defaults to false for compatibility)",
    )
    parser.add_argument(
        "--strategic-grams",
        default=None,
        help="Optional path to strategic grams JSON (defaults to built-ins)",
    )
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help="Enable trainer per-phase timing and Amdahl bottleneck summary",
    )
    args = parser.parse_args()

    compile_mode: Union[bool, str]
    if args.compile_training == "true":
        compile_mode = True
    elif args.compile_training == "false":
        compile_mode = False
    else:
        compile_mode = "auto"

    cfg = ReasoningConfig(
        model_id=args.model,
        max_steps=args.steps,
        output_dir=args.output,
        learning_rate=args.lr,
        num_problems=args.num_problems,
        episodes_per_step=args.episodes_per_step,
        batch_size=args.batch_size,
        temperature=args.temperature,
        dataset_seed=args.seed,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        lora_scale=args.lora_scale,
        entropy_weight=args.entropy_weight,
        hicra_alpha=args.hicra_alpha,
        strategic_grams_path=args.strategic_grams,
        compile_training=compile_mode,
        profile_training=args.profile_training,
    )
    run_experiment(cfg)
