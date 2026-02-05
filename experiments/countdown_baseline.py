#!/usr/bin/env python3
"""
Baseline GRPO run on the Countdown task.

Self-contained experiment script that wires existing textpolicy components
together: LoRA fine-tuning, GRPO training, countdown rewards, and
emergence logging. Produces structured output for analysis.

Usage:
    uv run python experiments/countdown_baseline.py --steps 500
    uv run python experiments/countdown_baseline.py --model Qwen/Qwen3-0.6B --steps 10 --output results/test_run
"""

import argparse
import functools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.analysis import EmergenceLogger
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.lora import create_lora_setup
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.generation.reload import save_adapters
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.training import Trainer


@dataclass
class BaselineConfig:
    """Configuration for baseline GRPO countdown experiment."""

    # Model
    model_id: str = "Qwen/Qwen3-0.6B"

    # LoRA
    lora_layers: int = 8
    lora_rank: int = 8
    lora_scale: float = 20.0

    # Training
    learning_rate: float = 5e-6
    max_steps: int = 500
    max_grad_norm: float = 0.5

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

    # Output
    output_dir: str = "results/countdown_baseline"

    # Clip (DAPO-style asymmetric)
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28


def save_config(config: BaselineConfig, output_dir: Path) -> None:
    """Dump config as JSON for reproducibility."""
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")


def save_checkpoint(model, output_dir: Path, step: int) -> None:
    """Save LoRA adapter weights at a checkpoint."""
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(checkpoint_dir / "lora_adapters.safetensors")
    save_adapters(model, adapter_path)


def print_summary(output_dir: Path) -> None:
    """Read steps.jsonl and print key metrics."""
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
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"  Steps completed:  {last['step'] + 1}")
    print(f"  First step reward: {first['mean_reward']:.3f}")
    print(f"  Final step reward: {last['mean_reward']:.3f}")
    print(f"  Best accuracy:     {best_accuracy:.1%} (step {best_step})")
    print(f"  Final planning ratio: {last['planning_token_ratio']:.4f}")
    print("=" * 50)


def run_baseline(config: BaselineConfig) -> None:
    """Run the baseline GRPO experiment on countdown."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save config
    save_config(config, output_dir)

    # 2. Load model + tokenizer
    print(f"Loading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id)

    # 3. Apply LoRA
    print("Applying LoRA adapters...")
    model, memory_stats = create_lora_setup(
        model,
        lora_config={
            "lora_layers": config.lora_layers,
            "lora_rank": config.lora_rank,
            "lora_scale": config.lora_scale,
        },
    )
    print(f"  Memory savings: {memory_stats['memory_savings_percent']:.1f}%")

    # 4. Generate countdown problems
    print(f"Generating {config.num_problems} countdown problems (seed={config.dataset_seed})...")
    problems = generate_countdown_problems(
        config.num_problems, seed=config.dataset_seed
    )
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    # 5. Create environment factory
    def create_env():
        return TextGenerationEnv(
            prompts=prompts,
            reward_fn=countdown_reward,
            max_tokens=config.max_completion_tokens,
            tokenizer=tokenizer,
            examples=problems,
        )

    # 6. Create policy
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

    # 7. Create rollout coordinator
    rollout = RolloutCoordinator(
        env_fn=create_env,
        policy_fn=lambda: policy_fn,
        algorithm="grpo",
        num_workers=0,
        max_steps=config.episodes_per_step,
        max_episodes=config.episodes_per_step,
    )

    # 8. Create buffer + trainer
    buffer = Buffer(max_episodes=config.episodes_per_step)
    loss_fn = functools.partial(
        grpo.policy_loss,
        clip_ratio_low=config.clip_ratio_low,
        clip_ratio_high=config.clip_ratio_high,
    )
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=loss_fn,
        optimizer=optim.Adam(learning_rate=config.learning_rate),
        max_grad_norm=config.max_grad_norm,
        buffer=buffer,
        data_selector_fn=grpo.select_recent_data,
        compile_training=True,
    )

    # 9. EmergenceLogger
    emergence = EmergenceLogger(output_dir=output_dir / "emergence")

    # 10. Training loop
    print(f"\nStarting training for {config.max_steps} steps...")
    for step in range(config.max_steps):
        # Clear buffer each step (on-policy)
        buffer.clear()

        # Collect rollouts
        rollout_buffer = rollout.collect()
        for ep in rollout_buffer.episodes:
            buffer.add_episode_from_dict(ep.to_dict())

        # Map each episode to the correct problem.  The environment cycles
        # prompts via current_episode % len(prompts), so the global offset
        # must be accounted for — local episode index alone is wrong after
        # step 0.
        num_eps = len(rollout_buffer.episodes)
        step_examples = [
            problems[(step * config.episodes_per_step + i) % len(problems)]
            for i in range(num_eps)
        ]

        # Log generations BEFORE training
        step_stats = emergence.log_step(
            step=step,
            episodes=rollout_buffer.episodes,
            tokenizer=tokenizer,
            examples=step_examples,
        )

        # Train
        metrics = trainer.train()

        # Print progress
        if step % 10 == 0:
            print(
                f"Step {step}: loss={metrics['loss']:.4f} "
                f"reward={step_stats['mean_reward']:.3f} "
                f"correct={step_stats['correct_count']}/{step_stats['total_count']} "
                f"planning_ratio={step_stats['planning_token_ratio']:.4f}"
            )

        # Save checkpoint at intervals
        if step % 100 == 0 and step > 0:
            save_checkpoint(model, output_dir, step)

    # 11. Finish
    emergence.finish()
    rollout.close()
    save_checkpoint(model, output_dir, config.max_steps)
    print_summary(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run baseline GRPO on the Countdown task"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model ID")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument(
        "--output", default="results/countdown_baseline", help="Output directory"
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument(
        "--num-problems", type=int, default=50, help="Number of countdown problems"
    )
    parser.add_argument(
        "--episodes-per-step", type=int, default=8, help="Episodes per training step"
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=8, help="LoRA layers")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed")
    args = parser.parse_args()

    config = BaselineConfig(
        model_id=args.model,
        max_steps=args.steps,
        output_dir=args.output,
        learning_rate=args.lr,
        num_problems=args.num_problems,
        episodes_per_step=args.episodes_per_step,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        temperature=args.temperature,
        dataset_seed=args.seed,
    )
    run_baseline(config)
