#!/usr/bin/env python3
"""
Sweep learning rates on a log-uniform grid (arXiv 2602.04998).

With the alpha=rank convention the scaling factor is fixed, so the
learning rate is the single remaining knob.  This script runs a short
training probe at each LR and reports final loss / mean reward.

Usage:
    uv run python experiments/sweep_lr.py
    uv run python experiments/sweep_lr.py --steps 30 --lr-min 1e-5 --lr-max 1e-3
    uv run python experiments/sweep_lr.py --lora-rank 8 --lora-layers 8
    uv run python experiments/sweep_lr.py --json --output results/sweep_lr/results.json
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.algorithms.grpo import compute_advantages, policy_loss
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.lora import create_lora_setup
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.training import Trainer, build_gtpo_transform
from textpolicy.utils.memory import clear_memory


@dataclass
class SweepConfig:
    """Configuration for LR sweep."""

    model_id: str = "arcee-ai/Trinity-Nano-Preview"
    lora_rank: int = 2
    lora_layers: int = 4
    steps_per_lr: int = 30
    episodes_per_step: int = 8
    batch_size: int = 8
    max_completion_tokens: int = 256
    num_problems: int = 50
    dataset_seed: int = 42
    lr_min: float = 1e-6
    lr_max: float = 1e-3
    points_per_decade: int = 4
    output_dir: str = "results/sweep_lr"


@dataclass
class ProbeResult:
    """Result from a single LR probe."""

    lr: float
    final_loss: float
    mean_reward: float
    std_reward: float
    elapsed_s: float
    status: str = "OK"
    error_msg: Optional[str] = None


def build_lr_grid(
    lr_min: float, lr_max: float, points_per_decade: int
) -> List[float]:
    """Build a log-uniform grid of learning rates.

    Returns ``points_per_decade`` values per decade between ``lr_min``
    and ``lr_max`` (inclusive of both endpoints).
    """
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)
    n_decades = log_max - log_min
    n_points = max(2, int(round(n_decades * points_per_decade)) + 1)
    return [10 ** (log_min + i * (log_max - log_min) / (n_points - 1)) for i in range(n_points)]


def run_single_probe(
    config: SweepConfig,
    lr: float,
    problems: List[Dict[str, Any]],
    prompts: List[str],
) -> ProbeResult:
    """Run a short training probe at a single learning rate.

    Loads a fresh model for each LR to avoid cross-contamination.
    """
    clear_memory()

    try:
        model, tokenizer = load_model(config.model_id)

        # LoRA: alpha = rank convention
        lora_scale = float(config.lora_rank)
        lora_model, _ = create_lora_setup(
            model=model,
            lora_config={
                "lora_layers": config.lora_layers,
                "lora_rank": config.lora_rank,
                "lora_scale": lora_scale,
                "lora_dropout": 0.0,
            },
            auto_reload=False,
        )

        optimizer = optim.Adam(learning_rate=lr)
        transform = build_gtpo_transform(tokenizer=tokenizer, hicra_gamma=0.3)

        trainer = Trainer(
            model=lora_model,
            advantage_fn=compute_advantages,
            loss_fn=policy_loss,
            optimizer=optimizer,
            advantage_transform_fn=transform,
            compile_training=False,
            max_grad_norm=0.5,
        )

        def create_env():
            return TextGenerationEnv(
                prompts=prompts,
                reward_fn=countdown_reward,
                max_tokens=config.max_completion_tokens,
                tokenizer=tokenizer,
                examples=problems,
                group_size=config.episodes_per_step,
            )

        policy_fn = create_policy(
            lora_model,
            tokenizer,
            generation_params={
                "max_tokens": config.max_completion_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
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
            model=lora_model,
            tokenizer=tokenizer,
            generation_params={
                "max_tokens": config.max_completion_tokens,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
        )

        buffer = Buffer(max_episodes=config.episodes_per_step)
        trainer.link_buffer(buffer, data_selector_fn=grpo.select_recent_data)

        losses: List[float] = []
        rewards: List[float] = []
        t0 = time.perf_counter()

        for step in range(config.steps_per_lr):
            buffer.clear()

            rollout_buffer = rollout.collect()
            for ep in rollout_buffer.episodes:
                buffer.add_episode_from_dict(ep.to_dict())

            # Collect rewards from latest episodes
            for ep in rollout_buffer.episodes:
                rew = ep.rew
                if isinstance(rew, (list, tuple)) and rew:
                    val = rew[0]
                    if hasattr(val, "item"):
                        val = val.item()
                    rewards.append(float(val))

            metrics = trainer.train()
            losses.append(float(metrics["loss"]))

        elapsed = time.perf_counter() - t0
        rollout.close()

        final_loss = losses[-1] if losses else float("nan")
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        std_reward = (
            (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if rewards
            else 0.0
        )

        return ProbeResult(
            lr=lr,
            final_loss=final_loss,
            mean_reward=mean_reward,
            std_reward=std_reward,
            elapsed_s=round(elapsed, 1),
        )

    except Exception as e:
        return ProbeResult(
            lr=lr,
            final_loss=float("nan"),
            mean_reward=0.0,
            std_reward=0.0,
            elapsed_s=0.0,
            status="FAILED",
            error_msg=str(e)[:200],
        )


def run_sweep(config: SweepConfig) -> List[ProbeResult]:
    """Run the full LR sweep."""
    grid = build_lr_grid(config.lr_min, config.lr_max, config.points_per_decade)

    print(f"LR sweep: {len(grid)} points from {config.lr_min:.0e} to {config.lr_max:.0e}")
    print(f"  Model: {config.model_id}")
    print(f"  LoRA: rank={config.lora_rank}, scale={config.lora_rank} (alpha=rank)")
    print(f"  Steps per LR: {config.steps_per_lr}")
    print(f"  Grid: {[f'{lr:.2e}' for lr in grid]}")
    print()

    # Prepare dataset once
    problems = generate_countdown_problems(config.num_problems, seed=config.dataset_seed)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    results: List[ProbeResult] = []

    for i, lr in enumerate(grid):
        print(f"  [{i + 1}/{len(grid)}] lr={lr:.2e} ...", end=" ", flush=True)
        result = run_single_probe(config, lr, problems, prompts)
        results.append(result)

        if result.status == "OK":
            print(
                f"loss={result.final_loss:.4f}  "
                f"reward={result.mean_reward:.3f} +/- {result.std_reward:.3f}  "
                f"({result.elapsed_s:.0f}s)"
            )
        else:
            print(f"FAILED: {result.error_msg}")

    return results


def print_results_table(results: List[ProbeResult]) -> None:
    """Print sorted results table, highlighting the best LR."""
    ok_results = [r for r in results if r.status == "OK"]
    if not ok_results:
        print("\nNo successful probes.")
        return

    # Sort by mean reward (descending), then by loss (ascending) as tiebreaker
    sorted_results = sorted(ok_results, key=lambda r: (-r.mean_reward, r.final_loss))
    best = sorted_results[0]

    print("\n" + "=" * 70)
    print("LR SWEEP RESULTS (sorted by reward)")
    print("=" * 70)
    print(f"{'LR':>12} | {'Final Loss':>11} | {'Mean Reward':>12} | {'Std':>8} | {'Time':>6} | ")
    print("-" * 65)

    for r in sorted_results:
        marker = " <-- BEST" if r.lr == best.lr else ""
        print(
            f"  {r.lr:.2e} | {r.final_loss:>11.4f} | {r.mean_reward:>12.3f} | "
            f"{r.std_reward:>8.3f} | {r.elapsed_s:>5.0f}s |{marker}"
        )

    print("=" * 70)
    print(f"\nBest LR: {best.lr:.2e} (reward={best.mean_reward:.3f}, loss={best.final_loss:.4f})")
    print(f"Effective step size (lr): {best.lr:.2e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep learning rates on a log-uniform grid (arXiv 2602.04998)"
    )
    parser.add_argument(
        "--model", default="arcee-ai/Trinity-Nano-Preview", help="Model ID"
    )
    parser.add_argument("--lora-rank", type=int, default=2, help="LoRA rank (alpha=rank)")
    parser.add_argument("--lora-layers", type=int, default=4, help="LoRA layers")
    parser.add_argument(
        "--steps", type=int, default=30, help="Training steps per LR point"
    )
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR")
    parser.add_argument("--lr-max", type=float, default=1e-3, help="Maximum LR")
    parser.add_argument(
        "--points-per-decade",
        type=int,
        default=4,
        help="Grid points per decade (default: 4)",
    )
    parser.add_argument(
        "--output", default="results/sweep_lr", help="Output directory"
    )
    parser.add_argument("--json", action="store_true", help="Save results as JSON")
    args = parser.parse_args()

    config = SweepConfig(
        model_id=args.model,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        steps_per_lr=args.steps,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        points_per_decade=args.points_per_decade,
        output_dir=args.output,
    )

    results = run_sweep(config)
    print_results_table(results)

    if args.json:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
        output = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nJSON saved to {output_path}")


if __name__ == "__main__":
    main()
