#!/usr/bin/env python3
"""
Benchmark selective gradient checkpointing strategies.

Compares three checkpointing configurations on a real model:
  1. No checkpointing (baseline)
  2. sqrt(n) selective checkpointing (new default)
  3. Every-layer checkpointing (maximum memory savings)

Measures peak memory and training step time at multiple sequence lengths
to quantify the memory/compute trade-off from Chen et al. 2016.

Usage:
    uv run python experiments/benchmark_checkpointing.py
    uv run python experiments/benchmark_checkpointing.py --seq-lengths 256 512
    uv run python experiments/benchmark_checkpointing.py --json
"""

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.training import create_tinylora_reasoning_setup
from textpolicy.training.gradient_checkpointing import _get_layers
from textpolicy.utils.memory import clear_memory, get_memory_stats


@dataclass
class ProbeResult:
    strategy: str
    seq_length: int
    n_checkpointed: int
    n_layers: int
    train_time_s: float
    peak_memory_gb: float
    status: str = "OK"


def _reset_memory():
    """Best-effort memory cleanup between runs."""
    gc.collect()
    clear_memory()
    mx.reset_peak_memory()


def run_single_probe(
    *,
    model_id: str,
    gradient_checkpointing: Union[bool, int],
    seq_length: int,
    group_size: int,
    steps: int,
    problems: List[Dict],
    prompts: List[str],
) -> ProbeResult:
    """Run a single probe: load model, setup trainer, rollout, train, measure.

    Reloads the model from scratch for each probe to avoid LoRA-already-applied
    errors.  Model weights are served from HuggingFace cache so reload is fast.
    """
    _reset_memory()

    # Must check bool before int because True == 1 in Python.
    if gradient_checkpointing is False:
        strategy_name = "none"
    elif gradient_checkpointing is True:
        strategy_name = "sqrt(n)"
    else:
        strategy_name = f"every-{gradient_checkpointing}"

    # Load a fresh model (from cache) so LoRA can be applied cleanly.
    model, tokenizer = load_model(model_id)
    optimizer = optim.Adam(learning_rate=5e-6)

    trainer, _ = create_tinylora_reasoning_setup(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lora_config={
            "lora_layers": 4,
            "lora_rank": 2,
            "lora_scale": 8.0,
            "lora_dropout": 0.0,
        },
        auto_reload=False,
        compile_training=False,
        gradient_checkpointing=gradient_checkpointing,
        profile=True,
        max_grad_norm=0.5,
    )

    # Count checkpointed layers
    layers = _get_layers(trainer.model)
    n_layers = len(layers)
    n_ckpt = sum(
        1
        for layer in layers
        if getattr(type(layer), "_original_class", None) is not None
    )

    # Set up rollout — matching profile_hardware.py's pattern
    def create_env():
        return TextGenerationEnv(
            prompts=prompts,
            reward_fn=countdown_reward,
            max_tokens=seq_length,
            tokenizer=tokenizer,
            examples=problems,
            group_size=group_size,
        )

    policy_fn = create_policy(
        trainer.model,
        tokenizer,
        generation_params={
            "max_tokens": seq_length,
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
        max_steps=group_size,
        max_episodes=group_size,
        batch_size=group_size,
        model=trainer.model,
        tokenizer=tokenizer,
        generation_params={
            "max_tokens": seq_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    )

    buffer = Buffer(max_episodes=group_size)
    trainer.link_buffer(buffer, data_selector_fn=grpo.select_recent_data)

    # Reset peak memory before the measured training loop
    mx.reset_peak_memory()

    train_times: List[float] = []
    for _ in range(steps):
        buffer.clear()

        # Rollout
        rollout_buffer = rollout.collect()
        for ep in rollout_buffer.episodes:
            buffer.add_episode_from_dict(ep.to_dict())
        mx.eval(trainer.model.parameters())

        # Training (measured)
        t0 = time.perf_counter()
        trainer.train()
        mx.eval(trainer.model.parameters())
        t1 = time.perf_counter()
        train_times.append(t1 - t0)

    peak_gb = mx.get_peak_memory() / 1024 / 1024 / 1024
    avg_train = sum(train_times) / len(train_times)

    return ProbeResult(
        strategy=strategy_name,
        seq_length=seq_length,
        n_checkpointed=n_ckpt,
        n_layers=n_layers,
        train_time_s=round(avg_train, 3),
        peak_memory_gb=round(peak_gb, 2),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark selective gradient checkpointing strategies"
    )
    parser.add_argument(
        "--model",
        default="arcee-ai/Trinity-Nano-Preview",
        help="Model ID (default: arcee-ai/Trinity-Nano-Preview)",
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[256, 512],
        help="Sequence lengths to probe (default: 256 512)",
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="Episodes per step (default: 8)"
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="Training steps per probe (default: 3)"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=str, default=None, help="Save to file")
    args = parser.parse_args()

    strategies: List[Union[bool, int]] = [False, True, 1]
    strategy_labels = ["none", "sqrt(n)", "every-1"]

    # Quick model load just to get layer count
    print(f"Loading model: {args.model}")
    tmp_model, _ = load_model(args.model)
    tmp_layers = (
        tmp_model.model.layers
        if hasattr(tmp_model, "model")
        else tmp_model.layers
    )
    n_layers = len(tmp_layers)
    sqrt_stride = max(1, int(math.sqrt(n_layers)))
    n_sqrt_ckpt = len(range(0, n_layers, sqrt_stride))
    del tmp_model, tmp_layers
    _reset_memory()

    print(f"  Layers: {n_layers}")
    print(f"  sqrt(n) stride: {sqrt_stride} → {n_sqrt_ckpt} checkpointed layers")
    print(f"  Strategies: {strategy_labels}")
    print(f"  Sequence lengths: {args.seq_lengths}")
    print(f"  Group size: G={args.group_size}, steps: {args.steps}")

    problems = generate_countdown_problems(20, seed=42)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    all_results: List[ProbeResult] = []

    for seq_len in args.seq_lengths:
        print(f"\n{'='*60}")
        print(f"  seq_length = {seq_len}")
        print(f"{'='*60}")

        for strat, label in zip(strategies, strategy_labels):
            print(f"  [{label:>10}] ...", end=" ", flush=True)
            try:
                result = run_single_probe(
                    model_id=args.model,
                    gradient_checkpointing=strat,
                    seq_length=seq_len,
                    group_size=args.group_size,
                    steps=args.steps,
                    problems=problems,
                    prompts=prompts,
                )
                all_results.append(result)
                print(
                    f"train={result.train_time_s:.2f}s  "
                    f"peak={result.peak_memory_gb:.2f}GB  "
                    f"ckpt={result.n_checkpointed}/{result.n_layers}"
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"FAILED: {e}")
                all_results.append(
                    ProbeResult(
                        strategy=label,
                        seq_length=seq_len,
                        n_checkpointed=0,
                        n_layers=n_layers,
                        train_time_s=0.0,
                        peak_memory_gb=0.0,
                        status=f"FAILED: {e}",
                    )
                )

    # Summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(
        f"{'Strategy':>12} {'SeqLen':>8} {'Train(s)':>10} {'Peak(GB)':>10} "
        f"{'Ckpt Layers':>12} {'Status':>8}"
    )
    print("-" * 70)

    for r in all_results:
        print(
            f"{r.strategy:>12} {r.seq_length:>8} {r.train_time_s:>10.3f} "
            f"{r.peak_memory_gb:>10.2f} {r.n_checkpointed:>5}/{r.n_layers:<5} "
            f"{r.status:>8}"
        )

    # Compute relative metrics
    print(f"\n{'='*80}")
    print("RELATIVE COMPARISON (vs no-checkpointing baseline)")
    print(f"{'='*80}")

    for seq_len in args.seq_lengths:
        baseline = next(
            (
                r
                for r in all_results
                if r.seq_length == seq_len and r.strategy == "none"
            ),
            None,
        )
        if baseline is None or baseline.status != "OK":
            continue

        print(f"\n  seq_length={seq_len}:")
        for r in all_results:
            if r.seq_length != seq_len or r.status != "OK":
                continue
            time_delta = (
                (r.train_time_s - baseline.train_time_s)
                / baseline.train_time_s
                * 100
                if baseline.train_time_s > 0
                else 0.0
            )
            mem_delta = (
                (r.peak_memory_gb - baseline.peak_memory_gb)
                / baseline.peak_memory_gb
                * 100
                if baseline.peak_memory_gb > 0
                else 0.0
            )
            print(
                f"    {r.strategy:>12}: time {time_delta:+.1f}%  "
                f"memory {mem_delta:+.1f}%  "
                f"({r.n_checkpointed}/{r.n_layers} layers)"
            )

    if args.json:
        output = {
            "model": args.model,
            "n_layers": n_layers,
            "sqrt_stride": sqrt_stride,
            "results": [asdict(r) for r in all_results],
        }
        json_str = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json_str)
            print(f"\nJSON saved to {args.output}")
        else:
            print(json_str)
    elif args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        lines = []
        lines.append(f"Checkpointing Benchmark: {args.model}")
        lines.append(f"Layers: {n_layers}, sqrt(n) stride: {sqrt_stride}\n")
        lines.append(
            f"{'Strategy':>12} {'SeqLen':>8} {'Train(s)':>10} {'Peak(GB)':>10} "
            f"{'Ckpt':>5}"
        )
        lines.append("-" * 55)
        for r in all_results:
            if r.status == "OK":
                lines.append(
                    f"{r.strategy:>12} {r.seq_length:>8} {r.train_time_s:>10.3f} "
                    f"{r.peak_memory_gb:>10.2f} {r.n_checkpointed:>3}/{r.n_layers}"
                )
        Path(args.output).write_text("\n".join(lines) + "\n")
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
