#!/usr/bin/env python3
"""
Amdahl's-law profiling benchmark for textpolicy (Issue #25).

Measures per-phase wall-clock time inside the training loop so we can
identify bottlenecks *before* optimising (Issue #23).

Two modes
---------
**Synthetic** (default, no downloads)::

    uv run python scripts/benchmark_amdahl_v2.py

**Full pipeline** (real model, end-to-end)::

    uv run python scripts/benchmark_amdahl_v2.py --model Qwen/Qwen3-0.6B --steps 5

Outputs
-------
- JSON to ``artefacts/perf/amdahl_{mode}_{timestamp}.json``
- Markdown summary table to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.training.trainer import Trainer


# ---------------------------------------------------------------------------
# TinyLM — a minimal language-model-shaped network
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """
    Embedding → 4 linear layers → LM head.

    Produces logits of shape ``[batch, seq_len, vocab_size]`` which is all the
    Trainer's logprob extraction code needs.
    """

    def __init__(self, vocab_size: int = 256, dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = [nn.Linear(dim, dim) for _ in range(4)]
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embed(x)
        for layer in self.layers:
            h = nn.relu(layer(h))
        return self.head(h)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_batch(
    num_episodes: int, episode_length: int, vocab_size: int = 256
) -> Dict[str, Any]:
    """Build a flat-1D batch that mirrors real rollout data."""
    total = num_episodes * episode_length
    # 2D obs/act for the GRPO extraction path
    obs = mx.random.randint(0, vocab_size, shape=(num_episodes, episode_length))
    act = mx.random.randint(0, vocab_size, shape=(num_episodes, episode_length))
    logprob = -mx.abs(mx.random.normal((total,)))
    rewards = mx.random.normal((num_episodes,))
    return {
        "obs": obs,
        "act": act,
        "logprob": logprob,
        "rewards": rewards,
        "episode_lengths": [episode_length] * num_episodes,
    }


def _format_table(rows: List[List[str]], header: List[str]) -> str:
    """Render a simple markdown table."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    def _row(cols: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    lines = [_row(header), sep] + [_row(r) for r in rows]
    return "\n".join(lines)


def _save_json(data: Dict[str, Any], output_dir: str, mode: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / f"amdahl_{mode}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


# ---------------------------------------------------------------------------
# Synthetic benchmark
# ---------------------------------------------------------------------------

def run_synthetic(args: argparse.Namespace) -> Dict[str, Any]:
    """Benchmark Trainer internals with a TinyLM (no model download)."""
    mx.random.seed(args.seed)

    model = TinyLM(vocab_size=256, dim=64)
    optimizer = optim.Adam(learning_rate=1e-4)

    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optimizer,
        compile_training=False,  # compile interferes with per-episode forward pass
        profile=True,
    )

    batch = _make_synthetic_batch(args.episodes, args.episode_length)

    # Warmup
    for _ in range(args.warmup):
        trainer.train(batch)
        trainer._timer.reset()  # type: ignore[union-attr]

    # Measured steps
    all_metrics: List[Dict[str, float]] = []
    for _ in range(args.steps):
        m = trainer.train(batch)
        all_metrics.append(m)
        trainer._timer.reset()  # type: ignore[union-attr]

    return _aggregate(all_metrics, "synthetic", args)


# ---------------------------------------------------------------------------
# Full-pipeline benchmark
# ---------------------------------------------------------------------------

def run_full_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """End-to-end benchmark with a real model (requires --model)."""
    from textpolicy.generation.mlx_generation import load_model

    mx.random.seed(args.seed)
    model, tokenizer = load_model(args.model)

    optimizer = optim.Adam(learning_rate=1e-5)
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optimizer,
        compile_training=False,
        profile=True,
    )

    prompts = [f"Explain concept number {i} briefly." for i in range(args.episodes)]

    # Simple rollout timing
    outer_times: Dict[str, List[float]] = {
        "rollout": [], "buffer_transfer": [], "training": [],
    }

    from textpolicy.environment.text_generation import TextGenerationEnv
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy
    from textpolicy.buffer import Buffer
    import textpolicy as tp

    @tp.reward
    def _dummy_reward(prompt: str, completion: str, example: Dict[str, Any], **kw: Any) -> float:
        return float(len(completion.split()))

    env = TextGenerationEnv(
        prompts, _dummy_reward, max_tokens=args.episode_length, tokenizer=tokenizer
    )

    def policy_fn(obs: mx.array, deterministic: bool = False):
        logits = model(obs[None])
        tokens = mx.argmax(logits[0, -1:], axis=-1)
        return tokens, {}

    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=policy_fn, strategy=strategy, max_steps=args.episodes)

    all_metrics: List[Dict[str, float]] = []
    for step in range(args.warmup + args.steps):
        # Rollout
        t0 = time.perf_counter()
        rollout_buffer = runner.collect()
        mx.eval(rollout_buffer.episodes[0].obs if rollout_buffer.episodes else mx.array(0))
        t_rollout = time.perf_counter() - t0

        # Buffer transfer
        t0 = time.perf_counter()
        buffer = Buffer(max_episodes=args.episodes * 2)
        for ep in rollout_buffer.episodes:
            buffer.add_episode_from_dict(ep.to_dict())
        t_buffer = time.perf_counter() - t0

        # Training
        t0 = time.perf_counter()
        m = trainer.train(buffer)
        t_training = time.perf_counter() - t0

        trainer._timer.reset()  # type: ignore[union-attr]

        if step >= args.warmup:
            outer_times["rollout"].append(t_rollout)
            outer_times["buffer_transfer"].append(t_buffer)
            outer_times["training"].append(t_training)
            all_metrics.append(m)

    result = _aggregate(all_metrics, "full_pipeline", args)
    # Add outer-loop timing
    for phase, times in outer_times.items():
        if times:
            result["outer_phases"][phase] = {
                "mean_s": sum(times) / len(times),
                "min_s": min(times),
                "max_s": max(times),
            }
    return result


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------

def _aggregate(
    all_metrics: List[Dict[str, float]], mode: str, args: argparse.Namespace
) -> Dict[str, Any]:
    """Aggregate per-step timing metrics into a summary."""
    timing_keys = sorted({k for m in all_metrics for k in m if k.startswith("timing/")})
    summary: Dict[str, Any] = {
        "mode": mode,
        "episodes": args.episodes,
        "episode_length": args.episode_length,
        "warmup": args.warmup,
        "steps": args.steps,
        "phases": {},
        "outer_phases": {},
    }
    for key in timing_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if vals:
            summary["phases"][key] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
    return summary


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a markdown summary table to stdout."""
    mode = result["mode"]
    print(f"\n## Amdahl Benchmark — {mode}")
    print(f"Episodes: {result['episodes']}, length: {result['episode_length']}, "
          f"steps: {result['steps']} (warmup: {result['warmup']})\n")

    # Inner phases (Trainer)
    rows = []
    total_s = result["phases"].get("timing/total_s", {}).get("mean", 0.0)
    for key in sorted(result["phases"]):
        if key.endswith("_pct") or key == "timing/total_s":
            continue  # show per-phase seconds rows; total printed separately
        phase_name = key.removeprefix("timing/").removesuffix("_s")
        mean_s = result["phases"][key]["mean"]
        pct_key = key.removesuffix("_s") + "_pct"
        pct = result["phases"].get(pct_key, {}).get("mean", 0.0)
        rows.append([phase_name, f"{mean_s:.5f}", f"{pct:.1f}%"])
    if total_s:
        rows.append(["**total**", f"{total_s:.5f}", "100.0%"])

    if rows:
        print(_format_table(rows, ["Phase", "Mean (s)", "% of total"]))

    # Outer phases (full pipeline only)
    if result.get("outer_phases"):
        print("\n### Outer loop phases\n")
        rows = []
        for phase, stats in result["outer_phases"].items():
            rows.append([phase, f"{stats['mean_s']:.5f}", f"{stats['min_s']:.5f}", f"{stats['max_s']:.5f}"])
        print(_format_table(rows, ["Phase", "Mean (s)", "Min (s)", "Max (s)"]))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Amdahl's-law profiling benchmark for textpolicy."
    )
    parser.add_argument("--episodes", type=int, default=4,
                        help="Number of episodes per batch (default: 4)")
    parser.add_argument("--episode-length", type=int, default=32,
                        help="Tokens per episode (default: 32)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup steps excluded from measurement (default: 2)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Measured training steps (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID for full-pipeline mode "
                             "(e.g. Qwen/Qwen3-0.6B). Omit for synthetic mode.")
    parser.add_argument("--output-dir", type=str, default="artefacts/perf",
                        help="Directory for JSON output (default: artefacts/perf)")
    args = parser.parse_args()

    if args.model:
        mode = "full_pipeline"
        result = run_full_pipeline(args)
    else:
        mode = "synthetic"
        result = run_synthetic(args)

    _print_summary(result)

    path = _save_json(result, args.output_dir, mode)
    print(f"Results saved to: {path}")


if __name__ == "__main__":
    main()
