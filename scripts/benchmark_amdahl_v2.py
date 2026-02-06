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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.generation.mlx_generation import compute_prompt_reuse_stats
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
    num_episodes: int,
    episode_length: int,
    vocab_size: int = 256,
    prompt_pool_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a 2D batch that mirrors real rollout data from _pack_episodes.

    The batch includes ``prompt_lengths`` so the Trainer routes through
    ``compute_logprobs_batched`` (single forward pass) rather than falling
    back to the sequential per-episode path.
    """
    prompt_length = episode_length // 2
    response_length = episode_length - prompt_length
    total = num_episodes * response_length
    if prompt_pool_size is None:
        prompt_pool_size = num_episodes
    prompt_pool_size = max(1, min(prompt_pool_size, num_episodes))

    # 2D obs/act for the GRPO batched extraction path.
    # obs = full sequence (prompt + response), act = response only.
    if prompt_pool_size >= num_episodes:
        obs = mx.random.randint(0, vocab_size, shape=(num_episodes, episode_length))
        act = mx.random.randint(0, vocab_size, shape=(num_episodes, response_length))
    else:
        prompt_pool = [
            mx.random.randint(0, vocab_size, shape=(prompt_length,))
            for _ in range(prompt_pool_size)
        ]
        obs_rows: List[mx.array] = []
        act_rows: List[mx.array] = []
        for i in range(num_episodes):
            prompt = prompt_pool[i % prompt_pool_size]
            response = mx.random.randint(0, vocab_size, shape=(response_length,))
            obs_rows.append(mx.concatenate([prompt, response]))
            act_rows.append(response)
        obs = mx.stack(obs_rows)
        act = mx.stack(act_rows)
    logprob = -mx.abs(mx.random.normal((total,)))
    rewards = mx.random.normal((num_episodes,))
    return {
        "obs": obs,
        "act": act,
        "logprob": logprob,
        "rewards": rewards,
        "episode_lengths": [response_length] * num_episodes,
        "prompt_lengths": [prompt_length] * num_episodes,
    }


def _format_table(rows: List[List[str]], header: List[str]) -> str:
    """Render a simple markdown table."""
    widths = [max(len(h), *(len(r[i]) for r in rows), 0) for i, h in enumerate(header)]
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

    batch = _make_synthetic_batch(
        args.episodes,
        args.episode_length,
        prompt_pool_size=args.prompt_pool_size,
    )
    prompt_reuse = compute_prompt_reuse_stats(
        batch["obs"],
        batch["prompt_lengths"],
        batch["episode_lengths"],
    )

    # Warmup
    for _ in range(args.warmup):
        trainer.train(batch)
        trainer.reset_timer()

    # Measured steps
    all_metrics: List[Dict[str, float]] = []
    for _ in range(args.steps):
        m = trainer.train(batch)
        all_metrics.append(m)
        trainer.reset_timer()

    prompt_reuse_samples = [prompt_reuse for _ in all_metrics]
    return _aggregate(all_metrics, "synthetic", args, prompt_reuse_samples)


# ---------------------------------------------------------------------------
# Full-pipeline benchmark
# ---------------------------------------------------------------------------

def run_full_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """End-to-end benchmark with a real model (requires --model)."""
    from textpolicy.generation.mlx_generation import load_model

    mx.random.seed(args.seed)
    model, tokenizer = load_model(args.model)

    # Optional LoRA setup — freeze base, keep only adapters trainable.
    if args.lora:
        from textpolicy.generation.lora import create_lora_setup

        model, memory_stats = create_lora_setup(
            model,
            lora_config={
                "lora_layers": args.lora_layers,
                "lora_rank": args.lora_rank,
            },
            auto_reload=False,
        )
        from mlx.utils import tree_flatten

        n_trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
        n_total = sum(p.size for _, p in tree_flatten(model.parameters()))
        print(f"LoRA: {n_trainable:,} trainable / {n_total:,} total "
              f"({n_trainable / n_total * 100:.3f}%)")

    optimizer = optim.Adam(learning_rate=1e-5)
    compile_flag = args.compile if hasattr(args, "compile") else False
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optimizer,
        compile_training=compile_flag,
        profile=True,
    )

    prompt_pool_size = args.prompt_pool_size
    if prompt_pool_size is None:
        prompt_pool_size = args.episodes
    prompt_pool_size = max(1, min(prompt_pool_size, args.episodes))
    prompts = [
        f"Explain concept number {i % prompt_pool_size} briefly."
        for i in range(args.episodes)
    ]

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
        # Compute logprobs — required by _prepare_batch_from_buffer downstream.
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tokens = mx.argmax(logits[0, -1:], axis=-1)
        token_logprob = log_probs[0, -1, tokens[0]]
        return tokens, {"logprob": token_logprob}

    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=policy_fn, strategy=strategy, max_steps=args.episodes)

    all_metrics: List[Dict[str, float]] = []
    prompt_reuse_samples: List[Dict[str, float]] = []
    for step in range(args.warmup + args.steps):
        # Clear the runner's persistent buffer so episodes don't accumulate
        # across iterations — each step should measure a fixed-size workload.
        runner.buffer.clear()

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

        # Use the GRPO data pipeline (_pack_episodes) which produces 2D
        # obs/act with prompt_lengths — required for the batched logprob
        # extraction path that is mx.compile-safe.
        batch_data = grpo._pack_episodes(buffer.episodes)
        if "prompt_lengths" in batch_data and batch_data["obs"].ndim == 2:
            prompt_reuse = compute_prompt_reuse_stats(
                batch_data["obs"],
                batch_data["prompt_lengths"],
                batch_data["episode_lengths"],
            )
        else:
            prompt_reuse = {
                "num_episodes": float(len(batch_data["episode_lengths"])),
                "repeat_rate": 0.0,
                "prompt_token_reduction_upper_bound": 0.0,
            }

        # Training
        t0 = time.perf_counter()
        m = trainer.train(batch_data)
        t_training = time.perf_counter() - t0

        trainer.reset_timer()

        if step >= args.warmup:
            outer_times["rollout"].append(t_rollout)
            outer_times["buffer_transfer"].append(t_buffer)
            outer_times["training"].append(t_training)
            all_metrics.append(m)
            prompt_reuse_samples.append(prompt_reuse)

    mode = "full_pipeline_lora" if args.lora else "full_pipeline"
    result = _aggregate(all_metrics, mode, args, prompt_reuse_samples)
    # Add outer-loop timing
    for phase, times in outer_times.items():
        if times:
            result["outer_phases"][phase] = {
                "mean_s": sum(times) / len(times),
                "min_s": min(times),
                "max_s": max(times),
            }
    if args.lora:
        result["lora"] = {
            "rank": args.lora_rank,
            "layers": args.lora_layers,
            "trainable_params": n_trainable,
            "total_params": n_total,
            "trainable_pct": n_trainable / n_total * 100,
        }
    return result


# ---------------------------------------------------------------------------
# Aggregation & display
# ---------------------------------------------------------------------------

def _aggregate(
    all_metrics: List[Dict[str, float]],
    mode: str,
    args: argparse.Namespace,
    prompt_reuse_samples: Optional[List[Dict[str, float]]] = None,
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
    if prompt_reuse_samples:
        summary["prompt_reuse"] = _aggregate_prompt_reuse(prompt_reuse_samples)
    return summary


def _aggregate_prompt_reuse(samples: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate prompt-reuse metrics collected across benchmark steps."""
    out: Dict[str, float] = {"samples": float(len(samples))}
    keys = sorted(set().union(*(s.keys() for s in samples)))
    for key in keys:
        vals = [float(s[key]) for s in samples if key in s]
        if not vals:
            continue
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_max"] = max(vals)
    return out


def _build_summary_md(result: Dict[str, Any]) -> str:
    """Build a markdown summary string from benchmark results."""
    lines: List[str] = []
    mode = result["mode"]
    lines.append(f"## Amdahl Benchmark — {mode}")
    lines.append(f"Episodes: {result['episodes']}, length: {result['episode_length']}, "
                 f"steps: {result['steps']} (warmup: {result['warmup']})")
    lines.append("")

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
        lines.append(_format_table(rows, ["Phase", "Mean (s)", "% of total"]))

    # Outer phases (full pipeline only)
    if result.get("outer_phases"):
        lines.append("")
        lines.append("### Outer loop phases")
        lines.append("")
        rows = []
        for phase, stats in result["outer_phases"].items():
            rows.append([phase, f"{stats['mean_s']:.5f}", f"{stats['min_s']:.5f}", f"{stats['max_s']:.5f}"])
        lines.append(_format_table(rows, ["Phase", "Mean (s)", "Min (s)", "Max (s)"]))

    prompt_reuse = result.get("prompt_reuse")
    if prompt_reuse:
        lines.append("")
        lines.append("### Prompt reuse opportunity")
        lines.append("")
        rows = [
            [
                "repeat_rate",
                f"{prompt_reuse.get('repeat_rate_mean', 0.0) * 100:.1f}%",
                f"{prompt_reuse.get('repeat_rate_max', 0.0) * 100:.1f}%",
            ],
            [
                "prompt_token_reduction_upper_bound",
                f"{prompt_reuse.get('prompt_token_reduction_upper_bound_mean', 0.0) * 100:.1f}%",
                f"{prompt_reuse.get('prompt_token_reduction_upper_bound_max', 0.0) * 100:.1f}%",
            ],
            [
                "end_to_end_token_reduction_upper_bound",
                f"{prompt_reuse.get('end_to_end_token_reduction_upper_bound_mean', 0.0) * 100:.1f}%",
                f"{prompt_reuse.get('end_to_end_token_reduction_upper_bound_max', 0.0) * 100:.1f}%",
            ],
            [
                "unique_prompts",
                f"{prompt_reuse.get('unique_prompts_mean', 0.0):.2f}",
                f"{prompt_reuse.get('unique_prompts_max', 0.0):.2f}",
            ],
            [
                "max_group_size",
                f"{prompt_reuse.get('max_group_size_mean', 0.0):.2f}",
                f"{prompt_reuse.get('max_group_size_max', 0.0):.2f}",
            ],
        ]
        lines.append(_format_table(rows, ["Metric", "Mean", "Max"]))

    lines.append("")
    return "\n".join(lines)


def _print_summary(result: Dict[str, Any]) -> None:
    """Print a markdown summary table to stdout."""
    print()
    print(_build_summary_md(result))


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
    parser.add_argument("--prompt-pool-size", type=int, default=None,
                        help="Unique prompt pool size. Set below --episodes to "
                             "introduce repeated prompts and estimate cache opportunity.")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model ID for full-pipeline mode "
                             "(e.g. Qwen/Qwen3-0.6B). Omit for synthetic mode.")
    parser.add_argument("--lora", action="store_true",
                        help="Apply LoRA adapters (full-pipeline mode only).")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-layers", type=int, default=8,
                        help="Number of layers to apply LoRA to (default: 8)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable mx.compile for the training step "
                             "(full-pipeline mode with --model only; "
                             "ignored in synthetic mode).")
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

    json_path = _save_json(result, args.output_dir, mode)

    # Write markdown summary alongside the JSON
    md_path = Path(json_path).with_suffix(".md")
    md_path.write_text(_build_summary_md(result))

    print(f"Results saved to: {json_path}")
    print(f"Summary saved to: {md_path}")


if __name__ == "__main__":
    main()
