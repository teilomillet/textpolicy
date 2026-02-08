#!/usr/bin/env python3
"""
Sequence-packing A/B benchmark for textpolicy.

Compares:
  - ``_pack_episodes(..., sort_by_length=False)`` (baseline)
  - ``_pack_episodes(..., sort_by_length=True)``  (length-sorted)

Both paths are measured end-to-end through:
  1) episode packing
  2) GRPO batched logprob extraction

Usage:
    uv run python scripts/benchmark_sequence_packing.py
    uv run python scripts/benchmark_sequence_packing.py --dim 128 --vocab-size 1024
    uv run python scripts/benchmark_sequence_packing.py --micro-batch-size none
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.algorithms.grpo import _pack_episodes
from textpolicy.training import Trainer


class TinyLM(nn.Module):
    """Small LM-shaped model for repeatable local benchmarking."""

    def __init__(self, vocab_size: int = 2048, dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.embed(x))


def _parse_optional_int(value: str) -> Optional[int]:
    """Parse integer value or 'none'."""
    if value.lower() == "none":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "expected a positive integer or 'none'"
        )
    return parsed


def _make_episodes(
    n: int,
    prompt_min: int,
    prompt_max: int,
    response_min: int,
    response_max: int,
    vocab_size: int,
    seed: int,
) -> List[Any]:
    """Generate synthetic episodes with variable prompt/response lengths."""
    rng = random.Random(seed)
    episodes = []
    for _ in range(n):
        prompt_len = rng.randint(prompt_min, prompt_max)
        response_len = rng.randint(response_min, response_max)
        prompt = [rng.randint(1, vocab_size - 1) for _ in range(prompt_len)]
        response = [rng.randint(1, vocab_size - 1) for _ in range(response_len)]
        episodes.append(
            SimpleNamespace(
                obs=[prompt],
                act=[response],
                rew=[float(rng.random())],
                logprob=[-0.1] * response_len,
            )
        )
    return episodes


def _extract_segments(flat_values: mx.array, lengths: List[int]) -> List[mx.array]:
    """Split flat token vector into per-episode segments using lengths."""
    out = []
    offset = 0
    for length in lengths:
        out.append(flat_values[offset : offset + length])
        offset += length
    return out


def _check_correctness(
    trainer: Trainer,
    vocab_size: int,
    n_episodes: int = 24,
) -> Dict[str, float]:
    """Verify sorted vs unsorted paths are numerically equivalent."""
    # Use unique response lengths to map sorted segments back to baseline.
    episodes = []
    for i in range(1, n_episodes + 1):
        prompt_len = 32 + i
        response_len = i
        prompt = [((i * 13) + j) % (vocab_size - 1) + 1 for j in range(prompt_len)]
        response = [((i * 17) + j) % (vocab_size - 1) + 1 for j in range(response_len)]
        episodes.append(
            SimpleNamespace(obs=[prompt], act=[response], rew=[1.0], logprob=[-0.1] * response_len)
        )

    batch_unsorted = _pack_episodes(episodes, sort_by_length=False)
    batch_sorted = _pack_episodes(episodes, sort_by_length=True)

    logprobs_unsorted = trainer._extract_grpo_logprobs(
        batch_unsorted["obs"],
        batch_unsorted["act"],
        batch_unsorted["logprob"],
        batch_unsorted["episode_lengths"],
        batch_unsorted["prompt_lengths"],
    )
    logprobs_sorted = trainer._extract_grpo_logprobs(
        batch_sorted["obs"],
        batch_sorted["act"],
        batch_sorted["logprob"],
        batch_sorted["episode_lengths"],
        batch_sorted["prompt_lengths"],
    )
    mx.eval(logprobs_unsorted, logprobs_sorted)

    unsorted_segments = _extract_segments(logprobs_unsorted, batch_unsorted["episode_lengths"])
    sorted_segments = _extract_segments(logprobs_sorted, batch_sorted["episode_lengths"])

    unsorted_by_len = {
        length: segment
        for length, segment in zip(batch_unsorted["episode_lengths"], unsorted_segments)
    }
    sorted_by_len = {
        length: segment
        for length, segment in zip(batch_sorted["episode_lengths"], sorted_segments)
    }

    max_abs_diff = 0.0
    for length in sorted(unsorted_by_len.keys()):
        lhs = unsorted_by_len[length]
        rhs = sorted_by_len[length]
        mx.eval(lhs, rhs)
        if length > 0:
            diff = float(mx.max(mx.abs(lhs - rhs)).item())
            max_abs_diff = max(max_abs_diff, diff)

    sum_diff = abs(
        float(mx.sum(logprobs_unsorted).item()) - float(mx.sum(logprobs_sorted).item())
    )
    return {
        "max_abs_diff": max_abs_diff,
        "sum_diff": sum_diff,
    }


def _measure_variant(
    trainer: Trainer,
    episodes: List[Any],
    *,
    sort_by_length: bool,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    """Measure mean wall time for pack/extract phases."""
    pack_times: List[float] = []
    extract_times: List[float] = []
    total_times: List[float] = []

    for _ in range(warmup):
        batch = _pack_episodes(episodes, sort_by_length=sort_by_length)
        mx.eval(batch["obs"], batch["act"], batch["logprob"], batch["rewards"])
        out = trainer._extract_grpo_logprobs(
            batch["obs"],
            batch["act"],
            batch["logprob"],
            batch["episode_lengths"],
            batch["prompt_lengths"],
        )
        mx.eval(out)

    for _ in range(iters):
        t0 = time.perf_counter()
        batch = _pack_episodes(episodes, sort_by_length=sort_by_length)
        mx.eval(batch["obs"], batch["act"], batch["logprob"], batch["rewards"])
        t1 = time.perf_counter()

        out = trainer._extract_grpo_logprobs(
            batch["obs"],
            batch["act"],
            batch["logprob"],
            batch["episode_lengths"],
            batch["prompt_lengths"],
        )
        mx.eval(out)
        t2 = time.perf_counter()

        pack_times.append(t1 - t0)
        extract_times.append(t2 - t1)
        total_times.append(t2 - t0)

    def _p95(values: List[float]) -> float:
        if len(values) < 2:
            return values[0]
        return statistics.quantiles(values, n=20)[18]

    return {
        "pack_ms": statistics.mean(pack_times) * 1000.0,
        "extract_ms": statistics.mean(extract_times) * 1000.0,
        "total_ms": statistics.mean(total_times) * 1000.0,
        "p95_total_ms": _p95(total_times) * 1000.0,
    }


def _format_table(rows: List[List[str]], header: List[str]) -> str:
    """Render markdown table."""
    widths = [max(len(h), *(len(r[i]) for r in rows), 0) for i, h in enumerate(header)]

    def _row(cols: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([_row(header), sep] + [_row(r) for r in rows])


def _save_json(result: Dict[str, Any], output_dir: str) -> str:
    """Save benchmark result JSON with timestamped filename."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(output_dir) / f"sequence_packing_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return str(path)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the benchmark and return structured results."""
    if args.response_min < 0:
        raise ValueError("response-min must be >= 0")
    if args.prompt_min <= 0:
        raise ValueError("prompt-min must be >= 1")
    if args.prompt_min > args.prompt_max:
        raise ValueError("prompt-min must be <= prompt-max")
    if args.response_min > args.response_max:
        raise ValueError("response-min must be <= response-max")
    if args.vocab_size < 4:
        raise ValueError("vocab-size must be >= 4")

    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    if not seeds:
        raise ValueError("No seeds provided.")

    mx.random.seed(args.model_seed)
    model = TinyLM(vocab_size=args.vocab_size, dim=args.dim)
    mx.eval(model.parameters())

    trainer = Trainer(
        model=model,
        loss_fn=grpo.policy_loss,
        optimizer=optim.Adam(learning_rate=1e-3),
        advantage_fn=grpo.compute_advantages,
        compile_training=args.compile_training,
        micro_batch_size=args.micro_batch_size,
    )

    correctness = None
    if not args.skip_correctness_check:
        correctness = _check_correctness(trainer, args.vocab_size)

    seed_rows = []
    speedups = []
    for seed in seeds:
        episodes = _make_episodes(
            n=args.episodes,
            prompt_min=args.prompt_min,
            prompt_max=args.prompt_max,
            response_min=args.response_min,
            response_max=args.response_max,
            vocab_size=args.vocab_size,
            seed=seed,
        )
        unsorted = _measure_variant(
            trainer,
            episodes,
            sort_by_length=False,
            warmup=args.warmup,
            iters=args.iters,
        )
        sorted_ = _measure_variant(
            trainer,
            episodes,
            sort_by_length=True,
            warmup=args.warmup,
            iters=args.iters,
        )
        speedup_pct = ((unsorted["total_ms"] - sorted_["total_ms"]) / unsorted["total_ms"]) * 100.0
        speedups.append(speedup_pct)
        seed_rows.append(
            {
                "seed": seed,
                "unsorted": unsorted,
                "sorted": sorted_,
                "speedup_pct": speedup_pct,
            }
        )

    summary = {
        "avg_speedup_pct": statistics.mean(speedups),
        "min_speedup_pct": min(speedups),
        "max_speedup_pct": max(speedups),
    }

    return {
        "config": {
            "episodes": args.episodes,
            "prompt_min": args.prompt_min,
            "prompt_max": args.prompt_max,
            "response_min": args.response_min,
            "response_max": args.response_max,
            "vocab_size": args.vocab_size,
            "dim": args.dim,
            "micro_batch_size": args.micro_batch_size,
            "compile_training": args.compile_training,
            "warmup": args.warmup,
            "iters": args.iters,
            "seeds": seeds,
            "model_seed": args.model_seed,
        },
        "correctness": correctness,
        "per_seed": seed_rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sorted vs unsorted sequence packing in textpolicy."
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--prompt-min", type=int, default=32)
    parser.add_argument("--prompt-max", type=int, default=384)
    parser.add_argument("--response-min", type=int, default=8)
    parser.add_argument("--response-max", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument(
        "--micro-batch-size",
        type=_parse_optional_int,
        default=8,
        help="Positive integer or 'none' (default: 8).",
    )
    parser.add_argument("--compile-training", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument(
        "--seeds",
        type=str,
        default="11,22,33,44,55,66",
        help="Comma-separated seed list.",
    )
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--skip-correctness-check", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="artefacts/perf")
    args = parser.parse_args()

    result = run(args)

    print()
    print("## Sequence Packing Benchmark")
    print(
        f"episodes={result['config']['episodes']}, "
        f"micro_batch_size={result['config']['micro_batch_size']}, "
        f"dim={result['config']['dim']}, vocab={result['config']['vocab_size']}, "
        f"warmup={result['config']['warmup']}, iters={result['config']['iters']}"
    )
    print()

    if result["correctness"] is not None:
        corr = result["correctness"]
        print(
            "Correctness check: "
            f"max_abs_diff={corr['max_abs_diff']:.3e}, "
            f"sum_diff={corr['sum_diff']:.3e}"
        )
        print()

    rows = []
    for row in result["per_seed"]:
        rows.append(
            [
                str(row["seed"]),
                f"{row['unsorted']['total_ms']:.3f}",
                f"{row['sorted']['total_ms']:.3f}",
                f"{row['speedup_pct']:+.3f}%",
            ]
        )
    print(_format_table(rows, ["Seed", "Unsorted Total (ms)", "Sorted Total (ms)", "Speedup"]))
    print()
    print(
        "Summary: "
        f"avg={result['summary']['avg_speedup_pct']:+.3f}% "
        f"min={result['summary']['min_speedup_pct']:+.3f}% "
        f"max={result['summary']['max_speedup_pct']:+.3f}%"
    )

    json_path = _save_json(result, args.output_dir)
    print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
