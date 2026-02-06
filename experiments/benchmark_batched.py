#!/usr/bin/env python3
"""
Benchmark: Sequential vs Batched Text Generation.

Compares wall-clock time and tokens/sec for sequential single-prompt
generation (N calls to generate_tokens) against batched generation
(1 call to batch_generate_tokens) for various batch sizes.

Usage:
    uv run python experiments/benchmark_batched.py
    uv run python experiments/benchmark_batched.py --model Qwen/Qwen3-0.6B --max-tokens 128
"""

import argparse
import time
from dataclasses import dataclass

import mlx.core as mx

from textpolicy.generation.mlx_generation import (
    batch_generate_tokens,
    generate_tokens,
    load_model,
)
from textpolicy.tasks.countdown import (
    format_countdown_prompt,
    generate_countdown_problems,
)


@dataclass
class BenchmarkConfig:
    model_id: str = "Qwen/Qwen3-0.6B"
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    num_problems: int = 8
    batch_sizes: tuple = (1, 2, 4, 8)
    seed: int = 42


def run_sequential(model, tokenizer, prompt_token_lists, config):
    """Run N sequential generate_tokens calls."""
    results = []
    for prompt_list in prompt_token_lists:
        prompt_tokens = mx.array(prompt_list, dtype=mx.int32)
        resp, info = generate_tokens(
            model, tokenizer, prompt_tokens,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        results.append((resp, info))
    return results


def run_batched(model, tokenizer, prompt_token_lists, config):
    """Run 1 batch_generate_tokens call for N prompts."""
    return batch_generate_tokens(
        model, tokenizer, prompt_token_lists,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )


def validate_results(results, label):
    """Check logprob validity: negative, finite, no NaN."""
    total_tokens = 0
    for resp, info in results:
        lp = info["logprob"]
        n = len(lp)
        total_tokens += n
        if n > 0:
            mx.eval(lp)
            if mx.any(mx.isnan(lp)).item():
                print(f"  WARNING [{label}]: NaN logprobs detected")
                return False
            if mx.any(mx.isinf(lp)).item():
                print(f"  WARNING [{label}]: Inf logprobs detected")
                return False
            if mx.any(lp > 0).item():
                print(f"  WARNING [{label}]: Positive logprobs detected")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark sequential vs batched generation")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-problems", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = BenchmarkConfig(
        model_id=args.model,
        max_tokens=args.max_tokens,
        num_problems=args.num_problems,
        seed=args.seed,
    )

    # Load model
    print(f"Loading model: {config.model_id}")
    model, tokenizer = load_model(config.model_id)

    # Generate countdown prompts
    problems = generate_countdown_problems(config.num_problems, seed=config.seed)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    # Tokenize all prompts
    all_token_lists = [tokenizer.encode(p) for p in prompts]

    print(f"\nBenchmark: {config.num_problems} prompts, max_tokens={config.max_tokens}")
    print(f"Prompt lengths: {[len(t) for t in all_token_lists]}")
    print("=" * 70)

    for N in config.batch_sizes:
        if N > len(all_token_lists):
            continue

        subset = all_token_lists[:N]
        print(f"\n--- N={N} sequences ---")

        # Warmup (1 token each to prime caches)
        _ = batch_generate_tokens(model, tokenizer, subset[:1], max_tokens=1)

        # Sequential timing
        t0 = time.perf_counter()
        seq_results = run_sequential(model, tokenizer, subset, config)
        t_seq = time.perf_counter() - t0
        seq_tokens = sum(len(r[0]) for r in seq_results)
        seq_valid = validate_results(seq_results, "sequential")

        # Batched timing
        t0 = time.perf_counter()
        bat_results = run_batched(model, tokenizer, subset, config)
        t_bat = time.perf_counter() - t0
        bat_tokens = sum(len(r[0]) for r in bat_results)
        bat_valid = validate_results(bat_results, "batched")

        # Report
        speedup = t_seq / t_bat if t_bat > 0 else float("inf")
        print(f"  Sequential: {t_seq:.2f}s, {seq_tokens} tokens, "
              f"{seq_tokens / t_seq:.1f} tok/s, valid={seq_valid}")
        print(f"  Batched:    {t_bat:.2f}s, {bat_tokens} tokens, "
              f"{bat_tokens / t_bat:.1f} tok/s, valid={bat_valid}")
        print(f"  Speedup:    {speedup:.2f}x")

    print("\n" + "=" * 70)
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
