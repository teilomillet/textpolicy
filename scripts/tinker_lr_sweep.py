#!/usr/bin/env python3
"""
Learning rate sweep for Tinker LoRA training.

Runs short probes (20 steps) at multiple learning rates for both algorithms,
following the methodology of Lee et al. (2026) "Learning Rate Matters."

After the sweep, prints the best lr for each algorithm and the recommended
campaign command.

Usage:
    # Run the sweep (~2.5 hours):
    nohup uv run python scripts/tinker_lr_sweep.py \
        --sweep-root results/lr_sweep \
        > results/lr_sweep.log 2>&1 &

    # Check progress:
    tail -20 results/lr_sweep.log
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Load .env file if it exists (for TINKER_API_KEY)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    with _env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Sweep configuration following Lee et al. (2026)
# They sweep lr from 1e-6 to 1e-3 with 16 points.
# We use 4 points spanning the most relevant range for Qwen + math.
DEFAULT_LR_VALUES = [5e-5, 2e-4, 5e-4, 1e-3]
DEFAULT_ALGORITHMS = ["grpo", "full"]


def _run_probe(
    algorithm: str,
    lr: float,
    log_dir: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run a short training probe and return metrics summary."""
    cmd = [
        sys.executable,
        "-m", "textpolicy.tinker.train_math",
        "--algorithm", algorithm,
        "--model", args.model,
        "--max-steps", str(args.probe_steps),
        "--batch-size", str(args.batch_size),
        "--group-size", str(args.group_size),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--lr", str(lr),
        "--lora-rank", str(args.lora_rank),
        "--save-every", "0",  # No checkpoints for sweep
        "--log-dir", log_dir,
        "--sepa-steps", str(args.sepa_steps),
        "--sepa-schedule", args.sepa_schedule,
        "--sepa-delay-steps", str(args.sepa_delay_steps),
        "--sepa-correct-rate-gate", str(args.sepa_correct_rate_gate),
    ]

    if args.base_url:
        cmd.extend(["--base-url", args.base_url])

    print(f"\n{'─'*60}")
    print(f"  Probe: algorithm={algorithm}, lr={lr}")
    print(f"  Log dir: {log_dir}")
    print(f"{'─'*60}")

    proc = subprocess.run(cmd, cwd=str(_repo_root()), check=False)

    # Read metrics
    metrics_path = Path(log_dir) / "metrics.jsonl"
    metrics = []
    if metrics_path.exists():
        with metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))

    if not metrics:
        return {
            "algorithm": algorithm,
            "lr": lr,
            "return_code": proc.returncode,
            "training_steps": 0,
            "final_correct_rate": 0.0,
            "final_running_correct_rate": 0.0,
            "mean_loss": 0.0,
            "total_datums": 0,
            "status": "failed",
        }

    losses = [m["loss"] for m in metrics]
    total_datums = sum(m["num_datums"] for m in metrics)
    final = metrics[-1]

    return {
        "algorithm": algorithm,
        "lr": lr,
        "return_code": proc.returncode,
        "training_steps": len(metrics),
        "final_correct_rate": final.get("correct_rate", 0.0),
        "final_running_correct_rate": final.get("running_correct_rate", 0.0),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "total_datums": total_datums,
        "status": "ok",
    }


def _print_sweep_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted table of sweep results."""
    print(f"\n{'='*80}")
    print(f"  LEARNING RATE SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Algorithm':>10} {'LR':>10} {'Steps':>6} {'Datums':>7} "
          f"{'Correct%':>9} {'Running%':>9} {'MeanLoss':>10} {'Status':>8}")
    print(f"{'─'*80}")

    for r in results:
        print(f"{r['algorithm']:>10} {r['lr']:>10.0e} {r['training_steps']:>6} "
              f"{r['total_datums']:>7} "
              f"{r['final_correct_rate']*100:>8.1f}% "
              f"{r['final_running_correct_rate']*100:>8.1f}% "
              f"{r['mean_loss']:>10.4f} "
              f"{r['status']:>8}")

    print(f"{'='*80}")


def _find_best_lr(results: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """Find best lr for each algorithm by running correct rate."""
    best: Dict[str, Tuple[float, float]] = {}

    for algo in DEFAULT_ALGORITHMS:
        algo_results = [r for r in results if r["algorithm"] == algo and r["status"] == "ok"]
        if not algo_results:
            continue
        # Primary metric: running_correct_rate (accounts for full trajectory)
        best_result = max(algo_results, key=lambda r: r["final_running_correct_rate"])
        best[algo] = (best_result["lr"], best_result["final_running_correct_rate"])

    return best


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Learning rate sweep for Tinker LoRA training"
    )

    parser.add_argument("--sweep-root", default=None,
                        help="Output directory (default: results/lr_sweep_<timestamp>)")
    parser.add_argument("--probe-steps", type=int, default=20,
                        help="Steps per probe run")

    # Shared training settings
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank (default: 64, per Lee et al. 2026)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Full pipeline hyperparameters
    parser.add_argument("--sepa-steps", type=int, default=100)
    parser.add_argument("--sepa-schedule", default="linear")
    parser.add_argument("--sepa-delay-steps", type=int, default=10)
    parser.add_argument("--sepa-correct-rate-gate", type=float, default=0.1)

    # Sweep parameters
    parser.add_argument("--lr-values", type=str, default=None,
                        help="Comma-separated lr values (default: 5e-5,2e-4,5e-4,1e-3)")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated algorithms (default: grpo,full)")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    sweep_root = (
        Path(args.sweep_root).expanduser().resolve()
        if args.sweep_root
        else _repo_root() / "results" / f"lr_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)

    lr_values = (
        [float(x) for x in args.lr_values.split(",")]
        if args.lr_values
        else DEFAULT_LR_VALUES
    )
    algorithms = (
        args.algorithms.split(",")
        if args.algorithms
        else DEFAULT_ALGORITHMS
    )

    print(f"Learning Rate Sweep")
    print(f"  Root: {sweep_root}")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Probe steps: {args.probe_steps}")
    print(f"  Batch: {args.batch_size} prompts × {args.group_size} completions")
    print(f"  LR values: {lr_values}")
    print(f"  Algorithms: {algorithms}")
    print(f"  Estimated time: ~{len(lr_values) * len(algorithms) * args.probe_steps * 50 / 60:.0f} min")

    results: List[Dict[str, Any]] = []

    for algo in algorithms:
        for lr in lr_values:
            lr_label = f"{lr:.0e}".replace("+", "")
            log_dir = str(sweep_root / f"{algo}_lr{lr_label}")

            result = _run_probe(algo, lr, log_dir, args)
            results.append(result)

            print(f"  → {algo} lr={lr:.0e}: "
                  f"correct={result['final_running_correct_rate']*100:.1f}%, "
                  f"loss={result['mean_loss']:.4f}, "
                  f"datums={result['total_datums']}")

    # Print results table
    _print_sweep_table(results)

    # Find best lr for each algorithm
    best = _find_best_lr(results)
    print(f"\n{'='*80}")
    print(f"  RECOMMENDED LEARNING RATES")
    print(f"{'='*80}")
    for algo, (lr, rate) in best.items():
        print(f"  {algo}: lr={lr:.0e} (running correct rate: {rate*100:.1f}%)")

    # Print recommended campaign command
    if len(best) == 2 and "grpo" in best and "full" in best:
        grpo_lr, _ = best["grpo"]
        full_lr, _ = best["full"]

        if grpo_lr == full_lr:
            # Same lr for both — can use the campaign script
            print(f"\n  Recommended campaign command (same lr for both arms):")
            print(f"  nohup uv run python scripts/tinker_campaign.py \\")
            print(f"      --max-steps 100 --batch-size {args.batch_size} \\")
            print(f"      --group-size {args.group_size} --lr {grpo_lr} \\")
            print(f"      --lora-rank {args.lora_rank} \\")
            print(f"      --sepa-steps {args.sepa_steps} --sepa-delay-steps {args.sepa_delay_steps} \\")
            print(f"      --campaign-root results/tinker_campaign_v3_tuned \\")
            print(f"      --save-every 25 --execute \\")
            print(f"      > results/tinker_campaign_v3_tuned.log 2>&1 &")
        else:
            # Different lr — need separate runs
            print(f"\n  Different optimal lr per algorithm. Run separately:")
            print(f"  # Baseline arm:")
            print(f"  nohup uv run python -m textpolicy.tinker.train_math \\")
            print(f"      --algorithm grpo --lr {grpo_lr} --lora-rank {args.lora_rank} \\")
            print(f"      --max-steps 100 --batch-size {args.batch_size} --group-size {args.group_size} \\")
            print(f"      --log-dir results/tinker_campaign_v3_tuned/baseline \\")
            print(f"      > results/tinker_campaign_v3_baseline.log 2>&1 &")
            print(f"  # Candidate arm:")
            print(f"  nohup uv run python -m textpolicy.tinker.train_math \\")
            print(f"      --algorithm full --lr {full_lr} --lora-rank {args.lora_rank} \\")
            print(f"      --max-steps 100 --batch-size {args.batch_size} --group-size {args.group_size} \\")
            print(f"      --sepa-steps {args.sepa_steps} --sepa-delay-steps {args.sepa_delay_steps} \\")
            print(f"      --log-dir results/tinker_campaign_v3_tuned/candidate \\")
            print(f"      > results/tinker_campaign_v3_candidate.log 2>&1 &")

    print(f"\n{'='*80}")

    # Save sweep results
    sweep_json = sweep_root / "sweep_results.json"
    sweep_json.write_text(json.dumps({
        "sweep_root": str(sweep_root),
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "lora_rank": args.lora_rank,
        "probe_steps": args.probe_steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "lr_values": lr_values,
        "algorithms": algorithms,
        "results": results,
        "best_lr": {algo: {"lr": lr, "correct_rate": rate} for algo, (lr, rate) in best.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  Results saved to: {sweep_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
