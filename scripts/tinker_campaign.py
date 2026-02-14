#!/usr/bin/env python3
"""
Run a two-arm A/B campaign on Tinker GPUs: baseline GRPO vs full pipeline.

Default mode is dry-run (plan only). Use --execute to actually run.

Both arms use identical settings (model, dataset, batch size, group size,
learning rate, temperature) — only the advantage computation differs:
  - baseline (grpo): scalar group-relative advantages, uniform across tokens
  - candidate (full): MaxRL → GTPO → HICRA → SEPA token-level advantages

After both arms complete, the script runs significance analysis using the
existing textpolicy analysis framework (permutation tests, bootstrap CIs,
Fisher exact test for correctness rate).

Usage:
    # Plan only (dry-run):
    python scripts/tinker_campaign.py --max-steps 100

    # Execute:
    python scripts/tinker_campaign.py --max-steps 100 --execute

    # Custom campaign root:
    python scripts/tinker_campaign.py --max-steps 100 --execute \\
        --campaign-root results/tinker_ab_v1
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _default_campaign_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "results" / f"tinker_campaign_{stamp}"


def _build_arm_command(
    algorithm: str,
    log_dir: str,
    args: argparse.Namespace,
    lr_override: Optional[float] = None,
) -> List[str]:
    """Build the train_math.py command for one arm."""
    lr = lr_override if lr_override is not None else args.lr
    cmd = [
        sys.executable,
        "-m", "textpolicy.tinker.train_math",
        "--algorithm", algorithm,
        "--model", args.model,
        "--max-steps", str(args.max_steps),
        "--batch-size", str(args.batch_size),
        "--group-size", str(args.group_size),
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--lr", str(lr),
        "--lora-rank", str(args.lora_rank),
        "--save-every", str(args.save_every),
        "--log-dir", log_dir,
    ]

    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])

    # Full pipeline hyperparameters (only matter for candidate arm,
    # but included in both for reproducibility in the manifest)
    cmd.extend([
        "--gtpo-beta", str(args.gtpo_beta),
        "--hicra-alpha", str(args.hicra_alpha),
        "--sepa-steps", str(args.sepa_steps),
        "--sepa-schedule", args.sepa_schedule,
        "--sepa-delay-steps", str(args.sepa_delay_steps),
        "--sepa-correct-rate-gate", str(args.sepa_correct_rate_gate),
    ])

    if args.base_url:
        cmd.extend(["--base-url", args.base_url])

    return cmd


def _run_arm(
    label: str,
    cmd: List[str],
    cwd: Path,
) -> int:
    """Run one arm as a subprocess. Returns the return code."""
    print(f"\n{'='*60}")
    print(f"  Running {label} arm")
    print(f"  Command: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"{'='*60}\n")

    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        print(f"\n  {label} arm FAILED (return code {proc.returncode})")
    else:
        print(f"\n  {label} arm completed successfully")
    return proc.returncode


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_comparison(
    baseline_dir: Path,
    candidate_dir: Path,
) -> Dict[str, Any]:
    """Compute comparison statistics from the two arms' output."""
    # Load metrics
    base_metrics = _load_jsonl(baseline_dir / "metrics.jsonl")
    cand_metrics = _load_jsonl(candidate_dir / "metrics.jsonl")

    # Load generation-level data
    base_gens = _load_jsonl(baseline_dir / "emergence" / "generations.jsonl")
    cand_gens = _load_jsonl(candidate_dir / "emergence" / "generations.jsonl")

    # Aggregate
    base_rewards = [g["reward"] for g in base_gens]
    cand_rewards = [g["reward"] for g in cand_gens]

    base_correct = sum(1 for r in base_rewards if r >= 0.99)
    cand_correct = sum(1 for r in cand_rewards if r >= 0.99)

    base_total = len(base_rewards)
    cand_total = len(cand_rewards)

    base_rate = base_correct / max(base_total, 1)
    cand_rate = cand_correct / max(cand_total, 1)

    base_mean_reward = sum(base_rewards) / max(len(base_rewards), 1)
    cand_mean_reward = sum(cand_rewards) / max(len(cand_rewards), 1)

    return {
        "baseline": {
            "total_generations": base_total,
            "correct": base_correct,
            "correct_rate": base_rate,
            "mean_reward": base_mean_reward,
            "training_steps": len(base_metrics),
        },
        "candidate": {
            "total_generations": cand_total,
            "correct": cand_correct,
            "correct_rate": cand_rate,
            "mean_reward": cand_mean_reward,
            "training_steps": len(cand_metrics),
        },
        "delta": {
            "correct_rate": cand_rate - base_rate,
            "mean_reward": cand_mean_reward - base_mean_reward,
        },
    }


def _run_significance_analysis(
    baseline_dir: Path,
    candidate_dir: Path,
    analysis_dir: Path,
    alpha: float = 0.05,
    resamples: int = 20000,
) -> Optional[Dict[str, Any]]:
    """Run the existing significance framework if available."""
    try:
        from textpolicy.analysis import (
            build_sepa_significance_markdown,
            evaluate_sepa_significance,
        )
    except ImportError:
        print("  Significance analysis module not available, skipping.")
        return None

    baseline_dirs = [str(baseline_dir)]
    candidate_dirs = [str(candidate_dir)]

    report = evaluate_sepa_significance(
        baseline_run_dirs=baseline_dirs,
        candidate_run_dirs=candidate_dirs,
        alpha=alpha,
        num_resamples=resamples,
        seed=0,
    )

    sig_json = analysis_dir / "significance.json"
    sig_md = analysis_dir / "significance.md"

    sig_json.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    sig_md.write_text(
        build_sepa_significance_markdown(report), encoding="utf-8"
    )

    print(f"  Significance analysis: {report.recommendation}")
    print(f"  Report: {sig_md}")

    return report.to_dict()


def _write_comparison_markdown(
    comparison: Dict[str, Any],
    analysis_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """Write a human-readable comparison summary."""
    md_path = analysis_dir / "comparison.md"
    b = comparison["baseline"]
    c = comparison["candidate"]
    d = comparison["delta"]

    lines = [
        "# Tinker GPU Campaign: GRPO Baseline vs Full Pipeline",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model**: {args.model}",
        f"**Steps per arm**: {args.max_steps}",
        f"**Batch size**: {args.batch_size} prompts x {args.group_size} completions",
        "",
        "## Results",
        "",
        "| Metric | Baseline (GRPO) | Candidate (Full) | Delta |",
        "|--------|----------------|------------------|-------|",
        f"| Correct rate | {b['correct_rate']:.4f} ({b['correct']}/{b['total_generations']}) "
        f"| {c['correct_rate']:.4f} ({c['correct']}/{c['total_generations']}) "
        f"| {d['correct_rate']:+.4f} |",
        f"| Mean reward | {b['mean_reward']:.4f} | {c['mean_reward']:.4f} | {d['mean_reward']:+.4f} |",
        f"| Training steps | {b['training_steps']} | {c['training_steps']} | — |",
        f"| Total generations | {b['total_generations']} | {c['total_generations']} | — |",
        "",
        "## Configuration",
        "",
        "### Shared (both arms)",
        f"- Learning rate: {args.lr}"
        + (f" (baseline: {args.baseline_lr}, candidate: {args.candidate_lr})"
           if args.baseline_lr or args.candidate_lr else ""),
        f"- LoRA rank: {args.lora_rank}",
        f"- Temperature: {args.temperature}",
        f"- Max tokens: {args.max_tokens}",
        "",
        "### Candidate (full pipeline) hyperparameters",
        f"- GTPO beta: {args.gtpo_beta}",
        f"- HICRA alpha: {args.hicra_alpha}",
        f"- SEPA steps: {args.sepa_steps}",
        f"- SEPA schedule: {args.sepa_schedule}",
        "",
        "## Arms",
        "",
        "### Baseline (GRPO)",
        "Simple group-relative advantages: `A_i = r_i - mean(r)`, uniform across tokens.",
        "",
        "### Candidate (Full Pipeline)",
        "MaxRL inverse success-rate reweighting → GTPO entropy-weighted credit assignment → "
        "HICRA planning token amplification → SEPA selective entropy pooling.",
        "",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a two-arm A/B campaign on Tinker: GRPO vs Full pipeline"
    )

    parser.add_argument(
        "--campaign-root", default=None,
        help="Campaign output directory (default: results/tinker_campaign_<timestamp>)",
    )
    parser.add_argument("--execute", action="store_true",
                        help="Execute the campaign (default is dry-run/plan only)")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run candidate arm even if baseline fails")

    # Model & Tinker
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-url", default=None, help="Tinker service URL")
    parser.add_argument("--lora-rank", type=int, default=64)

    # Training (shared across both arms)
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Training steps per arm")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Prompts per step")
    parser.add_argument("--group-size", type=int, default=8,
                        help="Completions per prompt")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate (used for both arms unless overridden)")
    parser.add_argument("--baseline-lr", type=float, default=None,
                        help="Override lr for baseline arm (from sweep)")
    parser.add_argument("--candidate-lr", type=float, default=None,
                        help="Override lr for candidate arm (from sweep)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)

    # Full pipeline hyperparameters (candidate arm)
    parser.add_argument("--gtpo-beta", type=float, default=0.1)
    parser.add_argument("--hicra-alpha", type=float, default=0.2)
    parser.add_argument("--sepa-steps", type=int, default=100)
    parser.add_argument("--sepa-schedule", default="linear", choices=["linear", "auto"])
    parser.add_argument("--sepa-delay-steps", type=int, default=10)
    parser.add_argument("--sepa-correct-rate-gate", type=float, default=0.1)

    # Analysis
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level")
    parser.add_argument("--resamples", type=int, default=20000,
                        help="Resamples for permutation/bootstrap tests")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    campaign_root = (
        Path(args.campaign_root).expanduser().resolve()
        if args.campaign_root
        else _default_campaign_root()
    )
    campaign_root.mkdir(parents=True, exist_ok=True)

    baseline_dir = campaign_root / "baseline"
    candidate_dir = campaign_root / "candidate"
    analysis_dir = campaign_root / "analysis"

    # Build commands for both arms (with optional per-arm lr from sweep)
    baseline_cmd = _build_arm_command("grpo", str(baseline_dir), args,
                                      lr_override=args.baseline_lr)
    candidate_cmd = _build_arm_command("full", str(candidate_dir), args,
                                       lr_override=args.candidate_lr)

    # Write manifest
    manifest: Dict[str, Any] = {
        "campaign_root": str(campaign_root),
        "created_at": datetime.now().isoformat(),
        "execute": args.execute,
        "model": args.model,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "arms": {
            "baseline": {
                "algorithm": "grpo",
                "log_dir": str(baseline_dir),
                "command": baseline_cmd,
            },
            "candidate": {
                "algorithm": "full",
                "log_dir": str(candidate_dir),
                "command": candidate_cmd,
            },
        },
        "hyperparameters": {
            "lr": args.lr,
            "lora_rank": args.lora_rank,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "gtpo_beta": args.gtpo_beta,
            "hicra_alpha": args.hicra_alpha,
            "sepa_steps": args.sepa_steps,
            "sepa_schedule": args.sepa_schedule,
        },
    }

    # Write plan script
    plan_path = campaign_root / "run_commands.sh"
    plan_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Baseline arm (GRPO)",
        " ".join(shlex.quote(c) for c in baseline_cmd),
        "",
        "# Candidate arm (Full pipeline)",
        " ".join(shlex.quote(c) for c in candidate_cmd),
        "",
    ]
    plan_path.write_text("\n".join(plan_lines), encoding="utf-8")
    try:
        plan_path.chmod(0o755)
    except Exception:
        pass

    if not args.execute:
        manifest_path = campaign_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Campaign plan written to: {campaign_root}")
        print(f"  Baseline: {baseline_dir}")
        print(f"  Candidate: {candidate_dir}")
        print(f"  Run script: {plan_path}")
        print("\nTo execute: add --execute flag")
        return 0

    # --- Execute both arms ---
    cwd = _repo_root()

    baseline_rc = _run_arm("baseline (GRPO)", baseline_cmd, cwd)
    if baseline_rc != 0 and not args.continue_on_error:
        print("\nBaseline arm failed. Aborting campaign.")
        manifest["baseline_return_code"] = baseline_rc
        manifest["status"] = "failed"
        (campaign_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    candidate_rc = _run_arm("candidate (Full Pipeline)", candidate_cmd, cwd)

    manifest["baseline_return_code"] = baseline_rc
    manifest["candidate_return_code"] = candidate_rc
    manifest["status"] = "completed" if (baseline_rc == 0 and candidate_rc == 0) else "partial"

    # --- Analysis ---
    if baseline_rc == 0 and candidate_rc == 0:
        analysis_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 60)
        print("  Computing comparison statistics...")
        print("=" * 60 + "\n")

        comparison = _compute_comparison(baseline_dir, candidate_dir)
        manifest["comparison"] = comparison

        # Write comparison JSON
        comp_json = analysis_dir / "comparison.json"
        comp_json.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )

        # Write markdown report
        md_path = _write_comparison_markdown(comparison, analysis_dir, args)
        print(f"  Comparison report: {md_path}")

        # Run significance analysis (uses existing framework)
        sig_result = _run_significance_analysis(
            baseline_dir, candidate_dir, analysis_dir,
            alpha=args.alpha, resamples=args.resamples,
        )
        if sig_result:
            manifest["significance"] = {
                "recommendation": sig_result.get("recommendation", "unknown"),
            }

        # Print summary
        b = comparison["baseline"]
        c = comparison["candidate"]
        d = comparison["delta"]
        print(f"\n{'='*60}")
        print("  CAMPAIGN SUMMARY")
        print(f"{'='*60}")
        print(f"  Baseline correct rate: {b['correct_rate']:.4f} ({b['correct']}/{b['total_generations']})")
        print(f"  Candidate correct rate: {c['correct_rate']:.4f} ({c['correct']}/{c['total_generations']})")
        print(f"  Delta: {d['correct_rate']:+.4f}")
        print(f"  Baseline mean reward: {b['mean_reward']:.4f}")
        print(f"  Candidate mean reward: {c['mean_reward']:.4f}")
        print(f"  Delta: {d['mean_reward']:+.4f}")
        print(f"{'='*60}\n")
    else:
        print("\nOne or both arms failed. Skipping analysis.")

    # Write final manifest
    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Manifest: {manifest_path}")

    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
