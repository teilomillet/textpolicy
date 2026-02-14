#!/usr/bin/env python3
"""
Run a method matrix for countdown reasoning at fixed generation budget.

Compares:
- grpo_alone
- grpo_hicra
- grpo_sepa
- gtpo
- maxrl_alone
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from textpolicy.analysis import evaluate_sepa_significance


METHODS: Tuple[str, ...] = (
    "grpo_alone",
    "grpo_hicra",
    "grpo_sepa",
    "gtpo",
    "maxrl_alone",
)


@dataclass
class RunSpec:
    method: str
    seed: int
    output_dir: str
    command: List[str]


@dataclass
class RunResult:
    method: str
    seed: int
    output_dir: str
    command: List[str]
    return_code: int
    success: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_campaign_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "results" / f"reasoning_method_matrix_{stamp}"


def _parse_seeds(text: str) -> List[int]:
    seeds = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("Seed list is empty.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates.")
    return seeds


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 5-method reasoning matrix (GRPO/GTPO/HICRA/SEPA/MaxRL) "
            "with aligned seed and budget."
        )
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help=(
            "Output directory for all runs. Default: "
            "results/reasoning_method_matrix_<timestamp>"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="601",
        help="Comma-separated seeds shared across methods.",
    )
    parser.add_argument(
        "--model",
        default="arcee-ai/Trinity-Nano-Preview",
        help="Model id for all methods.",
    )
    parser.add_argument("--steps", type=int, default=1, help="Training steps per run.")
    parser.add_argument("--num-problems", type=int, default=2, help="Problems per run.")
    parser.add_argument(
        "--episodes-per-step",
        type=int,
        default=2,
        help="Episodes per step for all methods.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--alpha-1", type=float, default=1.0, help="GTPO alpha_1.")
    parser.add_argument("--alpha-2", type=float, default=0.1, help="GTPO alpha_2.")
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=0.5,
        help="GTPO reward threshold.",
    )
    parser.add_argument(
        "--hicra-gamma",
        type=float,
        default=0.3,
        help="HICRA amplification parameter.",
    )
    parser.add_argument(
        "--sepa-steps",
        type=int,
        default=1,
        help="SEPA linear horizon for grpo_sepa method.",
    )
    parser.add_argument(
        "--sepa-schedule",
        choices=["linear", "auto"],
        default="linear",
        help="SEPA schedule for grpo_sepa method.",
    )
    parser.add_argument(
        "--sepa-delay-steps",
        type=int,
        default=0,
        help="SEPA delay before lambda ramp.",
    )
    parser.add_argument(
        "--sepa-correct-rate-gate",
        type=float,
        default=0.0,
        help="Sticky correct-rate gate for SEPA.",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional wandb project for all methods.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Force WANDB_MODE=offline when executing.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run commands. Default is dry-run plan only.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs after a failure.",
    )
    return parser


def _build_method_flags(args: argparse.Namespace, method: str) -> List[str]:
    if method == "grpo_alone":
        return ["--transform-mode", "none"]
    if method == "grpo_hicra":
        return [
            "--transform-mode",
            "hicra",
            "--hicra-gamma",
            str(args.hicra_gamma),
        ]
    if method == "grpo_sepa":
        return [
            "--transform-mode",
            "gtpo_sepa",
            "--sepa-steps",
            str(args.sepa_steps),
            "--sepa-schedule",
            str(args.sepa_schedule),
            "--sepa-delay-steps",
            str(args.sepa_delay_steps),
            "--sepa-correct-rate-gate",
            str(args.sepa_correct_rate_gate),
        ]
    if method == "gtpo":
        return ["--transform-mode", "gtpo"]
    if method == "maxrl_alone":
        return ["--maxrl", "--transform-mode", "none"]
    raise ValueError(f"Unknown method: {method}")


def _build_specs(
    args: argparse.Namespace,
    seeds: Sequence[int],
    campaign_root: Path,
) -> List[RunSpec]:
    exp_script = _repo_root() / "experiments" / "countdown_reasoning_lora.py"
    specs: List[RunSpec] = []
    for method in METHODS:
        for seed in seeds:
            out_dir = campaign_root / f"{method}_seed{seed}"
            cmd = [
                sys.executable,
                str(exp_script),
                "--model",
                str(args.model),
                "--steps",
                str(args.steps),
                "--num-problems",
                str(args.num_problems),
                "--episodes-per-step",
                str(args.episodes_per_step),
                "--batch-size",
                str(args.batch_size),
                "--max-tokens",
                str(args.max_tokens),
                "--temperature",
                str(args.temperature),
                "--lr",
                str(args.lr),
                "--seed",
                str(seed),
                "--alpha-1",
                str(args.alpha_1),
                "--alpha-2",
                str(args.alpha_2),
                "--reward-threshold",
                str(args.reward_threshold),
                "--output",
                str(out_dir),
                "--no-litmus",
                "--wandb-completion-log-interval",
                "1",
                "--wandb-completion-char-limit",
                "0",
            ]
            cmd.extend(_build_method_flags(args, method))
            if args.wandb_project:
                run_name = f"{campaign_root.name}-{method}-seed{seed}"
                cmd.extend(
                    [
                        "--wandb-project",
                        str(args.wandb_project),
                        "--wandb-run-name",
                        run_name,
                    ]
                )
            specs.append(
                RunSpec(
                    method=method,
                    seed=seed,
                    output_dir=str(out_dir),
                    command=cmd,
                )
            )
    return specs


def _write_plan_script(specs: Sequence[RunSpec], campaign_root: Path, wandb_offline: bool) -> Path:
    path = campaign_root / "run_commands.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in specs:
        cmd = " ".join(shlex.quote(x) for x in spec.command)
        if wandb_offline:
            cmd = f"WANDB_MODE=offline {cmd}"
        lines.append(cmd)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass
    return path


def _run_specs(
    specs: Sequence[RunSpec],
    *,
    cwd: Path,
    wandb_offline: bool,
    continue_on_error: bool,
) -> List[RunResult]:
    results: List[RunResult] = []
    env = os.environ.copy()
    if wandb_offline:
        env["WANDB_MODE"] = "offline"

    for spec in specs:
        print(f"Running {spec.method} seed={spec.seed} -> {spec.output_dir}")
        proc = subprocess.run(spec.command, cwd=str(cwd), env=env, check=False)
        result = RunResult(
            method=spec.method,
            seed=spec.seed,
            output_dir=spec.output_dir,
            command=spec.command,
            return_code=proc.returncode,
            success=(proc.returncode == 0),
        )
        results.append(result)
        if not result.success and not continue_on_error:
            break
    return results


def _run_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    steps = _read_jsonl(run_dir / "emergence" / "steps.jsonl")
    if not steps:
        return None
    total_count = sum(_safe_int(x.get("total_count")) for x in steps)
    correct_count = sum(_safe_int(x.get("correct_count")) for x in steps)
    hit_count = sum(_safe_int(x.get("max_tokens_hit_count")) for x in steps)
    step_rewards = [_safe_float(x.get("mean_reward")) for x in steps]
    planning = [_safe_float(x.get("planning_token_ratio")) for x in steps]
    completion_len = [_safe_float(x.get("mean_completion_length")) for x in steps]
    return {
        "num_steps": float(len(steps)),
        "total_count": float(total_count),
        "correct_count": float(correct_count),
        "correct_rate": (float(correct_count) / float(total_count)) if total_count > 0 else 0.0,
        "mean_reward_steps": _mean(step_rewards),
        "final_reward": step_rewards[-1] if step_rewards else 0.0,
        "mean_planning_ratio": _mean(planning),
        "mean_completion_length": _mean(completion_len),
        "max_tokens_hit_count": float(hit_count),
        "max_tokens_hit_rate": (float(hit_count) / float(total_count)) if total_count > 0 else 0.0,
    }


def _aggregate_by_method(run_results: Sequence[RunResult]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Tuple[RunResult, Dict[str, float]]]] = {m: [] for m in METHODS}
    for result in run_results:
        if not result.success:
            continue
        metrics = _run_metrics(Path(result.output_dir))
        if metrics is None:
            continue
        grouped[result.method].append((result, metrics))

    out: Dict[str, Dict[str, float]] = {}
    for method, pairs in grouped.items():
        if not pairs:
            continue
        run_count = float(len(pairs))
        total_count = sum(m["total_count"] for _, m in pairs)
        correct_count = sum(m["correct_count"] for _, m in pairs)
        hit_count = sum(m["max_tokens_hit_count"] for _, m in pairs)
        total_steps = sum(m["num_steps"] for _, m in pairs)
        out[method] = {
            "run_count": run_count,
            "total_steps": total_steps,
            "total_count": total_count,
            "correct_count": correct_count,
            "correct_rate": (correct_count / total_count) if total_count > 0 else 0.0,
            "mean_reward_steps": (
                sum(m["mean_reward_steps"] * m["num_steps"] for _, m in pairs) / total_steps
                if total_steps > 0
                else 0.0
            ),
            "mean_planning_ratio": (
                sum(m["mean_planning_ratio"] * m["num_steps"] for _, m in pairs) / total_steps
                if total_steps > 0
                else 0.0
            ),
            "mean_completion_length": (
                sum(m["mean_completion_length"] * m["num_steps"] for _, m in pairs) / total_steps
                if total_steps > 0
                else 0.0
            ),
            "max_tokens_hit_count": hit_count,
            "max_tokens_hit_rate": (hit_count / total_count) if total_count > 0 else 0.0,
        }
    return out


def _pairwise_vs_grpo(run_results: Sequence[RunResult]) -> Dict[str, dict]:
    by_method: Dict[str, List[str]] = {m: [] for m in METHODS}
    for result in run_results:
        if result.success:
            by_method[result.method].append(result.output_dir)

    baseline = by_method.get("grpo_alone", [])
    if not baseline:
        return {}

    out: Dict[str, dict] = {}
    for method in METHODS:
        if method == "grpo_alone":
            continue
        candidate = by_method.get(method, [])
        if not candidate:
            continue
        report = evaluate_sepa_significance(
            baseline_run_dirs=baseline,
            candidate_run_dirs=candidate,
            alpha=0.05,
            num_resamples=20000,
            seed=0,
        )
        payload = report.to_dict()
        out[method] = {
            "recommendation": payload.get("recommendation"),
            "rate_test": payload.get("rate_test"),
            "mean_tests": payload.get("mean_tests"),
            "sample_size_estimates": payload.get("sample_size_estimates"),
        }
    return out


def _build_markdown(
    campaign_root: Path,
    summary: Dict[str, Dict[str, float]],
    pairwise: Dict[str, dict],
) -> str:
    lines: List[str] = []
    lines.append("# Reasoning Method Matrix")
    lines.append("")
    lines.append(f"- campaign: `{campaign_root}`")
    lines.append("")
    lines.append("## Per-Method Summary")
    lines.append("")
    lines.append(
        "| method | runs | episodes | correct_rate | mean_reward | max_tok_hit_rate | mean_len |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method in METHODS:
        rec = summary.get(method)
        if rec is None:
            continue
        lines.append(
            "| "
            f"{method} | "
            f"{int(rec['run_count'])} | "
            f"{int(rec['total_count'])} | "
            f"{rec['correct_rate']:.4%} | "
            f"{rec['mean_reward_steps']:.6f} | "
            f"{rec['max_tokens_hit_rate']:.4%} | "
            f"{rec['mean_completion_length']:.2f} |"
        )
    lines.append("")
    lines.append("## Pairwise vs grpo_alone")
    lines.append("")
    if not pairwise:
        lines.append("No pairwise reports available.")
        return "\n".join(lines) + "\n"
    lines.append("| candidate | correct_delta | p_value | recommendation |")
    lines.append("|---|---:|---:|---|")
    for method in METHODS:
        if method == "grpo_alone":
            continue
        report = pairwise.get(method)
        if not report:
            continue
        rate = report.get("rate_test") or {}
        lines.append(
            "| "
            f"{method} | "
            f"{_safe_float(rate.get('delta')):+.4%} | "
            f"{_safe_float(rate.get('p_value')):.4f} | "
            f"{report.get('recommendation', 'n/a')} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _build_parser().parse_args()
    seeds = _parse_seeds(args.seeds)
    campaign_root = (
        Path(args.campaign_root).expanduser().resolve()
        if args.campaign_root
        else _default_campaign_root()
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = campaign_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args, seeds, campaign_root)
    plan_script = _write_plan_script(specs, campaign_root, args.wandb_offline)

    manifest: Dict[str, object] = {
        "created_at": datetime.now().isoformat(),
        "campaign_root": str(campaign_root),
        "execute": bool(args.execute),
        "seeds": list(seeds),
        "methods": list(METHODS),
        "planned_runs": [
            {
                "method": s.method,
                "seed": s.seed,
                "output_dir": s.output_dir,
                "command": s.command,
            }
            for s in specs
        ],
        "plan_script": str(plan_script),
    }

    if not args.execute:
        manifest_path = campaign_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared matrix plan at: {campaign_root}")
        print(f"Planned runs: {len(specs)}")
        print(f"Run script: {plan_script}")
        print(f"Manifest: {manifest_path}")
        return 0

    run_results = _run_specs(
        specs,
        cwd=_repo_root(),
        wandb_offline=args.wandb_offline,
        continue_on_error=args.continue_on_error,
    )
    manifest["run_results"] = [asdict(r) for r in run_results]

    summary = _aggregate_by_method(run_results)
    pairwise = _pairwise_vs_grpo(run_results)
    manifest["summary"] = summary
    manifest["pairwise_vs_grpo_alone"] = pairwise

    (analysis_dir / "method_matrix_summary.json").write_text(
        json.dumps(
            {
                "campaign_root": str(campaign_root),
                "summary": summary,
                "pairwise_vs_grpo_alone": pairwise,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "method_matrix_summary.md").write_text(
        _build_markdown(campaign_root, summary, pairwise),
        encoding="utf-8",
    )

    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote matrix summary: {analysis_dir / 'method_matrix_summary.json'}")
    print(f"Wrote markdown summary: {analysis_dir / 'method_matrix_summary.md'}")
    print(f"Manifest: {manifest_path}")

    failed = any(not r.success for r in run_results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
