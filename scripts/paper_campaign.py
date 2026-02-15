#!/usr/bin/env python3
"""
Run the 8-cell ablation matrix for the MaxRL+SEPA paper.

2 advantage modes (grpo, maxrl) x 4 transform modes (none, gtpo, gtpo_hicra, gtpo_sepa)
= 8 experimental conditions, all with identical hyperparameters except the algorithm.

Hyperparameters are locked from tinker V3 tuned experiments (2026-02-14).

Usage:
    # Smoke test (1 seed, 5 steps — verify all 8 conditions run):
    python scripts/paper_campaign.py --mode smoke --execute

    # Pilot (2 seeds, 20 steps — check for signal):
    python scripts/paper_campaign.py --mode pilot --execute

    # Full campaign (8 seeds, 100 steps — the real experiment):
    python scripts/paper_campaign.py --mode full --execute

    # Dry-run (plan only, no execution):
    python scripts/paper_campaign.py --mode full
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Load .env file if it exists (for TINKER_API_KEY)
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.exists():
    with _env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


# ---- Conditions ----

CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("grpo", "none"),          # C1: baseline
    ("grpo", "gtpo"),          # C2: +entropy signal
    ("grpo", "gtpo_hicra"),    # C3: +planning boost
    ("grpo", "gtpo_sepa"),     # C4: +selective pooling
    ("maxrl", "none"),         # C5: +ML objective
    ("maxrl", "gtpo"),         # C6: +both levels
    ("maxrl", "gtpo_hicra"),   # C7: +both+boost
    ("maxrl", "gtpo_sepa"),    # C8: full stack
)


def condition_label(adv_mode: str, tf_mode: str) -> str:
    return f"{adv_mode}_{tf_mode}"


# ---- Tuned hyperparameters (from tinker V3 experiments, 2026-02-14) ----

TUNED_HPARAMS = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "lora_rank": 64,
    "lr": 5e-5,
    "temperature": 0.7,
    "max_tokens": 2048,
    "batch_size": 16,
    "group_size": 16,
    "gtpo_beta": 0.1,
    "hicra_alpha": 0.2,
    "sepa_steps": 100,
    "sepa_schedule": "linear",
    "sepa_delay_steps": 10,
    "sepa_correct_rate_gate": 0.1,
    "save_every": 20,
}

# ---- Run modes ----

MODES = {
    "smoke": {"max_steps": 5, "seeds": [601]},
    "pilot": {"max_steps": 20, "seeds": [601, 602]},
    "lean":  {"max_steps": 40, "seeds": [601, 602, 603, 604]},
    "full":  {"max_steps": 100, "seeds": [601, 602, 603, 604, 605, 606, 607, 608]},
}

# Lean mode drops H2-only conditions (C2: grpo_gtpo, C6: maxrl_gtpo, C7: maxrl_gtpo_hicra)
# Keeps: C1, C3, C4, C5, C8 — enough for H1, H3, H4, H5
LEAN_CONDITIONS: Tuple[Tuple[str, str], ...] = (
    ("grpo", "none"),          # C1: baseline
    ("grpo", "gtpo_hicra"),    # C3: H3 comparator
    ("grpo", "gtpo_sepa"),     # C4: H3 core claim
    ("maxrl", "none"),         # C5: H1 replication
    ("maxrl", "gtpo_sepa"),    # C8: H4 composition
)


# ---- Data classes ----

@dataclass
class RunSpec:
    condition: str
    advantage_mode: str
    transform_mode: str
    seed: int
    output_dir: str
    command: List[str]


@dataclass
class RunResult:
    condition: str
    seed: int
    output_dir: str
    return_code: int
    success: bool


# ---- Helpers ----

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_exe() -> str:
    """Use the project venv Python if available, else sys.executable."""
    venv_python = _repo_root() / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _default_campaign_root(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "results" / f"paper_campaign_{mode}_{stamp}"


def _build_command(
    adv_mode: str,
    tf_mode: str,
    seed: int,
    log_dir: str,
    args: argparse.Namespace,
) -> List[str]:
    """Build the train_math.py command for one condition+seed."""
    hp = TUNED_HPARAMS
    cmd = [
        _python_exe(),
        "-m", "textpolicy.tinker.train_math",
        "--advantage-mode", adv_mode,
        "--transform-mode", tf_mode,
        "--model", args.model or hp["model"],
        "--max-steps", str(args.max_steps),
        "--batch-size", str(hp["batch_size"]),
        "--group-size", str(hp["group_size"]),
        "--max-tokens", str(hp["max_tokens"]),
        "--temperature", str(hp["temperature"]),
        "--lr", str(args.lr or hp["lr"]),
        "--lora-rank", str(hp["lora_rank"]),
        "--save-every", str(hp["save_every"]),
        "--log-dir", log_dir,
        "--gtpo-beta", str(hp["gtpo_beta"]),
        "--hicra-alpha", str(hp["hicra_alpha"]),
        "--sepa-steps", str(hp["sepa_steps"]),
        "--sepa-schedule", hp["sepa_schedule"],
        "--sepa-delay-steps", str(hp["sepa_delay_steps"]),
        "--sepa-correct-rate-gate", str(hp["sepa_correct_rate_gate"]),
    ]

    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])

    if args.base_url:
        cmd.extend(["--base-url", args.base_url])

    if args.wandb_project:
        run_name = f"{condition_label(adv_mode, tf_mode)}_s{seed}"
        cmd.extend([
            "--wandb-project", args.wandb_project,
            "--wandb-run-name", run_name,
        ])

    return cmd


def _build_specs(
    args: argparse.Namespace,
    seeds: Sequence[int],
    campaign_root: Path,
) -> List[RunSpec]:
    """Build run specs for all conditions x seeds."""
    conditions = LEAN_CONDITIONS if args.mode == "lean" else CONDITIONS
    specs: List[RunSpec] = []
    for adv_mode, tf_mode in conditions:
        label = condition_label(adv_mode, tf_mode)
        for seed in seeds:
            out_dir = campaign_root / f"{label}_seed{seed}"
            cmd = _build_command(
                adv_mode, tf_mode, seed,
                str(out_dir), args,
            )
            specs.append(RunSpec(
                condition=label,
                advantage_mode=adv_mode,
                transform_mode=tf_mode,
                seed=seed,
                output_dir=str(out_dir),
                command=cmd,
            ))
    return specs


def _write_plan_script(specs: Sequence[RunSpec], campaign_root: Path) -> Path:
    """Write a bash script with all commands for manual execution."""
    path = campaign_root / "run_commands.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in specs:
        cmd = " ".join(shlex.quote(x) for x in spec.command)
        lines.append(f"# {spec.condition} seed={spec.seed}")
        lines.append(cmd)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        path.chmod(0o755)
    except Exception:
        pass
    return path


def _run_specs(
    specs: Sequence[RunSpec],
    cwd: Path,
    continue_on_error: bool,
    parallel: bool = False,
) -> List[RunResult]:
    """Execute specs sequentially or in parallel."""
    if parallel:
        return _run_specs_parallel(specs, cwd)
    return _run_specs_sequential(specs, cwd, continue_on_error)


def _run_specs_parallel(
    specs: Sequence[RunSpec],
    cwd: Path,
) -> List[RunResult]:
    """Execute all specs in parallel, wait for all to finish."""
    print(f"  Launching {len(specs)} runs in parallel...")
    processes: List[tuple] = []

    for spec in specs:
        log_path = Path(spec.output_dir).parent / f"{Path(spec.output_dir).name}.log"
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            spec.command, cwd=str(cwd),
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        processes.append((spec, proc, log_file, log_path))
        print(f"    Started: {spec.condition} seed={spec.seed} (pid {proc.pid})")

    print(f"\n  Waiting for all {len(processes)} to finish...")
    results: List[RunResult] = []

    for spec, proc, log_file, log_path in processes:
        proc.wait()
        log_file.close()
        success = proc.returncode == 0
        status = "OK" if success else f"FAILED ({proc.returncode})"
        print(f"    {spec.condition} seed={spec.seed}: {status}")
        results.append(RunResult(
            condition=spec.condition,
            seed=spec.seed,
            output_dir=spec.output_dir,
            return_code=proc.returncode,
            success=success,
        ))

    return results


def _run_specs_sequential(
    specs: Sequence[RunSpec],
    cwd: Path,
    continue_on_error: bool,
) -> List[RunResult]:
    """Execute all specs sequentially."""
    results: List[RunResult] = []
    total = len(specs)

    for i, spec in enumerate(specs, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{total}] {spec.condition} seed={spec.seed}")
        print(f"{'='*60}")

        proc = subprocess.run(spec.command, cwd=str(cwd), check=False)
        result = RunResult(
            condition=spec.condition,
            seed=spec.seed,
            output_dir=spec.output_dir,
            return_code=proc.returncode,
            success=(proc.returncode == 0),
        )
        results.append(result)

        if not result.success:
            print(f"  FAILED (return code {proc.returncode})")
            if not continue_on_error:
                print("  Stopping. Use --continue-on-error to keep going.")
                break
        else:
            print(f"  OK")

    return results


# ---- Analysis ----

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _run_metrics(run_dir: Path) -> Optional[Dict[str, float]]:
    """Extract summary metrics from one run's emergence data."""
    steps = _load_jsonl(run_dir / "emergence" / "steps.jsonl")
    if not steps:
        # Fall back to metrics.jsonl
        steps = _load_jsonl(run_dir / "metrics.jsonl")
    if not steps:
        return None

    total_count = sum(int(s.get("total_count", 0)) for s in steps)
    correct_count = sum(int(s.get("correct_count", 0)) for s in steps)
    step_rewards = [_safe_float(s.get("mean_reward")) for s in steps]

    # Entropy stats (for H5)
    exec_vars = [_safe_float(s.get("exec_entropy_var")) for s in steps
                 if "exec_entropy_var" in s]
    plan_means = [_safe_float(s.get("plan_entropy_mean")) for s in steps
                  if "plan_entropy_mean" in s]

    metrics: Dict[str, float] = {
        "num_steps": float(len(steps)),
        "total_count": float(total_count),
        "correct_count": float(correct_count),
        "correct_rate": (float(correct_count) / float(total_count))
                        if total_count > 0 else 0.0,
        "mean_reward": sum(step_rewards) / len(step_rewards)
                       if step_rewards else 0.0,
        "final_reward": step_rewards[-1] if step_rewards else 0.0,
    }

    if exec_vars:
        metrics["mean_exec_entropy_var"] = sum(exec_vars) / len(exec_vars)
    if plan_means:
        metrics["mean_plan_entropy_mean"] = sum(plan_means) / len(plan_means)

    return metrics


def _aggregate_by_condition(
    run_results: Sequence[RunResult],
) -> Dict[str, Dict[str, float]]:
    """Aggregate metrics across seeds for each condition."""
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for result in run_results:
        if not result.success:
            continue
        metrics = _run_metrics(Path(result.output_dir))
        if metrics is None:
            continue
        grouped.setdefault(result.condition, []).append(metrics)

    out: Dict[str, Dict[str, float]] = {}
    for cond, metric_list in grouped.items():
        n = float(len(metric_list))
        total_count = sum(m["total_count"] for m in metric_list)
        correct_count = sum(m["correct_count"] for m in metric_list)
        out[cond] = {
            "num_seeds": n,
            "total_generations": total_count,
            "correct_count": correct_count,
            "correct_rate": (correct_count / total_count)
                            if total_count > 0 else 0.0,
            "mean_reward": sum(m["mean_reward"] for m in metric_list) / n,
            "final_reward": sum(m["final_reward"] for m in metric_list) / n,
        }
        # Include entropy stats if available
        exec_vars = [m["mean_exec_entropy_var"] for m in metric_list
                     if "mean_exec_entropy_var" in m]
        if exec_vars:
            out[cond]["mean_exec_entropy_var"] = sum(exec_vars) / len(exec_vars)

    return out


def _build_results_markdown(
    campaign_root: Path,
    summary: Dict[str, Dict[str, float]],
    mode: str,
) -> str:
    """Build a markdown summary of the 8-cell matrix results."""
    lines: List[str] = []
    lines.append("# Paper Campaign Results")
    lines.append("")
    lines.append(f"- mode: `{mode}`")
    lines.append(f"- campaign: `{campaign_root.name}`")
    lines.append(f"- date: {datetime.now().isoformat()}")
    lines.append("")

    # Hypothesis mapping
    lines.append("## Hypothesis mapping")
    lines.append("")
    lines.append("| Hypothesis | Comparison | Cells |")
    lines.append("|---|---|---|")
    lines.append("| H1: MaxRL > GRPO | grpo_none vs maxrl_none | C1 vs C5 |")
    lines.append("| H2: GTPO > flat | grpo_none vs grpo_gtpo | C1 vs C2 |")
    lines.append("| H3: SEPA > HICRA | grpo_gtpo_hicra vs grpo_gtpo_sepa | C3 vs C4 |")
    lines.append("| H4: Composition | C1, C5, C4, C8 | orthogonal gains |")
    lines.append("| H5: Entropy shift | C3 vs C4 (entropy logs) | mechanistic |")
    lines.append("")

    # Results table
    lines.append("## 8-cell matrix")
    lines.append("")
    lines.append("| Condition | Seeds | Generations | Correct | Rate | Mean reward | Exec H var |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    active_conditions = LEAN_CONDITIONS if mode == "lean" else CONDITIONS
    for adv_mode, tf_mode in active_conditions:
        label = condition_label(adv_mode, tf_mode)
        rec = summary.get(label)
        if rec is None:
            lines.append(f"| {label} | — | — | — | — | — | — |")
            continue
        exec_var = rec.get("mean_exec_entropy_var")
        exec_var_str = f"{exec_var:.4f}" if exec_var is not None else "—"
        lines.append(
            f"| {label} "
            f"| {int(rec['num_seeds'])} "
            f"| {int(rec['total_generations'])} "
            f"| {int(rec['correct_count'])} "
            f"| {rec['correct_rate']:.4%} "
            f"| {rec['mean_reward']:.4f} "
            f"| {exec_var_str} |"
        )

    lines.append("")

    # H1 comparison
    c1 = summary.get("grpo_none")
    c5 = summary.get("maxrl_none")
    if c1 and c5:
        lines.append("## H1: MaxRL vs GRPO (episode-level)")
        lines.append("")
        delta = c5["correct_rate"] - c1["correct_rate"]
        direction = "+" if delta >= 0 else ""
        lines.append(f"- GRPO correct rate: {c1['correct_rate']:.4%}")
        lines.append(f"- MaxRL correct rate: {c5['correct_rate']:.4%}")
        lines.append(f"- Delta: {direction}{delta:.4%}")
        lines.append(f"- Verdict: {'PASS' if delta >= 0 else 'FAIL'} (directional)")
        lines.append("")

    # H3 comparison
    c3 = summary.get("grpo_gtpo_hicra")
    c4 = summary.get("grpo_gtpo_sepa")
    if c3 and c4:
        lines.append("## H3: SEPA vs HICRA (token-level)")
        lines.append("")
        delta = c4["correct_rate"] - c3["correct_rate"]
        direction = "+" if delta >= 0 else ""
        lines.append(f"- HICRA correct rate: {c3['correct_rate']:.4%}")
        lines.append(f"- SEPA correct rate: {c4['correct_rate']:.4%}")
        lines.append(f"- Delta: {direction}{delta:.4%}")
        c3_var = c3.get("mean_exec_entropy_var")
        c4_var = c4.get("mean_exec_entropy_var")
        if c3_var is not None and c4_var is not None:
            lines.append(f"- Exec entropy var (HICRA): {c3_var:.4f}")
            lines.append(f"- Exec entropy var (SEPA): {c4_var:.4f}")
            lines.append(f"- Entropy var reduction: {c3_var - c4_var:.4f}")
        lines.append(f"- Verdict: {'PASS' if delta >= 0 else 'FAIL'} (directional)")
        lines.append("")

    # H4 composition
    c8 = summary.get("maxrl_gtpo_sepa")
    if c1 and c5 and c4 and c8:
        lines.append("## H4: Orthogonal composition")
        lines.append("")
        rates = {
            "grpo_none (C1)": c1["correct_rate"],
            "maxrl_none (C5)": c5["correct_rate"],
            "grpo_gtpo_sepa (C4)": c4["correct_rate"],
            "maxrl_gtpo_sepa (C8)": c8["correct_rate"],
        }
        ranked = sorted(rates.items(), key=lambda x: -x[1])
        for i, (name, rate) in enumerate(ranked, 1):
            lines.append(f"- #{i}: {name} = {rate:.4%}")
        lines.append(f"- Verdict: {'PASS' if ranked[0][0].startswith('maxrl_gtpo_sepa') else 'FAIL'} (C8 should be #1)")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---- CLI ----

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 8-cell paper ablation matrix on Tinker GPUs."
    )
    parser.add_argument(
        "--mode", type=str, default="smoke",
        choices=["smoke", "pilot", "lean", "full"],
        help="Run mode: smoke (1 seed, 5 steps), pilot (2 seeds, 20 steps), "
             "lean (4 seeds, 40 steps, 5 conditions), full (8 seeds, 100 steps)",
    )
    parser.add_argument("--campaign-root", default=None)
    parser.add_argument("--model", default=None, help="Override model")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override steps from mode")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--wandb-project", default=None,
                        help="Wandb project for all runs")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel", action="store_true",
                        help="Run all conditions in parallel (default: sequential)")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    mode_config = MODES[args.mode]
    seeds = mode_config["seeds"]
    if args.max_steps is None:
        args.max_steps = mode_config["max_steps"]

    campaign_root = (
        Path(args.campaign_root).expanduser().resolve()
        if args.campaign_root
        else _default_campaign_root(args.mode)
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = campaign_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_specs(args, seeds, campaign_root)
    plan_script = _write_plan_script(specs, campaign_root)

    # Write manifest
    manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "campaign_root": str(campaign_root),
        "mode": args.mode,
        "execute": bool(args.execute),
        "seeds": list(seeds),
        "max_steps": args.max_steps,
        "conditions": [condition_label(a, t) for a, t in CONDITIONS],
        "tuned_hparams": TUNED_HPARAMS,
        "planned_runs": [asdict(s) for s in specs],
    }

    active_conditions = LEAN_CONDITIONS if args.mode == "lean" else CONDITIONS

    if not args.execute:
        manifest_path = campaign_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Paper campaign plan ({args.mode} mode)")
        print(f"  Conditions: {len(active_conditions)}")
        print(f"  Seeds: {seeds}")
        print(f"  Steps: {args.max_steps}")
        print(f"  Total runs: {len(specs)}")
        print(f"  Plan script: {plan_script}")
        print(f"  Campaign root: {campaign_root}")
        return 0

    # Execute
    print(f"\nPaper campaign ({args.mode} mode)")
    print(f"  {len(active_conditions)} conditions x {len(seeds)} seeds = {len(specs)} runs")
    print(f"  {args.max_steps} steps per run")
    print(f"  Hyperparams: lr={TUNED_HPARAMS['lr']}, rank={TUNED_HPARAMS['lora_rank']}")
    print()

    run_results = _run_specs(
        specs, cwd=_repo_root(),
        continue_on_error=args.continue_on_error,
        parallel=args.parallel,
    )
    manifest["run_results"] = [asdict(r) for r in run_results]

    # Analysis
    summary = _aggregate_by_condition(run_results)
    manifest["summary"] = summary

    # Write outputs
    (analysis_dir / "summary.json").write_text(
        json.dumps({"summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )

    md = _build_results_markdown(campaign_root, summary, args.mode)
    (analysis_dir / "results.md").write_text(md, encoding="utf-8")

    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}\n")

    succeeded = sum(1 for r in run_results if r.success)
    failed = sum(1 for r in run_results if not r.success)
    print(f"  Runs: {succeeded} succeeded, {failed} failed")
    print()

    active_conditions = LEAN_CONDITIONS if args.mode == "lean" else CONDITIONS
    for adv_mode, tf_mode in active_conditions:
        label = condition_label(adv_mode, tf_mode)
        rec = summary.get(label)
        if rec:
            print(f"  {label:25s}  correct={rec['correct_rate']:7.4%}  "
                  f"reward={rec['mean_reward']:+.4f}")
        else:
            print(f"  {label:25s}  (no data)")

    print(f"\n  Results: {analysis_dir / 'results.md'}")
    print(f"  Manifest: {manifest_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
