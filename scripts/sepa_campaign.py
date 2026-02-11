#!/usr/bin/env python3
"""
Prepare or run a multi-seed SEPA vs baseline campaign.

Default mode is dry-run (plan only). Use --execute to actually run.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from textpolicy.analysis import (
    build_litmus_markdown,
    build_sepa_significance_markdown,
    evaluate_sepa_litmus,
    evaluate_sepa_significance,
    get_sepa_litmus_profile,
)


@dataclass
class RunSpec:
    group: str  # baseline | candidate
    seed: int
    output_dir: str
    command: List[str]


@dataclass
class RunResult:
    group: str
    seed: int
    output_dir: str
    command: List[str]
    return_code: int
    success: bool


def _parse_seed_list(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("Seed list is empty.")
    seeds: List[int] = []
    for part in parts:
        seeds.append(int(part))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed list contains duplicates.")
    return seeds


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare/run a multi-seed A/B campaign for SEPA significance. "
            "Dry-run by default."
        )
    )
    parser.add_argument(
        "--campaign-root",
        default=None,
        help=(
            "Campaign output directory. Default: "
            "results/sepa_campaign_<YYYYMMDD_HHMMSS>"
        ),
    )
    parser.add_argument(
        "--seeds",
        default="101,102,103,104,105,106,107,108",
        help="Comma-separated dataset seeds (paired across both arms).",
    )
    parser.add_argument(
        "--model",
        default="arcee-ai/Trinity-Nano-Preview",
        help="Model ID for both arms.",
    )
    parser.add_argument("--steps", type=int, default=12, help="Training steps per run.")
    parser.add_argument("--num-problems", type=int, default=16, help="Problems per run.")
    parser.add_argument(
        "--episodes-per-step",
        type=int,
        default=8,
        help="Episodes per step for both arms.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")

    parser.add_argument(
        "--baseline-hicra-gamma",
        type=float,
        default=0.3,
        help="HICRA gamma for baseline arm (SEPA disabled).",
    )
    parser.add_argument(
        "--candidate-sepa-steps",
        type=int,
        default=8,
        help="SEPA steps for candidate arm.",
    )
    parser.add_argument(
        "--candidate-sepa-schedule",
        choices=["linear", "auto"],
        default="linear",
        help="SEPA schedule for candidate arm.",
    )

    parser.add_argument(
        "--wandb-project",
        default=None,
        help="Optional wandb project for both arms.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Set WANDB_MODE=offline when executing runs.",
    )

    parser.add_argument(
        "--litmus-profile",
        default="official_v1",
        help="Litmus profile used for campaign-level verdict.",
    )
    parser.add_argument(
        "--litmus-min-run-pairs",
        type=int,
        default=0,
        help=(
            "Campaign-level min_run_pairs override. "
            "0 means use number of planned paired seeds."
        ),
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=20000,
        help="Resamples for significance tests.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance alpha.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute all planned runs (default is dry-run only).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="When executing, continue remaining runs after failures.",
    )
    return parser


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_campaign_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "results" / f"sepa_campaign_{stamp}"


def _build_run_specs(args: argparse.Namespace, seeds: Sequence[int], campaign_root: Path) -> List[RunSpec]:
    exp_script = _repo_root() / "experiments" / "countdown_reasoning_lora.py"
    specs: List[RunSpec] = []

    for seed in seeds:
        for group in ("baseline", "candidate"):
            out_dir = campaign_root / f"{group}_seed{seed}"
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
                "--output",
                str(out_dir),
                "--no-litmus",
                "--wandb-completion-log-interval",
                "1",
                "--wandb-completion-char-limit",
                "0",
            ]

            if group == "baseline":
                cmd.extend(
                    [
                        "--sepa-steps",
                        "0",
                        "--sepa-schedule",
                        "linear",
                        "--hicra-gamma",
                        str(args.baseline_hicra_gamma),
                    ]
                )
            else:
                cmd.extend(
                    [
                        "--sepa-steps",
                        str(args.candidate_sepa_steps),
                        "--sepa-schedule",
                        str(args.candidate_sepa_schedule),
                    ]
                )

            if args.wandb_project:
                run_name = f"{campaign_root.name}-{group}-seed{seed}"
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
                    group=group,
                    seed=seed,
                    output_dir=str(out_dir),
                    command=cmd,
                )
            )
    return specs


def _write_plan_script(specs: Sequence[RunSpec], campaign_root: Path, wandb_offline: bool) -> Path:
    plan_path = campaign_root / "run_commands.sh"
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for spec in specs:
        cmd = " ".join(shlex.quote(part) for part in spec.command)
        if wandb_offline:
            cmd = f"WANDB_MODE=offline {cmd}"
        lines.append(cmd)
    plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        plan_path.chmod(0o755)
    except Exception:
        pass
    return plan_path


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
        print(f"Running {spec.group} seed={spec.seed} -> {spec.output_dir}")
        proc = subprocess.run(spec.command, cwd=str(cwd), env=env, check=False)
        success = proc.returncode == 0
        results.append(
            RunResult(
                group=spec.group,
                seed=spec.seed,
                output_dir=spec.output_dir,
                command=spec.command,
                return_code=proc.returncode,
                success=success,
            )
        )
        if not success and not continue_on_error:
            break
    return results


def _paired_success_dirs(results: Sequence[RunResult]) -> tuple[List[str], List[str]]:
    by_seed: Dict[int, Dict[str, RunResult]] = {}
    for result in results:
        by_seed.setdefault(result.seed, {})[result.group] = result

    baseline_dirs: List[str] = []
    candidate_dirs: List[str] = []
    for seed in sorted(by_seed):
        groups = by_seed[seed]
        base = groups.get("baseline")
        cand = groups.get("candidate")
        if base and cand and base.success and cand.success:
            baseline_dirs.append(base.output_dir)
            candidate_dirs.append(cand.output_dir)
    return baseline_dirs, candidate_dirs


def main() -> int:
    args = _build_parser().parse_args()
    seeds = _parse_seed_list(args.seeds)
    campaign_root = (
        Path(args.campaign_root).expanduser().resolve()
        if args.campaign_root
        else _default_campaign_root()
    )
    campaign_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = campaign_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    specs = _build_run_specs(args, seeds, campaign_root)
    plan_script = _write_plan_script(specs, campaign_root, args.wandb_offline)

    manifest: Dict[str, object] = {
        "campaign_root": str(campaign_root),
        "created_at": datetime.now().isoformat(),
        "execute": bool(args.execute),
        "seeds": seeds,
        "planned_runs": [
            {
                "group": spec.group,
                "seed": spec.seed,
                "output_dir": spec.output_dir,
                "command": spec.command,
            }
            for spec in specs
        ],
        "plan_script": str(plan_script),
    }

    if not args.execute:
        manifest_path = campaign_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared campaign plan at: {campaign_root}")
        print(f"Planned runs: {len(specs)} ({len(seeds)} paired seeds)")
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

    baseline_dirs, candidate_dirs = _paired_success_dirs(run_results)
    manifest["paired_success_count"] = len(baseline_dirs)
    manifest["paired_baseline_dirs"] = baseline_dirs
    manifest["paired_candidate_dirs"] = candidate_dirs

    if baseline_dirs and candidate_dirs:
        profile = get_sepa_litmus_profile(args.litmus_profile)
        target_pairs = args.litmus_min_run_pairs if args.litmus_min_run_pairs > 0 else len(seeds)
        evidence = profile.evidence
        if target_pairs > evidence.min_run_pairs:
            evidence = replace(evidence, min_run_pairs=target_pairs)

        litmus = evaluate_sepa_litmus(
            baseline_run_dirs=baseline_dirs,
            candidate_run_dirs=candidate_dirs,
            thresholds=profile.thresholds,
            evidence=evidence,
        )
        litmus_json_path = analysis_dir / "litmus.json"
        litmus_md_path = analysis_dir / "litmus.md"
        litmus_json_path.write_text(
            json.dumps(litmus.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        litmus_md_path.write_text(build_litmus_markdown(litmus), encoding="utf-8")

        significance = evaluate_sepa_significance(
            baseline_run_dirs=baseline_dirs,
            candidate_run_dirs=candidate_dirs,
            alpha=float(args.alpha),
            num_resamples=int(args.resamples),
            seed=0,
        )
        sig_json_path = analysis_dir / "significance.json"
        sig_md_path = analysis_dir / "significance.md"
        sig_json_path.write_text(
            json.dumps(significance.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        sig_md_path.write_text(
            build_sepa_significance_markdown(significance), encoding="utf-8"
        )

        manifest["analysis"] = {
            "litmus_status": litmus.status,
            "litmus_json": str(litmus_json_path),
            "litmus_md": str(litmus_md_path),
            "significance_recommendation": significance.recommendation,
            "significance_json": str(sig_json_path),
            "significance_md": str(sig_md_path),
        }
        print(f"Litmus status: {litmus.status}")
        print(f"Significance recommendation: {significance.recommendation}")
        print(f"Wrote analysis: {analysis_dir}")
    else:
        print("No successful paired runs available; skipped litmus/significance analysis.")

    manifest_path = campaign_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    had_failure = any(not r.success for r in run_results)
    return 1 if had_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
