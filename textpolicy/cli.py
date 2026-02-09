"""
Minimal CLI entry point for TextPolicy.

Design goals:
- Keep it tiny and dependency-free (argparse only)
- Provide a single high-signal command for students: `validate`
- Exit non-zero on failure for CI integration

This CLI is intentionally small; a config-driven runner can be added later.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from .validate import validate_installation


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run installation validation and print results.

    Uses the programmatic validate_installation() so behavior stays consistent.
    """
    report: Dict[str, Any] = validate_installation(verbose=not args.json)
    if args.json:
        print(json.dumps(report, indent=2))
    # Exit code communicates status for CI/automation
    return 0 if report.get("status") == "ok" else 1


def _cmd_sweep_lr(args: argparse.Namespace) -> int:
    """Run a learning-rate sweep on a log-uniform grid."""
    from experiments.sweep_lr import SweepConfig, run_sweep, print_results_table

    config = SweepConfig(
        model_id=args.model,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        steps_per_lr=args.steps,
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        points_per_decade=args.points_per_decade,
        output_dir=args.output,
    )

    results = run_sweep(config)
    print_results_table(results)

    if args.json:
        from dataclasses import asdict
        from pathlib import Path

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "results.json"
        output = {
            "config": asdict(config),
            "results": [asdict(r) for r in results],
        }
        output_path.write_text(json.dumps(output, indent=2))
        print(f"\nJSON saved to {output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser.

    Default behavior is `validate` when no subcommand is provided for convenience.
    """
    parser = argparse.ArgumentParser(prog="textpolicy", description="TextPolicy command-line tools")
    subparsers = parser.add_subparsers(dest="command")

    p_validate = subparsers.add_parser("validate", help="Validate installation and environment")
    p_validate.add_argument("--json", action="store_true", help="Output machine-readable JSON report")
    p_validate.set_defaults(func=_cmd_validate)

    p_sweep = subparsers.add_parser(
        "sweep-lr", help="Sweep learning rates on a log-uniform grid",
    )
    p_sweep.add_argument("--model", default="arcee-ai/Trinity-Nano-Preview", help="Model ID")
    p_sweep.add_argument("--lora-rank", type=int, default=2, help="LoRA rank (alpha=rank)")
    p_sweep.add_argument("--lora-layers", type=int, default=4, help="LoRA layers")
    p_sweep.add_argument("--steps", type=int, default=30, help="Training steps per LR point")
    p_sweep.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR")
    p_sweep.add_argument("--lr-max", type=float, default=1e-3, help="Maximum LR")
    p_sweep.add_argument("--points-per-decade", type=int, default=4, help="Grid points per decade")
    p_sweep.add_argument("--output", default="results/sweep_lr", help="Output directory")
    p_sweep.add_argument("--json", action="store_true", help="Save results as JSON")
    p_sweep.set_defaults(func=_cmd_sweep_lr)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Defaults to `validate` if no command is provided."""
    parser = build_parser()
    # If no args given, behave like `textpolicy validate` for a quick health check
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["validate"]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

