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


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser.

    We avoid subcommand bloat; a single `validate` command covers health checks.
    Default behavior is `validate` when no subcommand is provided for convenience.
    """
    parser = argparse.ArgumentParser(prog="textpolicy", description="TextPolicy command-line tools")
    subparsers = parser.add_subparsers(dest="command")

    p_validate = subparsers.add_parser("validate", help="Validate installation and environment")
    p_validate.add_argument("--json", action="store_true", help="Output machine-readable JSON report")
    p_validate.set_defaults(func=_cmd_validate)

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

