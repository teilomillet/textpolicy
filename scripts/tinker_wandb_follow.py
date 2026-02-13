#!/usr/bin/env python3
"""
Tail a Tinker metrics JSONL file and mirror it to Weights & Biases.

This lets us monitor long runs live in W&B even when the training process
doesn't emit W&B logs directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Follow Tinker metrics.jsonl and stream to W&B",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        required=True,
        help="Path to metrics.jsonl",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="textpolicy-tinker",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (optional)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="W&B run group (optional)",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=5.0,
        help="Polling interval while following",
    )
    parser.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=600.0,
        help="Stop after this many idle seconds with no new metrics",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Backfill once and exit (do not tail)",
    )
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _log_rows(
    rows: Iterable[Dict],
    seen_steps: Set[int],
) -> int:
    logged = 0
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        if step in seen_steps:
            continue
        seen_steps.add(step)

        payload = {f"tinker/{k}": v for k, v in row.items() if k != "step"}
        payload["tinker/step"] = step
        wandb.log(payload, step=step)
        logged += 1
    return logged


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics_path

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        group=args.group,
        config={
            "source": "tinker_metrics_jsonl_follower",
            "metrics_path": str(metrics_path),
            "poll_seconds": args.poll_seconds,
            "idle_timeout_seconds": args.idle_timeout_seconds,
            "follow": not args.no_follow,
        },
    )

    seen_steps: Set[int] = set()
    idle_started: Optional[float] = None

    # Initial backfill
    backfilled = _log_rows(_iter_jsonl(metrics_path), seen_steps)
    print(f"[follower] Backfilled {backfilled} steps from {metrics_path}")
    if args.no_follow:
        wandb.finish()
        return 0

    while True:
        logged_now = _log_rows(_iter_jsonl(metrics_path), seen_steps)
        if logged_now > 0:
            idle_started = None
            last_step = max(seen_steps) if seen_steps else -1
            print(f"[follower] Logged {logged_now} new steps (latest step={last_step})")
        else:
            if idle_started is None:
                idle_started = time.time()
            elif (time.time() - idle_started) >= args.idle_timeout_seconds:
                print(
                    f"[follower] Idle timeout reached ({args.idle_timeout_seconds}s). Exiting."
                )
                break
        time.sleep(args.poll_seconds)

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
