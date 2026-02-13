"""Tests for SEPA statistical significance analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textpolicy.analysis import (
    build_sepa_significance_markdown,
    evaluate_sepa_significance,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_run(
    tmp_path: Path,
    name: str,
    *,
    steps: int,
    gens: int,
    step_reward: float,
    gen_reward: float,
    gram_delta: float,
    correct_rate: float,
) -> Path:
    run_dir = tmp_path / name
    step_rows = [
        {
            "step": i,
            "mean_reward": step_reward,
            "gram_entropy_delta": gram_delta,
        }
        for i in range(steps)
    ]
    n_correct = int(round(gens * correct_rate))
    gen_rows = []
    for i in range(gens):
        gen_rows.append(
            {
                "step": i % max(steps, 1),
                "reward": gen_reward,
                "gram_entropy_delta": gram_delta,
                "metadata": {"correctness": i < n_correct},
            }
        )

    _write_jsonl(run_dir / "emergence" / "steps.jsonl", step_rows)
    _write_jsonl(run_dir / "emergence" / "generations.jsonl", gen_rows)
    return run_dir


@pytest.mark.unit
def test_significance_detects_clear_improvement(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=20,
        gens=200,
        step_reward=-0.4,
        gen_reward=-0.4,
        gram_delta=0.01,
        correct_rate=0.0,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=20,
        gens=200,
        step_reward=0.4,
        gen_reward=0.4,
        gram_delta=0.2,
        correct_rate=1.0,
    )

    report = evaluate_sepa_significance(
        [baseline],
        [candidate],
        num_resamples=2000,
        seed=0,
    )

    assert report.recommendation == "statistically_significant_improvement"
    assert "step_mean_reward" in report.significant_improvement_metrics
    assert "gen_correct_rate" in report.significant_improvement_metrics


@pytest.mark.unit
def test_significance_reports_insufficient_evidence_for_tiny_delta(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=12,
        gens=96,
        step_reward=-0.1,
        gen_reward=-0.1,
        gram_delta=0.03,
        correct_rate=0.02,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=12,
        gens=96,
        step_reward=-0.09,
        gen_reward=-0.09,
        gram_delta=0.031,
        correct_rate=0.03,
    )

    report = evaluate_sepa_significance(
        [baseline],
        [candidate],
        num_resamples=2000,
        seed=1,
    )

    assert report.recommendation == "insufficient_statistical_evidence"
    assert report.rate_test.significant_improvement is False


@pytest.mark.unit
def test_significance_markdown_contains_sections(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=6,
        gens=48,
        step_reward=-0.2,
        gen_reward=-0.2,
        gram_delta=0.01,
        correct_rate=0.0,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=6,
        gens=48,
        step_reward=0.2,
        gen_reward=0.2,
        gram_delta=0.05,
        correct_rate=0.5,
    )
    report = evaluate_sepa_significance(
        [baseline],
        [candidate],
        num_resamples=1000,
        seed=2,
    )
    md = build_sepa_significance_markdown(report)

    assert "SEPA Significance Report" in md
    assert "Mean-Difference Tests" in md
    assert "Correctness Rate Test" in md
