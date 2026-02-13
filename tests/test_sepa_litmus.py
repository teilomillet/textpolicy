"""Tests for SEPA litmus threshold/evidence evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from textpolicy.analysis import (
    OFFICIAL_SEPA_LITMUS_PROFILE,
    SEPALitmusEvidence,
    SEPALitmusThresholds,
    build_litmus_markdown,
    evaluate_sepa_litmus,
    get_sepa_litmus_profile,
    load_emergence_run_stats,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_run(
    tmp_path: Path,
    name: str,
    *,
    steps: int,
    generations: int,
    mean_reward: float,
    correct_per_step: int,
    total_per_step: int,
    gram_delta: float,
    gram_pairs_per_step: int,
    planning_ratio: float = 0.02,
    sepa_lambda: float | None = None,
) -> Path:
    run_dir = tmp_path / name
    step_records = []
    for step in range(steps):
        rec = {
            "step": step,
            "mean_reward": mean_reward,
            "planning_token_ratio": planning_ratio,
            "gram_entropy_delta": gram_delta,
            "gram_entropy_pair_count": gram_pairs_per_step,
            "correct_count": correct_per_step,
            "total_count": total_per_step,
        }
        if sepa_lambda is not None:
            rec["sepa_lambda"] = sepa_lambda
        step_records.append(rec)

    generation_records = [
        {"step": idx % max(steps, 1), "completion": f"sample-{idx}"}
        for idx in range(generations)
    ]

    _write_jsonl(run_dir / "emergence" / "steps.jsonl", step_records)
    _write_jsonl(run_dir / "emergence" / "generations.jsonl", generation_records)
    return run_dir


@pytest.mark.unit
def test_load_emergence_run_stats_reads_expected_fields(tmp_path):
    run_dir = _make_run(
        tmp_path,
        "candidate_run",
        steps=10,
        generations=40,
        mean_reward=0.15,
        correct_per_step=2,
        total_per_step=4,
        gram_delta=0.12,
        gram_pairs_per_step=4,
        sepa_lambda=0.5,
    )

    stats = load_emergence_run_stats(run_dir)

    assert stats.num_steps == 10
    assert stats.num_generations == 40
    assert stats.mean_reward_steps == pytest.approx(0.15)
    assert stats.overall_correct_rate == pytest.approx(0.5)
    assert stats.mean_gram_entropy_delta == pytest.approx(0.12)
    assert stats.gram_pairs == 40
    assert stats.mean_sepa_lambda == pytest.approx(0.5)


@pytest.mark.unit
def test_evaluate_sepa_litmus_confirmed_when_thresholds_pass(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=12,
        generations=96,
        mean_reward=-0.12,
        correct_per_step=1,
        total_per_step=8,
        gram_delta=0.02,
        gram_pairs_per_step=8,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=12,
        generations=96,
        mean_reward=-0.08,
        correct_per_step=2,
        total_per_step=8,
        gram_delta=0.04,
        gram_pairs_per_step=8,
        sepa_lambda=0.6,
    )

    result = evaluate_sepa_litmus(
        baseline_run_dirs=[baseline],
        candidate_run_dirs=[candidate],
        thresholds=SEPALitmusThresholds(
            min_mean_reward_lift=0.01,
            min_correct_rate_lift=0.01,
            min_gram_entropy_delta_lift=0.01,
        ),
        evidence=SEPALitmusEvidence(
            min_run_pairs=1,
            min_steps_per_run=8,
            min_generations_per_run=32,
            min_gram_pairs_per_run=8,
        ),
    )

    assert result.status == "CONFIRMED"
    assert result.evidence_ok is True
    assert all(check.passed for check in result.checks)


@pytest.mark.unit
def test_evaluate_sepa_litmus_failed_when_threshold_missed(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=12,
        generations=96,
        mean_reward=-0.10,
        correct_per_step=2,
        total_per_step=8,
        gram_delta=0.03,
        gram_pairs_per_step=8,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=12,
        generations=96,
        mean_reward=-0.105,
        correct_per_step=2,
        total_per_step=8,
        gram_delta=0.031,
        gram_pairs_per_step=8,
        sepa_lambda=0.5,
    )

    result = evaluate_sepa_litmus(
        baseline_run_dirs=[baseline],
        candidate_run_dirs=[candidate],
        thresholds=SEPALitmusThresholds(
            min_mean_reward_lift=0.01,
            min_correct_rate_lift=0.001,
            min_gram_entropy_delta_lift=0.001,
        ),
        evidence=SEPALitmusEvidence(),
    )

    assert result.status == "FAILED"
    assert result.evidence_ok is True
    assert any(check.metric == "mean_reward_steps" and not check.passed for check in result.checks)


@pytest.mark.unit
def test_evaluate_sepa_litmus_inconclusive_when_evidence_insufficient(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline_small",
        steps=4,
        generations=16,
        mean_reward=-0.05,
        correct_per_step=1,
        total_per_step=4,
        gram_delta=0.01,
        gram_pairs_per_step=2,
    )
    candidate = _make_run(
        tmp_path,
        "candidate_small",
        steps=4,
        generations=16,
        mean_reward=0.02,
        correct_per_step=1,
        total_per_step=4,
        gram_delta=0.03,
        gram_pairs_per_step=2,
        sepa_lambda=0.25,
    )

    result = evaluate_sepa_litmus(
        baseline_run_dirs=[baseline],
        candidate_run_dirs=[candidate],
        evidence=SEPALitmusEvidence(
            min_run_pairs=1,
            min_steps_per_run=8,
            min_generations_per_run=32,
            min_gram_pairs_per_run=8,
        ),
    )

    assert result.status == "INCONCLUSIVE"
    assert result.evidence_ok is False
    assert result.evidence_failures


@pytest.mark.unit
def test_build_litmus_markdown_includes_status_and_checks(tmp_path):
    baseline = _make_run(
        tmp_path,
        "baseline",
        steps=8,
        generations=32,
        mean_reward=-0.1,
        correct_per_step=1,
        total_per_step=4,
        gram_delta=0.01,
        gram_pairs_per_step=4,
    )
    candidate = _make_run(
        tmp_path,
        "candidate",
        steps=8,
        generations=32,
        mean_reward=0.0,
        correct_per_step=2,
        total_per_step=4,
        gram_delta=0.03,
        gram_pairs_per_step=4,
        sepa_lambda=0.5,
    )

    result = evaluate_sepa_litmus([baseline], [candidate])
    md = build_litmus_markdown(result)

    assert "SEPA Litmus Result" in md
    assert "Threshold Checks" in md
    assert result.status in md


@pytest.mark.unit
def test_get_sepa_litmus_profile_returns_official_profile():
    profile = get_sepa_litmus_profile("official_v1")
    assert profile.name == OFFICIAL_SEPA_LITMUS_PROFILE.name
    assert profile.thresholds == OFFICIAL_SEPA_LITMUS_PROFILE.thresholds
    assert profile.evidence == OFFICIAL_SEPA_LITMUS_PROFILE.evidence


@pytest.mark.unit
def test_get_sepa_litmus_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown SEPA litmus profile"):
        get_sepa_litmus_profile("v2")
