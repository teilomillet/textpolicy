"""Unit tests for generation performance target checks in profile_hardware."""

from __future__ import annotations

import pytest

from experiments.profile_hardware import (
    ProfileConfig,
    ProbeResult,
    evaluate_generation_target,
)


def _ok_probe(
    *,
    seq_length: int = 512,
    gen_time_s: float = 2.0,
    train_time_s: float = 3.0,
    rollout_generation_s: float | None = 1.2,
    total_tokens_generated: int = 1000,
    steps_completed: int = 2,
) -> ProbeResult:
    rollout_phases = None
    if rollout_generation_s is not None:
        rollout_phases = {
            "generation": rollout_generation_s,
            "env_reset": 0.1,
            "env_step": 0.1,
            "buffer_store": 0.1,
            "total": 1.5,
        }
    return ProbeResult(
        seq_length=seq_length,
        group_size=8,
        gen_time_s=gen_time_s,
        train_time_s=train_time_s,
        total_time_s=gen_time_s + train_time_s,
        peak_memory_mb=1024.0,
        status="OK",
        rollout_phases=rollout_phases,
        trainer_phases={"total": train_time_s},
        total_tokens_generated=total_tokens_generated,
        steps_completed=steps_completed,
    )


@pytest.mark.unit
def test_generation_target_passes_when_all_thresholds_met():
    cfg = ProfileConfig(
        target_seq_length=512,
        target_gen_tps_min=200.0,
        target_gen_fraction_max=0.30,
        target_gen_time_max_s=1.5,
    )
    check = evaluate_generation_target(cfg, [_ok_probe()])

    assert check.enabled is True
    assert check.passed is True
    assert check.evaluated_seq_length == 512
    assert check.failures == []
    assert check.observed_gen_tps == pytest.approx(250.0)
    assert check.observed_gen_fraction == pytest.approx(0.24)
    assert check.observed_gen_time_s == pytest.approx(1.2)


@pytest.mark.unit
def test_generation_target_fails_when_thresholds_missed():
    cfg = ProfileConfig(
        target_seq_length=512,
        target_gen_tps_min=300.0,
        target_gen_fraction_max=0.20,
        target_gen_time_max_s=1.0,
    )
    check = evaluate_generation_target(cfg, [_ok_probe()])

    assert check.enabled is True
    assert check.passed is False
    assert any("gen_tps" in msg for msg in check.failures)
    assert any("gen_fraction" in msg for msg in check.failures)
    assert any("generation_s" in msg for msg in check.failures)


@pytest.mark.unit
def test_generation_target_fails_when_requested_seq_missing():
    cfg = ProfileConfig(
        target_seq_length=1024,
        target_gen_tps_min=100.0,
    )
    check = evaluate_generation_target(cfg, [_ok_probe(seq_length=512)])

    assert check.enabled is True
    assert check.passed is False
    assert check.evaluated_seq_length == 1024
    assert check.failures == ["No successful probe at seq_length=1024."]


@pytest.mark.unit
def test_generation_target_uses_gen_time_fallback_without_rollout_split():
    cfg = ProfileConfig(
        target_gen_time_max_s=2.5,
        target_gen_fraction_max=0.45,
    )
    probe = _ok_probe(rollout_generation_s=None)
    check = evaluate_generation_target(cfg, [probe])

    assert check.enabled is True
    assert check.passed is True
    assert check.observed_gen_time_s == pytest.approx(2.0)
    assert check.observed_gen_fraction == pytest.approx(0.4)


@pytest.mark.unit
def test_generation_target_disabled_when_no_thresholds_configured():
    cfg = ProfileConfig()
    check = evaluate_generation_target(cfg, [_ok_probe()])

    assert check.enabled is False
    assert check.passed is True
    assert check.failures == []
