"""
Unit tests for standalone SEPA components.
"""

import pytest
import mlx.core as mx

from textpolicy.training.sepa import SEPAController, normalize_sepa_schedule


@pytest.mark.unit
class TestSEPAController:
    def test_normalize_schedule_rejects_invalid(self):
        with pytest.raises(ValueError, match="sepa_schedule"):
            normalize_sepa_schedule("invalid")

    def test_linear_schedule_lambda(self):
        sepa = SEPAController(sepa_steps=10, sepa_schedule="linear")

        assert sepa.enabled is True
        assert sepa.resolve_lambda(step=0.0) == 0.0
        assert sepa.resolve_lambda(step=5.0) == 0.5
        assert sepa.resolve_lambda(step=20.0) == 1.0

    def test_auto_schedule_lambda_increases_after_variance_drop(self):
        sepa = SEPAController(
            sepa_schedule="auto",
            sepa_ema_decay=0.0,
            sepa_var_threshold=0.5,
            sepa_warmup=1,
        )

        planning_mask = mx.array(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=mx.float32
        )
        high_var = mx.array(
            [2.0, 3.0, 1.0, 0.5, 4.5, 0.5, 4.5, 0.5, 4.5], dtype=mx.float32
        )
        low_var = mx.array(
            [2.0, 3.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=mx.float32
        )

        assert sepa.resolve_lambda(step=0.0) == 0.0
        sepa.update_auto_state(high_var, planning_mask)
        assert sepa.resolve_lambda(step=0.0) == 0.0

        sepa.update_auto_state(low_var, planning_mask)
        lam = sepa.resolve_lambda(step=0.0)
        assert lam > 0.0
        assert lam <= 1.0

    def test_auto_schedule_uses_linear_fallback_cap(self):
        sepa = SEPAController(
            sepa_schedule="auto",
            sepa_steps=3,  # cap at step >= 3
            sepa_warmup=50,  # keep auto branch inactive
        )
        assert sepa.resolve_lambda(step=0.0) == 0.0
        assert sepa.resolve_lambda(step=3.0) == 1.0

    def test_auto_schedule_lambda_decreases_after_variance_spike(self):
        sepa = SEPAController(
            sepa_schedule="auto",
            sepa_ema_decay=0.0,
            sepa_var_threshold=0.5,
            sepa_warmup=1,
        )

        planning_mask = mx.array([1, 1, 1, 0, 0, 0], dtype=mx.float32)
        high_var = mx.array([2.0, 3.0, 1.0, 0.0, 4.0, 0.0], dtype=mx.float32)
        low_var = mx.array([2.0, 3.0, 1.0, 2.0, 2.0, 2.0], dtype=mx.float32)

        sepa.update_auto_state(high_var, planning_mask)
        assert sepa.resolve_lambda(step=0.0) == 0.0

        sepa.update_auto_state(low_var, planning_mask)
        lam_after_drop = sepa.resolve_lambda(step=0.0)
        assert lam_after_drop > 0.0

        sepa.update_auto_state(high_var, planning_mask)
        lam_after_spike = sepa.resolve_lambda(step=0.0)
        assert lam_after_spike < lam_after_drop
        assert lam_after_spike == 0.0

    def test_apply_pooling_boundaries(self):
        sepa = SEPAController(sepa_steps=10)
        entropies = mx.array([2.0, 3.0, 1.0, 4.0, 1.0, 2.0], dtype=mx.float32)
        planning_mask = mx.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=mx.float32)

        out0 = sepa.apply(entropies, planning_mask, lambda_t=0.0)
        out1 = sepa.apply(entropies, planning_mask, lambda_t=1.0)
        mx.eval(out0, out1)

        assert mx.allclose(out0, entropies, atol=1e-6)

        expected_exec_mean = (4.0 + 1.0 + 2.0) / 3.0
        out1_list = out1.tolist()
        assert abs(out1_list[0] - 2.0) < 1e-6
        assert abs(out1_list[1] - 3.0) < 1e-6
        assert abs(out1_list[2] - 1.0) < 1e-6
        assert abs(out1_list[3] - expected_exec_mean) < 1e-6
        assert abs(out1_list[4] - expected_exec_mean) < 1e-6
        assert abs(out1_list[5] - expected_exec_mean) < 1e-6

    def test_prepare_batch_injects_lambda(self):
        sepa = SEPAController(sepa_steps=4, sepa_schedule="linear")
        batch = {"step": 2}
        sepa.prepare_batch(batch)
        assert batch["sepa_lambda"] == 0.5

    def test_state_dict_roundtrip_preserves_auto_state(self):
        sepa = SEPAController(
            sepa_schedule="auto",
            sepa_ema_decay=0.0,
            sepa_var_threshold=0.5,
            sepa_warmup=1,
        )
        planning_mask = mx.array([1, 1, 1, 0, 0, 0], dtype=mx.float32)
        high_var = mx.array([2.0, 3.0, 1.0, 0.0, 4.0, 0.0], dtype=mx.float32)
        low_var = mx.array([2.0, 3.0, 1.0, 2.0, 2.0, 2.0], dtype=mx.float32)

        sepa.update_auto_state(high_var, planning_mask)
        sepa.update_auto_state(low_var, planning_mask)
        expected_lambda = sepa.resolve_lambda(step=0.0)

        restored = SEPAController(sepa_schedule="auto")
        restored.load_state_dict(sepa.state_dict())
        assert restored.resolve_lambda(step=0.0) == pytest.approx(expected_lambda)
