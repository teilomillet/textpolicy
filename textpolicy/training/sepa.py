# textpolicy/training/sepa.py
"""
SEPA: Selective Entropy Pooling with Annealing.

This module is intentionally independent from any specific RL objective
(GTPO/GRPO/etc.). It provides reusable scheduling/state + entropy pooling
primitives that can be composed into different training transforms.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import mlx.core as mx  # type: ignore


def normalize_sepa_schedule(sepa_schedule: str) -> str:
    """Normalize and validate schedule mode."""
    normalized = sepa_schedule.lower().strip()
    if normalized not in {"linear", "auto"}:
        raise ValueError(
            f"sepa_schedule must be 'linear' or 'auto', got {sepa_schedule!r}."
        )
    return normalized


class SEPAController:
    """Reusable SEPA scheduler + entropy pooling controller.

    This class owns SEPA schedule state only. It does not depend on GTPO
    equations, reward shaping, or advantage normalization.
    """

    def __init__(
        self,
        *,
        sepa_steps: int = 0,
        sepa_schedule: str = "linear",
        sepa_ema_decay: float = 0.99,
        sepa_var_threshold: float = 0.2,
        sepa_warmup: int = 50,
        eps: float = 1e-8,
    ) -> None:
        if sepa_steps < 0:
            raise ValueError(f"sepa_steps must be >= 0, got {sepa_steps}.")
        if not (0.0 <= sepa_ema_decay <= 1.0):
            raise ValueError(
                f"sepa_ema_decay must be in [0, 1], got {sepa_ema_decay}."
            )
        if sepa_var_threshold <= 0.0:
            raise ValueError(
                f"sepa_var_threshold must be > 0, got {sepa_var_threshold}."
            )
        if sepa_warmup < 1:
            raise ValueError(f"sepa_warmup must be >= 1, got {sepa_warmup}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.sepa_steps = sepa_steps
        self.sepa_schedule = normalize_sepa_schedule(sepa_schedule)
        self.sepa_ema_decay = sepa_ema_decay
        self.sepa_var_threshold = sepa_var_threshold
        self.sepa_warmup = sepa_warmup
        self.eps = eps

        # Auto schedule state.
        self._var_ema: Optional[float] = None
        self._var_0: Optional[float] = None
        self._warmup_seen = 0

    @property
    def enabled(self) -> bool:
        """Whether SEPA pooling is active."""
        return self.sepa_steps > 0 or self.sepa_schedule == "auto"

    @property
    def requires_postprocess(self) -> bool:
        """Whether this schedule needs post-batch state updates."""
        return self.sepa_schedule == "auto"

    def _linear_lambda(self, *, step: float) -> float:
        if self.sepa_steps <= 0 or step <= 0.0:
            return 0.0
        return min(step / float(self.sepa_steps), 1.0)

    def _auto_lambda(self) -> float:
        if self.sepa_schedule != "auto":
            return 0.0
        if self._var_ema is None or self._var_0 is None:
            return 0.0

        threshold = max(self.sepa_var_threshold, self.eps)
        ratio = self._var_ema / max(self._var_0, self.eps)
        ratio = max(ratio, 0.0)
        return 1.0 - min(ratio / threshold, 1.0)

    def resolve_lambda(self, *, step: float) -> float:
        """Resolve λ using schedule mode and optional linear fallback cap."""
        linear_lambda = self._linear_lambda(step=step)
        if self.sepa_schedule == "auto":
            # One-sided cap: guarantees convergence even if auto signal stalls.
            return max(self._auto_lambda(), linear_lambda)
        return linear_lambda

    def prepare_batch(self, batch_data: Dict[str, Any]) -> None:
        """Inject precomputed SEPA λ for this batch."""
        if not self.enabled:
            return
        step = float(batch_data.get("step", 0))
        batch_data["sepa_lambda"] = self.resolve_lambda(step=step)

    def apply(
        self,
        flat_entropies: mx.array,
        planning_mask: mx.array,
        *,
        lambda_t: Optional[float] = None,
    ) -> mx.array:
        """Apply SEPA pooling to entropies, independent of downstream objective."""
        if not self.enabled:
            return flat_entropies
        if flat_entropies.shape != planning_mask.shape:
            raise ValueError(
                f"token_entropies shape {flat_entropies.shape} does not match "
                f"planning_mask shape {planning_mask.shape}."
            )

        if lambda_t is None:
            lambda_t = 0.0
        lambda_t = min(max(float(lambda_t), 0.0), 1.0)

        mask = mx.stop_gradient(planning_mask.astype(mx.float32))
        exec_mask = 1.0 - mask
        exec_count = mx.maximum(mx.sum(exec_mask), mx.array(1.0))
        mean_H_exec = mx.sum(flat_entropies * exec_mask) / exec_count

        return mx.where(
            mask.astype(mx.bool_),
            flat_entropies,
            lambda_t * mean_H_exec + (1.0 - lambda_t) * flat_entropies,
        )

    def update_auto_state(
        self,
        flat_entropies: mx.array,
        planning_mask: mx.array,
    ) -> None:
        """Update auto-schedule state from flat entropy + planning arrays."""
        if self.sepa_schedule != "auto":
            return
        if flat_entropies.shape != planning_mask.shape:
            raise ValueError(
                f"token_entropies shape {flat_entropies.shape} does not match "
                f"planning_mask shape {planning_mask.shape}."
            )
        if flat_entropies.size == 0:
            return

        mask = planning_mask.astype(mx.float32)
        exec_mask = 1.0 - mask

        exec_count_arr = mx.sum(exec_mask)
        mx.eval(exec_count_arr)
        exec_count = float(exec_count_arr.item())
        if exec_count <= 0.0:
            # No execution tokens in this batch; retain prior schedule state.
            return

        exec_count_safe = mx.maximum(exec_count_arr, mx.array(1.0))
        mean_H_exec = mx.sum(flat_entropies * exec_mask) / exec_count_safe
        var_batch_arr = mx.sum(exec_mask * (flat_entropies - mean_H_exec) ** 2) / exec_count_safe
        mx.eval(var_batch_arr)
        var_batch = float(var_batch_arr.item())
        if not math.isfinite(var_batch):
            return

        if self._var_ema is None:
            self._var_ema = var_batch
        else:
            d = self.sepa_ema_decay
            self._var_ema = d * self._var_ema + (1.0 - d) * var_batch

        if self._var_0 is None:
            self._warmup_seen += 1
            if self._warmup_seen >= self.sepa_warmup:
                self._var_0 = max(self._var_ema, self.eps)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize scheduler configuration + runtime state."""
        return {
            "sepa_steps": self.sepa_steps,
            "sepa_schedule": self.sepa_schedule,
            "sepa_ema_decay": self.sepa_ema_decay,
            "sepa_var_threshold": self.sepa_var_threshold,
            "sepa_warmup": self.sepa_warmup,
            "eps": self.eps,
            "var_ema": self._var_ema,
            "var_0": self._var_0,
            "warmup_seen": self._warmup_seen,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore runtime state from a state_dict payload."""
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dict, got {type(state)!r}.")

        def _maybe_float(key: str) -> Optional[float]:
            value = state.get(key)
            if value is None:
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"state[{key!r}] must be float or None.") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"state[{key!r}] must be finite, got {parsed}.")
            return parsed

        var_ema = _maybe_float("var_ema")
        var_0 = _maybe_float("var_0")

        warmup_seen = state.get("warmup_seen", self._warmup_seen)
        try:
            warmup_seen = int(warmup_seen)
        except (TypeError, ValueError) as exc:
            raise ValueError("state['warmup_seen'] must be an integer.") from exc
        if warmup_seen < 0:
            raise ValueError(
                f"state['warmup_seen'] must be >= 0, got {warmup_seen}."
            )

        self._var_ema = var_ema
        self._var_0 = var_0
        self._warmup_seen = warmup_seen
