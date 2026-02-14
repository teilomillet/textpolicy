# textpolicy/tinker/sepa.py
"""
SEPA Controller for Tinker GPU training (pure Python, no MLX).

Ports the scheduling and state-tracking logic from
textpolicy/training/sepa.py into a framework-agnostic form.
The actual entropy pooling is done by advantages.apply_sepa_pooling();
this class handles when and how much to pool.

Source reference: textpolicy/training/sepa.py:28-283
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _normalize_schedule(schedule: str) -> str:
    """Normalize and validate schedule mode."""
    normalized = schedule.lower().strip()
    if normalized not in {"linear", "auto"}:
        raise ValueError(
            f"sepa_schedule must be 'linear' or 'auto', got {schedule!r}."
        )
    return normalized


class SEPAController:
    """SEPA scheduler: controls pooling strength over training.

    Two scheduling modes:
        linear — λ ramps from 0 to 1 over sepa_steps (after optional delay).
        auto   — λ adapts based on execution-token entropy variance decay,
                 with linear as a fallback floor.

    Optional correctness gate: SEPA stays disabled (λ=0) until the model
    achieves a minimum correct rate, then becomes sticky-open.

    This is a pure-Python port of textpolicy.training.sepa.SEPAController.
    """

    def __init__(
        self,
        *,
        sepa_steps: int = 0,
        sepa_schedule: str = "linear",
        sepa_delay_steps: int = 0,
        sepa_correct_rate_gate: float = 0.0,
        sepa_ema_decay: float = 0.99,
        sepa_var_threshold: float = 0.2,
        sepa_warmup: int = 50,
        eps: float = 1e-8,
    ) -> None:
        if sepa_steps < 0:
            raise ValueError(f"sepa_steps must be >= 0, got {sepa_steps}.")
        if sepa_delay_steps < 0:
            raise ValueError(
                f"sepa_delay_steps must be >= 0, got {sepa_delay_steps}."
            )
        if not (0.0 <= sepa_correct_rate_gate <= 1.0):
            raise ValueError(
                f"sepa_correct_rate_gate must be in [0, 1], "
                f"got {sepa_correct_rate_gate}."
            )
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
        self.sepa_schedule = _normalize_schedule(sepa_schedule)
        self.sepa_delay_steps = sepa_delay_steps
        self.sepa_correct_rate_gate = sepa_correct_rate_gate
        self.sepa_ema_decay = sepa_ema_decay
        self.sepa_var_threshold = sepa_var_threshold
        self.sepa_warmup = sepa_warmup
        self.eps = eps

        # Auto schedule state
        self._var_ema: Optional[float] = None
        self._var_0: Optional[float] = None
        self._warmup_seen: int = 0
        self._gate_open: bool = sepa_correct_rate_gate <= 0.0

    @property
    def enabled(self) -> bool:
        """Whether SEPA pooling is active."""
        return self.sepa_steps > 0 or self.sepa_schedule == "auto"

    @property
    def gate_open(self) -> bool:
        """Whether correctness gate currently allows non-zero lambda."""
        return self._gate_open

    def observe_correct_rate(self, correct_rate: Optional[float]) -> None:
        """Update correctness gate from observed correct rate.

        Gate is sticky-open: once threshold is met, SEPA stays enabled.
        """
        if self._gate_open or self.sepa_correct_rate_gate <= 0.0:
            return
        if correct_rate is None:
            return
        rate = float(correct_rate)
        if not math.isfinite(rate):
            return
        if rate >= self.sepa_correct_rate_gate:
            self._gate_open = True

    def _linear_lambda(self, step: float) -> float:
        if self.sepa_steps <= 0:
            return 0.0
        shifted = step - float(self.sepa_delay_steps)
        if shifted <= 0.0:
            return 0.0
        return min(shifted / float(self.sepa_steps), 1.0)

    def _auto_lambda(self) -> float:
        if self.sepa_schedule != "auto":
            return 0.0
        if self._var_ema is None or self._var_0 is None:
            return 0.0

        threshold = max(self.sepa_var_threshold, self.eps)
        ratio = self._var_ema / max(self._var_0, self.eps)
        ratio = max(ratio, 0.0)
        return 1.0 - min(ratio / threshold, 1.0)

    def resolve_lambda(self, step: float) -> float:
        """Resolve current pooling strength lambda.

        Args:
            step: Current training step.

        Returns:
            Lambda in [0, 1]. 0 = no pooling, 1 = full pooling.
        """
        linear_val = self._linear_lambda(step)
        if not self._gate_open:
            return 0.0
        if self.sepa_schedule == "auto":
            return max(self._auto_lambda(), linear_val)
        return linear_val

    def update_auto_state(
        self,
        exec_entropies: list[float],
    ) -> None:
        """Update auto-schedule variance tracking from execution entropies.

        Call this after each batch with the execution-token entropies
        (non-planning tokens only).

        Args:
            exec_entropies: List of entropy values for execution tokens.
        """
        if self.sepa_schedule != "auto":
            return
        if not exec_entropies:
            return

        n = len(exec_entropies)
        mean_h = sum(exec_entropies) / n
        var_batch = sum((h - mean_h) ** 2 for h in exec_entropies) / n

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
        """Serialize scheduler state for checkpointing."""
        return {
            "sepa_steps": self.sepa_steps,
            "sepa_schedule": self.sepa_schedule,
            "sepa_delay_steps": self.sepa_delay_steps,
            "sepa_correct_rate_gate": self.sepa_correct_rate_gate,
            "sepa_ema_decay": self.sepa_ema_decay,
            "sepa_var_threshold": self.sepa_var_threshold,
            "sepa_warmup": self.sepa_warmup,
            "eps": self.eps,
            "var_ema": self._var_ema,
            "var_0": self._var_0,
            "warmup_seen": self._warmup_seen,
            "gate_open": self._gate_open,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dict, got {type(state)!r}.")

        def _maybe_float(key: str) -> Optional[float]:
            value = state.get(key)
            if value is None:
                return None
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"state[{key!r}] must be finite, got {parsed}.")
            return parsed

        self._var_ema = _maybe_float("var_ema")
        self._var_0 = _maybe_float("var_0")

        warmup_seen = state.get("warmup_seen", self._warmup_seen)
        self._warmup_seen = int(warmup_seen)
        if self._warmup_seen < 0:
            raise ValueError(
                f"state['warmup_seen'] must be >= 0, got {self._warmup_seen}."
            )

        gate_open = state.get("gate_open", self._gate_open)
        if not isinstance(gate_open, bool):
            raise ValueError("state['gate_open'] must be a boolean.")
        self._gate_open = gate_open
