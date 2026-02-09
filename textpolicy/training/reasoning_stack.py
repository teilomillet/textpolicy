# textpolicy/training/reasoning_stack.py
"""
Composable transform builders for GTPO + HICRA advantage shaping.

- GTPO entropy-weighted advantages (arXiv:2508.04349)
- HICRA planning-token amplification (arXiv:2509.03646)

These builders return Trainer-compatible transforms that can be injected
via ``Trainer(advantage_transform_fn=...)``.  Experiment scripts compose
LoRA setup + transform + Trainer directly (see ``experiments/`` for examples).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import mlx.core as mx  # type: ignore
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore

from textpolicy.algorithms.grpo import (
    apply_entropy_weighting,
    compute_advantages,
    compute_gtpo_shaped_rewards,
    normalize_gtpo_advantages,
    policy_loss,
)
from textpolicy.algorithms.hicra import (
    apply_hicra_amplification,
    boost_entropy_with_planning,
    identify_planning_tokens,
)
from textpolicy.training.semantic_entropy import (
    SemanticEntropyTracker,
    build_prompt_group_keys,
)
from textpolicy.generation.lora import create_lora_setup
from textpolicy.training.sepa import SEPAController, normalize_sepa_schedule
from textpolicy.training.trainer import Trainer


_DEFAULT_STRATEGIC_GRAMS = [
    "let me think",
    "let me check",
    "on second thought",
    "another way",
    "the key is",
]


def _flatten_padded_token_rows(
    values: mx.array,
    episode_lengths: Optional[Sequence[int]],
    *,
    field_name: str,
) -> mx.array:
    """Flatten a [episodes, max_len] padded tensor into flat token layout."""
    if values.ndim == 1:
        return values

    if values.ndim != 2:
        raise ValueError(
            f"batch_data['{field_name}'] must be 1D or 2D, got {values.ndim}D."
        )

    if episode_lengths is None:
        raise ValueError(
            f"batch_data['{field_name}'] is 2D, so 'episode_lengths' is required."
        )

    if values.shape[0] != len(episode_lengths):  # type: ignore[arg-type]
        raise ValueError(
            f"batch_data['{field_name}'].shape[0]={values.shape[0]} does not match "
            f"len(episode_lengths)={len(episode_lengths)}."
        )

    max_len = values.shape[1]  # type: ignore[index]
    pieces: List[mx.array] = []
    for row_idx, row_len in enumerate(episode_lengths):
        if row_len < 0:
            raise ValueError(
                f"episode_lengths[{row_idx}] must be >= 0, got {row_len}."
            )
        if row_len == 0:
            continue
        if row_len > max_len:
            raise ValueError(
                f"episode_lengths[{row_idx}]={row_len} exceeds padded width {max_len} "
                f"for batch_data['{field_name}']."
            )
        pieces.append(values[row_idx, :row_len])

    if not pieces:
        return mx.array([], dtype=values.dtype)
    return mx.concatenate(pieces)


class _GTPOHICRATransform:
    """Trainer transform with optional eager batch preparation.

    ``identify_planning_tokens`` is intentionally eager/Pythonic and uses
    token decoding. The ``prepare_batch`` hook computes ``planning_mask``
    outside ``mx.compile`` so ``__call__`` stays compile-safe.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        strategic_grams: Optional[List[str]] = None,
        hicra_alpha: float = 0.2,
        entropy_weight: float = 0.1,
    ) -> None:
        self.tokenizer = tokenizer
        self.grams = list(strategic_grams or _DEFAULT_STRATEGIC_GRAMS)
        self.hicra_alpha = hicra_alpha
        self.entropy_weight = entropy_weight

    def _flatten_actions(self, batch_data: Dict[str, Any]) -> mx.array:
        episode_lengths = batch_data.get("episode_lengths")
        actions = batch_data.get("act")
        if actions is None:
            raise ValueError(
                "batch_data must include 'act' for HICRA token matching."
            )
        return _flatten_padded_token_rows(
            actions,
            episode_lengths,
            field_name="act",
        )

    def prepare_batch(self, batch_data: Dict[str, Any]) -> None:
        """Eagerly compute planning_mask for compile-safe training."""
        if self.hicra_alpha == 0.0 or not self.grams:
            return
        if batch_data.get("planning_mask") is not None:
            return
        token_ids = self._flatten_actions(batch_data)
        batch_data["planning_mask"] = identify_planning_tokens(
            token_ids, self.tokenizer, self.grams
        )

    def __call__(self, advantages: mx.array, batch_data: Dict[str, Any]) -> mx.array:
        episode_lengths = batch_data.get("episode_lengths")
        token_ids = self._flatten_actions(batch_data)
        if token_ids.shape != advantages.shape:
            raise ValueError(
                f"Flattened action token shape {token_ids.shape} does not match "
                f"advantages shape {advantages.shape}."
            )

        transformed = advantages
        token_entropies = batch_data.get("token_entropies")
        if token_entropies is not None and self.entropy_weight > 0.0:
            flat_entropies = _flatten_padded_token_rows(
                token_entropies,
                episode_lengths,
                field_name="token_entropies",
            )
            if flat_entropies.shape != transformed.shape:
                raise ValueError(
                    f"Flattened token_entropies shape {flat_entropies.shape} does "
                    f"not match advantages shape {transformed.shape}."
                )
            transformed = apply_entropy_weighting(
                transformed,
                flat_entropies,
                entropy_weight=self.entropy_weight,
            )

        if self.hicra_alpha == 0.0 or not self.grams:
            return transformed

        planning_mask = batch_data.get("planning_mask")
        if planning_mask is None:
            try:
                planning_mask = identify_planning_tokens(
                    token_ids, self.tokenizer, self.grams
                )
            except ValueError as exc:
                if "Attempting to eval an array" in str(exc):
                    raise ValueError(
                        "HICRA planning_mask must be prepared outside mx.compile. "
                        "Use Trainer.train() with a transform that exposes "
                        "prepare_batch()."
                    ) from exc
                raise
            batch_data["planning_mask"] = planning_mask
        else:
            planning_mask = _flatten_padded_token_rows(
                planning_mask,
                episode_lengths,
                field_name="planning_mask",
            )

        if planning_mask.shape != transformed.shape:
            raise ValueError(
                f"HICRA planning_mask shape {planning_mask.shape} does not match "
                f"advantages shape {transformed.shape}."
            )
        return apply_hicra_amplification(
            transformed,
            planning_mask,
            alpha=self.hicra_alpha,
        )


def build_gtpo_hicra_transform(
    tokenizer: Any,
    *,
    strategic_grams: Optional[List[str]] = None,
    hicra_alpha: float = 0.2,
    entropy_weight: float = 0.1,
) -> Callable[[mx.array, Dict[str, Any]], mx.array]:
    """
    Build a Trainer-compatible transform that composes GTPO then HICRA.

    Expected batch_data keys:
    - 'act': token ids (flat 1D or padded 2D)
    - optional 'episode_lengths': required when 'act' is 2D
    - optional 'token_entropies': token entropy values for GTPO

    The transform preserves the input advantage shape exactly.
    """
    if entropy_weight < 0.0:
        raise ValueError(f"entropy_weight must be >= 0, got {entropy_weight}.")
    if hicra_alpha < 0.0:
        raise ValueError(f"hicra_alpha must be >= 0, got {hicra_alpha}.")

    return _GTPOHICRATransform(
        tokenizer,
        strategic_grams=strategic_grams,
        hicra_alpha=hicra_alpha,
        entropy_weight=entropy_weight,
    )


class _GTPOTransform:
    """Trainer transform for GTPO (arXiv 2508.04349).

    Computes entropy-shaped, separately-normalized advantages per Eq. 3, 5, 6,
    completely replacing the standard GRPO advantages.  All operations are
    compile-safe (no ``mx.eval`` or ``.item()``).

    Optionally fuses HICRA planning-token amplification.  Three modes:

    **Boost** (``hicra_gamma > 0``, ``sepa_steps == 0``):
        Adds ``gamma * mean(H)`` to planning token entropy before GTPO
        processes it (via ``boost_entropy_with_planning``).

    **SEPA (linear)** (``sepa_steps > 0``, ``sepa_schedule='linear'``):
        Pools execution token entropy toward the class mean, removing
        per-token variance without removing tokens from the credit budget.
        Planning tokens pass through untouched.  λ ∈ [0, 1] anneals
        linearly over ``sepa_steps`` training steps:

            H_exec(t) = λ · mean(H_exec) + (1 − λ) · H_real(t)

        At λ=0 (step 0): pure GTPO, no HICRA influence.
        At λ=1 (step ≥ sepa_steps): execution tokens are uniform,
        entropy gradient operates only among planning tokens.

    **SEPA (auto)** (``sepa_schedule='auto'``):
        Uses an EMA of execution-token entropy variance as a self-supervised
        schedule. During warmup, λ stays at 0. After warmup captures a
        baseline variance, λ rises as variance drops:

            λ_auto = 1 - clamp((var_ema / var_0) / threshold, 0, 1)

        Optional ``sepa_steps > 0`` acts as a one-sided fallback cap:

            λ = max(λ_auto, min(step / sepa_steps, 1))

        This preserves adaptivity while guaranteeing convergence.

    The PPO clipped objective (Eq. 7) is handled by the Trainer's ``loss_fn``
    (e.g. ``grpo.policy_loss``).
    """

    def __init__(
        self,
        *,
        alpha_1: float = 1.0,
        alpha_2: float = 0.1,
        reward_threshold: float = 0.5,
        eps: float = 1e-8,
        tokenizer: Any = None,
        strategic_grams: Optional[List[str]] = None,
        hicra_gamma: float = 0.0,
        sepa_steps: int = 0,
        sepa_schedule: str = "linear",
        sepa_ema_decay: float = 0.99,
        sepa_var_threshold: float = 0.2,
        sepa_warmup: int = 50,
        semantic_entropy: bool = False,
        semantic_entropy_ema_decay: float = 0.99,
        semantic_entropy_stability_tol: float = 1e-3,
        semantic_entropy_stability_patience: int = 20,
        semantic_entropy_hash_bins: int = 256,
        semantic_entropy_positive_only: bool = True,
        semantic_entropy_on_stable: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.reward_threshold = reward_threshold
        self.eps = eps
        self.tokenizer = tokenizer
        self.grams = (
            list(strategic_grams)
            if strategic_grams is not None
            else (list(_DEFAULT_STRATEGIC_GRAMS) if tokenizer is not None else None)
        )
        self.hicra_gamma = hicra_gamma
        self._sepa = SEPAController(
            sepa_steps=sepa_steps,
            sepa_schedule=sepa_schedule,
            sepa_ema_decay=sepa_ema_decay,
            sepa_var_threshold=sepa_var_threshold,
            sepa_warmup=sepa_warmup,
            eps=eps,
        )
        self._sepa_enabled = self._sepa.enabled

        self._semantic_tracker: Optional[SemanticEntropyTracker]
        if semantic_entropy:
            if tokenizer is None:
                raise ValueError(
                    "semantic_entropy=True requires a tokenizer for planning "
                    "token identification."
                )
            self._semantic_tracker = SemanticEntropyTracker(
                ema_decay=semantic_entropy_ema_decay,
                stability_tol=semantic_entropy_stability_tol,
                stability_patience=semantic_entropy_stability_patience,
                hash_bins=semantic_entropy_hash_bins,
                positive_only=semantic_entropy_positive_only,
                reward_threshold=reward_threshold,
                eps=eps,
                on_stable=semantic_entropy_on_stable,
            )
        else:
            self._semantic_tracker = None

        # HICRA is active when boost or SEPA is enabled.
        self._hicra_enabled = (hicra_gamma > 0.0 or self._sepa_enabled)
        self._needs_planning_mask = (
            self._hicra_enabled or self._semantic_tracker is not None
        )

    def _flatten_actions(self, batch_data: Dict[str, Any]) -> mx.array:
        episode_lengths = batch_data.get("episode_lengths")
        actions = batch_data.get("act")
        if actions is None:
            raise ValueError(
                "batch_data must include 'act' for HICRA token matching."
            )
        return _flatten_padded_token_rows(
            actions,
            episode_lengths,
            field_name="act",
        )

    def prepare_batch(self, batch_data: Dict[str, Any]) -> None:
        """Eagerly compute planning_mask for compile-safe training.

        Called by the Trainer before ``mx.compile``-traced execution.
        Short-circuits when HICRA fusion is disabled.
        """
        if (
            self._needs_planning_mask
            and self.tokenizer is not None
            and self.grams
            and batch_data.get("planning_mask") is None
        ):
            token_ids = self._flatten_actions(batch_data)
            batch_data["planning_mask"] = identify_planning_tokens(
                token_ids, self.tokenizer, self.grams
            )

        if self._sepa_enabled:
            self._sepa.prepare_batch(batch_data)

    def postprocess_batch(self, batch_data: Dict[str, Any]) -> None:
        """Update auto-SEPA state outside ``mx.compile``.

        Trainer calls this hook after ``loss_and_grad_fn`` so token entropies
        are available and scalar extraction (``.item()``) is safe.
        """
        if not self._sepa.requires_postprocess and self._semantic_tracker is None:
            return

        planning_mask = batch_data.get("planning_mask")
        episode_lengths = batch_data.get("episode_lengths")
        if planning_mask is None or episode_lengths is None:
            return

        if self._sepa.requires_postprocess:
            token_entropies = batch_data.get("token_entropies")
            if token_entropies is not None:
                flat_entropies = _flatten_padded_token_rows(
                    token_entropies,
                    episode_lengths,
                    field_name="token_entropies",
                )
                flat_mask = _flatten_padded_token_rows(
                    planning_mask,
                    episode_lengths,
                    field_name="planning_mask",
                ).astype(mx.float32)
                self._sepa.update_auto_state(flat_entropies, flat_mask)

        if self._semantic_tracker is None:
            return

        actions = batch_data.get("act")
        if actions is None:
            return

        prompt_keys = None
        obs = batch_data.get("obs")
        prompt_lengths = batch_data.get("prompt_lengths")
        if obs is not None and prompt_lengths is not None:
            try:
                prompt_keys = build_prompt_group_keys(obs, prompt_lengths)
            except ValueError:
                prompt_keys = None

        semantic_stats = self._semantic_tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=batch_data.get("rewards"),
            prompt_keys=prompt_keys,
        )
        if semantic_stats:
            batch_data["semantic_entropy_stats"] = semantic_stats
            transform_metrics = batch_data.get("transform_metrics")
            if not isinstance(transform_metrics, dict):
                transform_metrics = {}
                batch_data["transform_metrics"] = transform_metrics
            transform_metrics.update(semantic_stats)

    def state_dict(self) -> Dict[str, Any]:
        """Serialize SEPA/semantic tracker state for optional checkpointing."""
        state: Dict[str, Any] = {"sepa": self._sepa.state_dict()}
        if self._semantic_tracker is not None:
            state["semantic_entropy"] = self._semantic_tracker.state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore scheduler/tracker state from ``state_dict`` payload."""
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dict, got {type(state)!r}.")
        if "sepa" in state:
            self._sepa.load_state_dict(state["sepa"])
        if (
            self._semantic_tracker is not None
            and "semantic_entropy" in state
        ):
            self._semantic_tracker.load_state_dict(state["semantic_entropy"])

    def __call__(self, advantages: mx.array, batch_data: Dict[str, Any]) -> mx.array:
        episode_lengths = batch_data.get("episode_lengths")
        if episode_lengths is None:
            raise ValueError(
                "batch_data must include 'episode_lengths' for GTPO. "
                "Ensure the rollout pipeline provides episode_lengths in the batch."
            )

        token_entropies = batch_data.get("token_entropies")
        if token_entropies is None:
            raise ValueError(
                "batch_data must include 'token_entropies' for GTPO. "
                "This is auto-computed when advantage_transform_fn is set."
            )

        rewards = batch_data.get("rewards")
        if rewards is None:
            raise ValueError(
                "batch_data must include 'rewards' for GTPO. "
                "Ensure the rollout pipeline provides episode-level rewards."
            )

        # Defensively flatten token_entropies if 2D padded
        flat_entropies = _flatten_padded_token_rows(
            token_entropies,
            episode_lengths,
            field_name="token_entropies",
        )

        # Optional HICRA fusion: modify entropies using planning token mask
        if self._hicra_enabled and self.tokenizer is not None:
            planning_mask = batch_data.get("planning_mask")
            if planning_mask is None:
                # On-demand computation (mirrors _GTPOHICRATransform pattern).
                # This fails inside mx.compile with a clear message.
                try:
                    token_ids = self._flatten_actions(batch_data)
                    planning_mask = identify_planning_tokens(
                        token_ids, self.tokenizer, self.grams
                    )
                except ValueError as exc:
                    if "Attempting to eval an array" in str(exc):
                        raise ValueError(
                            "HICRA planning_mask must be prepared outside "
                            "mx.compile. Use Trainer.train() which calls "
                            "prepare_batch() automatically."
                        ) from exc
                    raise
                batch_data["planning_mask"] = planning_mask
            else:
                planning_mask = _flatten_padded_token_rows(
                    planning_mask,
                    episode_lengths,
                    field_name="planning_mask",
                )

            if self._sepa_enabled:
                # SEPA: Selective Entropy Pooling with Annealing.
                # Pool execution token entropy toward class mean; planning
                # tokens pass through.  λ comes from prepare_batch() so the
                # scheduling logic can remain compile-safe.
                lambda_t = batch_data.get("sepa_lambda")
                if lambda_t is None:
                    # Fallback for direct transform() usage without Trainer.
                    lambda_t = self._sepa.resolve_lambda(
                        step=float(batch_data.get("step", 0))
                    )
                    batch_data["sepa_lambda"] = lambda_t
                flat_entropies = self._sepa.apply(
                    flat_entropies,
                    planning_mask,
                    lambda_t=lambda_t,
                )
            else:
                # Boost mode (legacy): additive entropy boost at planning positions.
                flat_entropies = boost_entropy_with_planning(
                    flat_entropies, planning_mask, gamma=self.hicra_gamma,
                )

        # Eq. 3 + Eq. 5: entropy-shaped token-level rewards
        shaped_rewards, is_positive = compute_gtpo_shaped_rewards(
            rewards,
            flat_entropies,
            episode_lengths,
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            reward_threshold=self.reward_threshold,
            eps=self.eps,
        )

        # Eq. 6: separate O+/O- normalization
        return normalize_gtpo_advantages(
            shaped_rewards, is_positive, eps=self.eps,
        )


def build_gtpo_transform(
    *,
    alpha_1: float = 1.0,
    alpha_2: float = 0.1,
    reward_threshold: float = 0.5,
    eps: float = 1e-8,
    tokenizer: Any = None,
    strategic_grams: Optional[List[str]] = None,
    hicra_gamma: float = 0.0,
    sepa_steps: int = 0,
    sepa_schedule: str = "linear",
    semantic_entropy: bool = False,
) -> Callable[[mx.array, Dict[str, Any]], mx.array]:
    """
    Build a Trainer-compatible transform for GTPO (arXiv 2508.04349).

    Implements Eq. 3, 5, 6 from arXiv 2508.04349. The returned transform
    completely replaces standard GRPO advantages with entropy-shaped,
    separately-normalized advantages.

    PPO clipping (Eq. 7) is handled by the Trainer's ``loss_fn`` — use
    ``functools.partial(grpo.policy_loss, clip_ratio=0.2)`` or similar.

    **HICRA Fusion (optional):**  When enabled and a tokenizer
    is provided, the transform modifies per-token entropies *before* GTPO
    processes them.  Two modes are available:

    - **Boost** (default, ``sepa_steps == 0``): adds ``gamma * mean(H)``
      to planning token entropy.
    - **SEPA** — Selective Entropy Pooling with Annealing:
      pools execution token entropy toward the class mean, removing per-token
      variance without removing tokens from the credit budget.  Planning tokens
      pass through untouched.
      - ``sepa_schedule='linear'``: λ anneals linearly from 0 to 1 over
        ``sepa_steps``.
      - ``sepa_schedule='auto'``: λ follows an EMA of execution-entropy
        variance with warmup; optional ``sepa_steps`` acts as a fallback cap
        via ``max(lambda_auto, lambda_linear)``.
      At λ=1, execution tokens have uniform entropy (no individual
      differentiation), while planning tokens retain their real entropy for
      GTPO's Eq. 3/5 weighting.

    Expected batch_data keys (auto-populated by the Trainer):
    - ``'rewards'``: episode-level rewards [num_episodes]
    - ``'token_entropies'``: per-token entropy [total_tokens] (auto-computed)
    - ``'episode_lengths'``: token count per episode
    - ``'act'``: token IDs (required when boost/SEPA is active)
    - ``'step'``: training step count (auto-injected by Trainer, used for SEPA)
    - ``'sepa_lambda'``: optional precomputed λ (injected by Trainer hook)
    - ``'transform_metrics'``: optional dict populated by transform postprocess
      (e.g., semantic-entropy tracking stats)

    Example::

        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo
        from functools import partial

        # Boost mode (legacy)
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optimizer,
            advantage_transform_fn=build_gtpo_transform(
                alpha_1=1.0, alpha_2=0.1,
                tokenizer=tokenizer, hicra_gamma=0.3,
            ),
        )

        # SEPA mode (linear anneal over 500 steps, hicra_gamma not needed)
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optimizer,
            advantage_transform_fn=build_gtpo_transform(
                alpha_1=1.0, alpha_2=0.1,
                tokenizer=tokenizer,
                sepa_steps=500,
            ),
        )

        # SEPA auto mode (self-supervised variance schedule + optional cap)
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optimizer,
            advantage_transform_fn=build_gtpo_transform(
                alpha_1=1.0, alpha_2=0.1,
                tokenizer=tokenizer,
                sepa_schedule="auto",
                sepa_steps=1000,  # optional fallback cap
            ),
        )

    Args:
        alpha_1: Base reward weight (default 1.0).
        alpha_2: Entropy-shaped weight (default 0.1).
        reward_threshold: Threshold for O+/O- partition (default 0.5).
        eps: Numerical stability constant (Remark 2.1).
        tokenizer: HF-compatible tokenizer (required when HICRA is active).
        strategic_grams: Planning phrases for HICRA (defaults to built-in list
                        when tokenizer is provided).
        hicra_gamma: Entropy boost factor for boost mode (default 0.0 = disabled).
                    Ignored when SEPA is active.
        sepa_steps: SEPA annealing horizon — number of training steps over
                   which λ linearly anneals from 0 to 1 in linear mode.
                   In auto mode, acts as optional fallback cap (default 0 = no cap).
        sepa_schedule: ``'linear'`` (default) or ``'auto'``.
                      ``'linear'`` uses ``sepa_steps`` only.
                      ``'auto'`` uses variance-driven λ, optionally capped by
                      linear ``sepa_steps``.
        semantic_entropy: Enable planning-level semantic-entropy tracking in
                         ``postprocess_batch``. This does not affect gradients
                         or the GTPO loss path.

    Returns:
        A callable ``(advantages, batch_data) -> advantages``.

    Raises:
        ValueError: If ``hicra_gamma < 0``, ``sepa_steps < 0``,
                   ``sepa_schedule`` is invalid,
                   or planning-mask-dependent features are active without
                   tokenizer.
    """
    if alpha_1 < 0.0:
        raise ValueError(f"alpha_1 must be >= 0, got {alpha_1}.")
    if alpha_2 < 0.0:
        raise ValueError(f"alpha_2 must be >= 0, got {alpha_2}.")
    if hicra_gamma < 0.0:
        raise ValueError(f"hicra_gamma must be >= 0, got {hicra_gamma}.")
    if sepa_steps < 0:
        raise ValueError(
            f"sepa_steps must be >= 0, got {sepa_steps}."
        )
    sepa_schedule = normalize_sepa_schedule(sepa_schedule)

    sepa_enabled = (sepa_steps > 0 or sepa_schedule == "auto")
    if sepa_enabled and tokenizer is None:
        raise ValueError(
            "SEPA requires a tokenizer for planning token identification."
        )
    if semantic_entropy and tokenizer is None:
        raise ValueError(
            "semantic_entropy=True requires a tokenizer for planning token "
            "identification."
        )
    if hicra_gamma > 0.0 and tokenizer is None:
        raise ValueError(
            "hicra_gamma > 0 requires a tokenizer for planning token identification."
        )
    if sepa_enabled and hicra_gamma > 0.0:
        warnings.warn(
            f"SEPA mode (sepa_schedule={sepa_schedule!r}, sepa_steps={sepa_steps}) "
            f"is active; "
            f"hicra_gamma={hicra_gamma} is ignored (boost is not used "
            f"when SEPA is active).",
            stacklevel=2,
        )

    return _GTPOTransform(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        reward_threshold=reward_threshold,
        eps=eps,
        tokenizer=tokenizer,
        strategic_grams=strategic_grams,
        hicra_gamma=hicra_gamma,
        sepa_steps=sepa_steps,
        sepa_schedule=sepa_schedule,
        semantic_entropy=semantic_entropy,
    )


# ---------------------------------------------------------------------------
# Deprecated convenience wrapper — will be removed in a future release.
# ---------------------------------------------------------------------------


_LEGACY_HICRA_ALPHA_DEFAULT = 0.2
_LEGACY_ENTROPY_WEIGHT_DEFAULT = 0.1


def create_tinylora_reasoning_setup(
    model: nn.Module,
    tokenizer: Any,
    optimizer: optim.Optimizer,
    *,
    lora_config: Optional[Dict[str, Any]] = None,
    advantage_transform_fn: Optional[
        Callable[[mx.array, Dict[str, Any]], mx.array]
    ] = None,
    strategic_grams: Optional[List[str]] = None,
    alpha_1: float = 1.0,
    alpha_2: float = 0.1,
    reward_threshold: float = 0.5,
    hicra_gamma: float = 0.3,
    hicra_alpha: float = _LEGACY_HICRA_ALPHA_DEFAULT,
    entropy_weight: float = _LEGACY_ENTROPY_WEIGHT_DEFAULT,
    compile_training: Union[bool, str] = "auto",
    gradient_checkpointing: Union[bool, int] = False,
    micro_batch_size: Optional[int] = None,
    auto_reload: bool = True,
    adapter_save_path: str = "./lora_adapters.safetensors",
    max_grad_norm: Optional[float] = 0.5,
    **trainer_kwargs: Any,
) -> Tuple[Trainer, Dict[str, float]]:
    """Create a Trainer wired for LoRA — **deprecated**, will be removed.

    Prefer composing the three steps directly::

        lora_model, stats = create_lora_setup(model, lora_config=...)
        transform = build_gtpo_transform(...)
        trainer = Trainer(model=lora_model, ..., advantage_transform_fn=transform)

    See ``experiments/countdown_reasoning_lora.py`` for the full pattern.
    """
    warnings.warn(
        "create_tinylora_reasoning_setup() is deprecated and will be removed "
        "in a future release. Compose the steps directly: "
        "create_lora_setup() → build_gtpo_transform() → Trainer(). "
        "See experiments/countdown_reasoning_lora.py for the pattern.",
        DeprecationWarning,
        stacklevel=2,
    )

    _DEFAULT_LORA_CONFIG: Dict[str, Any] = {
        "lora_layers": 4,
        "lora_rank": 2,
        "lora_scale": 8.0,
        "lora_dropout": 0.0,
    }

    merged_lora_config = dict(_DEFAULT_LORA_CONFIG)
    if lora_config:
        merged_lora_config.update(lora_config)

    lora_model, memory_stats = create_lora_setup(
        model=model,
        lora_config=merged_lora_config,
        auto_reload=auto_reload,
        adapter_save_path=adapter_save_path,
    )

    uses_legacy_simplified_params = (
        hicra_alpha != _LEGACY_HICRA_ALPHA_DEFAULT
        or entropy_weight != _LEGACY_ENTROPY_WEIGHT_DEFAULT
    )
    if advantage_transform_fn is None:
        if uses_legacy_simplified_params:
            advantage_transform_fn = build_gtpo_hicra_transform(
                tokenizer,
                strategic_grams=strategic_grams,
                hicra_alpha=hicra_alpha,
                entropy_weight=entropy_weight,
            )
        else:
            advantage_transform_fn = build_gtpo_transform(
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                reward_threshold=reward_threshold,
                tokenizer=tokenizer,
                strategic_grams=strategic_grams,
                hicra_gamma=hicra_gamma,
            )

    trainer = Trainer(
        model=lora_model,
        advantage_fn=compute_advantages,
        loss_fn=policy_loss,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        compile_training=compile_training,
        gradient_checkpointing=gradient_checkpointing,
        micro_batch_size=micro_batch_size,
        advantage_transform_fn=advantage_transform_fn,
        **trainer_kwargs,
    )
    return trainer, memory_stats
