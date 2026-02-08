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
from textpolicy.generation.lora import create_lora_setup
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

    Optionally fuses HICRA planning-token amplification by boosting the
    entropy array before GTPO processes it (via ``boost_entropy_with_planning``).
    When ``hicra_gamma > 0``, planning tokens receive artificially higher
    entropy so GTPO's O+/O- machinery naturally assigns them more credit
    (O+) or less blame (O-).

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
        Short-circuits when HICRA fusion is disabled (``hicra_gamma == 0``).
        """
        if self.hicra_gamma == 0.0 or self.tokenizer is None or not self.grams:
            return
        if batch_data.get("planning_mask") is not None:
            return
        token_ids = self._flatten_actions(batch_data)
        batch_data["planning_mask"] = identify_planning_tokens(
            token_ids, self.tokenizer, self.grams
        )

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

        # Optional HICRA fusion: boost entropies at planning positions
        if self.hicra_gamma > 0.0 and self.tokenizer is not None:
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
) -> Callable[[mx.array, Dict[str, Any]], mx.array]:
    """
    Build a Trainer-compatible transform for GTPO (arXiv 2508.04349).

    Implements Eq. 3, 5, 6 from arXiv 2508.04349. The returned transform
    completely replaces standard GRPO advantages with entropy-shaped,
    separately-normalized advantages.

    PPO clipping (Eq. 7) is handled by the Trainer's ``loss_fn`` — use
    ``functools.partial(grpo.policy_loss, clip_ratio=0.2)`` or similar.

    **HICRA Fusion (optional):**  When ``hicra_gamma > 0`` and a tokenizer
    is provided, the transform boosts entropy at planning token positions
    *before* GTPO processes it.  GTPO's O+/O- machinery then naturally
    assigns planning tokens more credit (O+) or less blame (O-).

    Expected batch_data keys (auto-populated by the Trainer):
    - ``'rewards'``: episode-level rewards [num_episodes]
    - ``'token_entropies'``: per-token entropy [total_tokens] (auto-computed)
    - ``'episode_lengths'``: token count per episode
    - ``'act'``: token IDs (required when ``hicra_gamma > 0``)

    Example::

        from textpolicy.training import Trainer, build_gtpo_transform
        from textpolicy.algorithms import grpo
        from functools import partial

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

    Args:
        alpha_1: Base reward weight (default 1.0).
        alpha_2: Entropy-shaped weight (default 0.1).
        reward_threshold: Threshold for O+/O- partition (default 0.5).
        eps: Numerical stability constant (Remark 2.1).
        tokenizer: HF-compatible tokenizer (required when ``hicra_gamma > 0``).
        strategic_grams: Planning phrases for HICRA (defaults to built-in list
                        when tokenizer is provided).
        hicra_gamma: Entropy boost factor for HICRA fusion (default 0.0 = disabled).

    Returns:
        A callable ``(advantages, batch_data) -> advantages``.

    Raises:
        ValueError: If ``hicra_gamma < 0``, or ``hicra_gamma > 0`` without tokenizer.
    """
    if alpha_1 < 0.0:
        raise ValueError(f"alpha_1 must be >= 0, got {alpha_1}.")
    if alpha_2 < 0.0:
        raise ValueError(f"alpha_2 must be >= 0, got {alpha_2}.")
    if hicra_gamma < 0.0:
        raise ValueError(f"hicra_gamma must be >= 0, got {hicra_gamma}.")
    if hicra_gamma > 0.0 and tokenizer is None:
        raise ValueError(
            "hicra_gamma > 0 requires a tokenizer for planning token identification."
        )

    return _GTPOTransform(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        reward_threshold=reward_threshold,
        eps=eps,
        tokenizer=tokenizer,
        strategic_grams=strategic_grams,
        hicra_gamma=hicra_gamma,
    )


# ---------------------------------------------------------------------------
# Deprecated convenience wrapper — will be removed in a future release.
# ---------------------------------------------------------------------------


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

    if advantage_transform_fn is None:
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


