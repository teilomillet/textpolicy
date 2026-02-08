# textpolicy/training/reasoning_stack.py
"""
Composable setup helpers for a combined reasoning stack:

- GTPO entropy-weighted advantages (arXiv:2508.04349)
- HICRA planning-token amplification (arXiv:2509.03646)
- TinyLoRA-inspired low-rank defaults (arXiv:2602.04118)

The helpers in this module are intentionally thin wrappers around existing
TextPolicy primitives so they remain easy to test and safe to evolve.
"""

from __future__ import annotations

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

_DEFAULT_TINYLORA_CONFIG: Dict[str, Any] = {
    "lora_layers": 4,
    "lora_rank": 2,
    "lora_scale": 8.0,
    "lora_dropout": 0.0,
}


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


class _GTPOFaithfulTransform:
    """Trainer transform for paper-faithful GTPO (arXiv 2508.04349).

    Computes entropy-shaped, separately-normalized advantages per Eq. 3, 5, 6,
    completely replacing the standard GRPO advantages.  All operations are
    compile-safe (no ``mx.eval`` or ``.item()``).

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
    ) -> None:
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.reward_threshold = reward_threshold
        self.eps = eps

    def __call__(self, advantages: mx.array, batch_data: Dict[str, Any]) -> mx.array:
        episode_lengths = batch_data.get("episode_lengths")
        if episode_lengths is None:
            raise ValueError(
                "batch_data must include 'episode_lengths' for faithful GTPO. "
                "Ensure the rollout pipeline provides episode_lengths in the batch."
            )

        token_entropies = batch_data.get("token_entropies")
        if token_entropies is None:
            raise ValueError(
                "batch_data must include 'token_entropies' for faithful GTPO. "
                "This is auto-computed when advantage_transform_fn is set."
            )

        rewards = batch_data.get("rewards")
        if rewards is None:
            raise ValueError(
                "batch_data must include 'rewards' for faithful GTPO. "
                "Ensure the rollout pipeline provides episode-level rewards."
            )

        # Defensively flatten token_entropies if 2D padded
        flat_entropies = _flatten_padded_token_rows(
            token_entropies,
            episode_lengths,
            field_name="token_entropies",
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
        faithful_advantages = normalize_gtpo_advantages(
            shaped_rewards, is_positive, eps=self.eps,
        )

        return faithful_advantages


def build_gtpo_faithful_transform(
    *,
    alpha_1: float = 1.0,
    alpha_2: float = 0.1,
    reward_threshold: float = 0.5,
    eps: float = 1e-8,
) -> Callable[[mx.array, Dict[str, Any]], mx.array]:
    """
    Build a Trainer-compatible transform for paper-faithful GTPO.

    Implements Eq. 3, 5, 6 from arXiv 2508.04349. The returned transform
    completely replaces standard GRPO advantages with entropy-shaped,
    separately-normalized advantages.

    PPO clipping (Eq. 7) is handled by the Trainer's ``loss_fn`` — use
    ``functools.partial(grpo.policy_loss, clip_ratio=0.2)`` or similar.

    Expected batch_data keys (auto-populated by the Trainer):
    - ``'rewards'``: episode-level rewards [num_episodes]
    - ``'token_entropies'``: per-token entropy [total_tokens] (auto-computed)
    - ``'episode_lengths'``: token count per episode

    Example::

        from textpolicy.training import Trainer, build_gtpo_faithful_transform
        from textpolicy.algorithms import grpo
        from functools import partial

        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=partial(grpo.policy_loss, clip_ratio=0.2),
            optimizer=optimizer,
            advantage_transform_fn=build_gtpo_faithful_transform(
                alpha_1=1.0, alpha_2=0.1,
            ),
        )

    Args:
        alpha_1: Base reward weight (default 1.0).
        alpha_2: Entropy-shaped weight (default 0.1).
        reward_threshold: Threshold for O+/O- partition (default 0.5).
        eps: Numerical stability constant (Remark 2.1).

    Returns:
        A callable ``(advantages, batch_data) -> advantages``.
    """
    if alpha_1 < 0.0:
        raise ValueError(f"alpha_1 must be >= 0, got {alpha_1}.")
    if alpha_2 < 0.0:
        raise ValueError(f"alpha_2 must be >= 0, got {alpha_2}.")

    return _GTPOFaithfulTransform(
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        reward_threshold=reward_threshold,
        eps=eps,
    )


def create_tinylora_reasoning_setup(
    model: nn.Module,
    tokenizer: Any,
    optimizer: optim.Optimizer,
    *,
    lora_config: Optional[Dict[str, Any]] = None,
    strategic_grams: Optional[List[str]] = None,
    hicra_alpha: float = 0.2,
    entropy_weight: float = 0.1,
    compile_training: Union[bool, str] = "auto",
    gradient_checkpointing: Union[bool, int] = False,
    micro_batch_size: Optional[int] = None,
    auto_reload: bool = True,
    adapter_save_path: str = "./lora_adapters.safetensors",
    max_grad_norm: Optional[float] = 0.5,
    **trainer_kwargs: Any,
) -> Tuple[Trainer, Dict[str, float]]:
    """
    Create a Trainer wired for GTPO + HICRA with TinyLoRA-style defaults.

    TinyLoRA-style here means a compact LoRA default configuration
    (small rank + fewer adapted layers).  Exact TinyLoRA internals from
    the paper are not re-implemented in this helper.

    Args:
        model: Base language model (will be wrapped with LoRA adapters).
        tokenizer: Tokenizer matching the model.
        optimizer: MLX optimizer instance (e.g. ``optim.Adam``).
        lora_config: Override default LoRA hyper-parameters (rank, alpha, layers).
        strategic_grams: Path or list of strategic n-grams for HICRA.
        hicra_alpha: Blend weight for HICRA credit assignment (default 0.2).
        entropy_weight: GTPO entropy re-weighting coefficient beta (default 0.1).
        compile_training: ``True``, ``False``, or ``"auto"`` for mx.compile.
        gradient_checkpointing: Re-compute activations during backward pass
            instead of caching them.  ``True`` uses sqrt(n) layer selection
            (Chen et al. 2016) for the best memory/compute trade-off.
            An ``int`` sets the explicit stride (``1`` = every layer,
            ``4`` = every 4th layer).
        micro_batch_size: Process at most *N* episodes per logprob-extraction
            forward pass. This bounds per-forward logits size and can reduce
            peak memory, though exact savings depend on MLX scheduling while
            preserving the same optimizer-step semantics.
        auto_reload: Auto-reload adapters for the rollout policy.
        adapter_save_path: Where to persist LoRA adapter weights.
        max_grad_norm: Clip gradient norm (``None`` to disable).
        **trainer_kwargs: Forwarded to :class:`Trainer`.

    Memory Optimization:
        For long sequences or memory-constrained hardware, start with
        ``micro_batch_size=4`` and then add ``gradient_checkpointing=True``
        if you still hit memory limits. Benchmark on your hardware before
        locking defaults. See ``docs/06_performance.md`` for examples and
        guidance.

    Returns:
        Tuple of ``(trainer, memory_stats)`` where *memory_stats* contains
        LoRA setup diagnostics (trainable params, frozen params, etc.).
    """
    merged_lora_config = dict(_DEFAULT_TINYLORA_CONFIG)
    if lora_config:
        merged_lora_config.update(lora_config)

    lora_model, memory_stats = create_lora_setup(
        model=model,
        lora_config=merged_lora_config,
        auto_reload=auto_reload,
        adapter_save_path=adapter_save_path,
    )

    transform = build_gtpo_hicra_transform(
        tokenizer,
        strategic_grams=strategic_grams,
        hicra_alpha=hicra_alpha,
        entropy_weight=entropy_weight,
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
        advantage_transform_fn=transform,
        **trainer_kwargs,
    )
    return trainer, memory_stats
