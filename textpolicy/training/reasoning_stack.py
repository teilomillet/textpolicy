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
    gradient_checkpointing: bool = False,
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
            instead of caching them — reduces peak memory at the cost of
            ~20-30 %% extra compute.
        micro_batch_size: Process at most *N* episodes per forward/backward
            pass, accumulating gradients across micro-batches.  Reduces peak
            activation memory roughly by a factor of N.
        auto_reload: Auto-reload adapters for the rollout policy.
        adapter_save_path: Where to persist LoRA adapter weights.
        max_grad_norm: Clip gradient norm (``None`` to disable).
        **trainer_kwargs: Forwarded to :class:`Trainer`.

    Memory Optimization:
        For long sequences or memory-constrained hardware, enabling both
        ``gradient_checkpointing=True`` and ``micro_batch_size=4`` can cut
        peak memory by ~34 %% and total step time by ~35 %% (benchmarked at
        seq_length=1024).  Start with ``micro_batch_size=4`` and adjust
        based on your hardware.  See ``docs/06_performance.md`` for full
        benchmark numbers.

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
