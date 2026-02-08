# textpolicy/training/gradient_checkpointing.py
"""
Gradient checkpointing for MLX transformer models.

Trades ~30% more compute for ~60% less activation memory by discarding
intermediate activations during the forward pass and recomputing them
on-the-fly during the backward pass.  This addresses the O(n^2)
memory wall from attention activations at long sequence lengths.

Critical MLX constraint: ``mx.checkpoint`` only preserves gradients
for **explicit positional arguments**.  Parameters captured via closure
get zero gradients.  The implementation therefore:

1. Extracts trainable parameters (LoRA adapters) via ``tree_flatten``
2. Passes them as explicit positional arguments to ``mx.checkpoint``
3. Inside the checkpoint: ``layer.update(param_dict)`` reconnects them
4. ``mask``/``cache`` kwargs pass through without checkpoint tracking

This is safe with LoRA because only adapter weights need gradients.
Frozen base weights are closed over and correctly receive no gradients.

Incompatibility: ``mx.checkpoint`` inside ``mx.compile`` fails.  When
gradient checkpointing is enabled, the Trainer forces compilation off.

References:
    Issue #55 — O(n^1.89) training scaling and memory wall
    MLX docs — mx.checkpoint
"""

import logging
from typing import List

import mlx.core as mx  # type: ignore
import mlx.nn as nn  # type: ignore
from mlx.utils import tree_flatten, tree_unflatten  # type: ignore

logger = logging.getLogger(__name__)


def _get_layers(model: nn.Module) -> List[nn.Module]:
    """Extract transformer layers from an MLX-LM model.

    Follows the same access pattern as ``lora.py:124``: models from
    ``mlx_lm.load()`` wrap layers in ``model.model.layers``, while
    bare transformer blocks use ``model.layers`` directly.

    Raises:
        ValueError: If the model has no ``layers`` attribute.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    else:
        raise ValueError(
            "Model has no 'layers' attribute. Gradient checkpointing "
            "requires a transformer model with a '.layers' list "
            "(or '.model.layers')."
        )


def apply_gradient_checkpointing(model: nn.Module) -> int:
    """Wrap each transformer layer's forward pass with ``mx.checkpoint``.

    For each layer in the model's transformer stack:
    1. Save the original ``__call__`` as ``layer._original_call``
    2. Replace ``__call__`` with a wrapper that:
       a. Extracts trainable parameters via ``tree_flatten``
       b. Passes them as explicit positional args to ``mx.checkpoint``
       c. Inside checkpoint: ``layer.update()`` reconnects params,
          then calls the original forward

    Already-checkpointed layers (those with ``_original_call``) are
    skipped, making this function idempotent.

    Args:
        model: MLX transformer model (with LoRA adapters applied).

    Returns:
        Number of layers that were newly checkpointed.
    """
    layers = _get_layers(model)
    count = 0

    for layer in layers:
        if hasattr(layer, "_original_call"):
            # Already checkpointed — skip for idempotency.
            continue

        original_call = layer.__call__
        layer._original_call = original_call

        def _make_checkpointed_call(layer_ref, orig_call):
            """Build a checkpoint wrapper closed over a specific layer.

            The closure captures ``layer_ref`` and ``orig_call`` by
            value (via the factory function's local scope), avoiding
            the late-binding pitfall of Python closures in loops.
            """

            def checkpointed_call(x, mask=None, cache=None):
                # Extract trainable parameters as explicit checkpoint args.
                # With LoRA, this is just the adapter weights (lora_a, lora_b).
                flat = tree_flatten(layer_ref.trainable_parameters())
                keys = [k for k, _ in flat]
                arrays = [a for _, a in flat]

                def forward(x_in, *params):
                    # Reconnect trainable parameters into the layer.
                    param_dict = tree_unflatten(list(zip(keys, params)))
                    layer_ref.update(param_dict)
                    return orig_call(x_in, mask=mask, cache=cache)

                return mx.checkpoint(forward)(x, *arrays)

            return checkpointed_call

        layer.__call__ = _make_checkpointed_call(layer, original_call)
        count += 1

    if count > 0:
        logger.info(
            "Gradient checkpointing enabled for %d transformer layers", count
        )

    return count


def remove_gradient_checkpointing(model: nn.Module) -> int:
    """Restore original ``__call__`` on all checkpointed layers.

    Args:
        model: MLX transformer model with checkpointing active.

    Returns:
        Number of layers restored.
    """
    layers = _get_layers(model)
    count = 0

    for layer in layers:
        if hasattr(layer, "_original_call"):
            layer.__call__ = layer._original_call
            del layer._original_call
            count += 1

    if count > 0:
        logger.info(
            "Gradient checkpointing removed from %d transformer layers", count
        )

    return count


def is_gradient_checkpointing_active(model: nn.Module) -> bool:
    """Check if any transformer layer has gradient checkpointing enabled.

    Args:
        model: MLX transformer model.

    Returns:
        True if at least one layer has a ``_original_call`` attribute.
    """
    try:
        layers = _get_layers(model)
    except ValueError:
        return False

    return any(hasattr(layer, "_original_call") for layer in layers)
