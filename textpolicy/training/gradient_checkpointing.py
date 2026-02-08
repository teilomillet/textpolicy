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

Implementation note: Python's data model looks up ``__call__`` on the
**type**, not the instance, when handling the ``()`` operator.  Setting
``layer.__call__ = wrapper`` as an instance attribute does NOT work —
``layer(x)`` still dispatches to ``type(layer).__call__``.  We use
``__class__`` reassignment to swap each layer's class to a dynamically
created subclass whose class-level ``__call__`` invokes
``mx.checkpoint``.  This preserves the parameter tree structure
(no extra ``inner.`` prefix) and all attribute access.

Compatible with ``mx.compile`` on MLX >= 0.30.6 when using explicit
positional args (our implementation pattern).

References:
    Issue #55 — O(n^1.89) training scaling and memory wall
    MLX docs — mx.checkpoint
    Python Data Model — implicit special method lookup
"""

import logging
import math
from typing import Dict, List, Optional, Type

import mlx.core as mx  # type: ignore
import mlx.nn as nn  # type: ignore
from mlx.utils import tree_flatten, tree_unflatten  # type: ignore

logger = logging.getLogger(__name__)

# Cache: original_class -> checkpointed_subclass.
# Avoids creating duplicate subclasses when apply is called multiple times
# or on multiple models that share the same layer class.
_checkpointed_class_cache: Dict[Type, Type] = {}


def _make_checkpointed_class(original_cls: Type) -> Type:
    """Create a subclass of ``original_cls`` with ``__call__`` wrapped by
    ``mx.checkpoint``.

    The subclass overrides ``__call__`` at the **class level**, which is
    where Python's ``()`` operator looks it up.  The ``_original_class``
    attribute lets ``remove_gradient_checkpointing`` restore the layer.
    """
    original_call = original_cls.__call__

    def checkpointed_call(self, x, mask=None, cache=None):
        # Extract trainable parameters as explicit checkpoint args.
        # With LoRA, this is just the adapter weights (lora_a, lora_b).
        flat = tree_flatten(self.trainable_parameters())
        keys = [k for k, _ in flat]
        arrays = [a for _, a in flat]
        # Preserve the pre-call parameter bindings. In compiled mode this
        # prevents transient checkpoint inputs from leaking into module state
        # after the forward pass.
        original_params = tree_unflatten(flat)

        def forward(x_in, *params):
            # Reconnect trainable parameters into the layer.
            param_dict = tree_unflatten(list(zip(keys, params)))
            self.update(param_dict)
            try:
                return original_call(self, x_in, mask=mask, cache=cache)
            finally:
                self.update(original_params)

        return mx.checkpoint(forward)(x, *arrays)

    cls_name = f"_Checkpointed_{original_cls.__name__}"
    checkpointed_cls = type(cls_name, (original_cls,), {
        "__call__": checkpointed_call,
        "_original_class": original_cls,
    })
    return checkpointed_cls


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


def apply_gradient_checkpointing(
    model: nn.Module, checkpoint_every: Optional[int] = None
) -> int:
    """Wrap selected transformer layers' forward passes with ``mx.checkpoint``.

    For each selected layer in the model's transformer stack, swaps the
    layer's ``__class__`` to a dynamically-created subclass whose
    ``__call__`` invokes ``mx.checkpoint``.  This ensures that ``layer(x)``
    (Python's ``()`` operator, which dispatches via ``type(layer).__call__``)
    actually hits the checkpointed path.

    Already-checkpointed layers (those whose class has ``_original_class``)
    are skipped, making this function idempotent.

    **Selective checkpointing (Chen et al. 2016):** Rather than
    checkpointing every layer (maximum recompute), checkpointing every
    sqrt(n)-th layer achieves O(sqrt(n)) memory with only ~one extra
    forward pass of compute.  This is the default strategy when
    ``checkpoint_every`` is ``None``.

    Selective checkpointing is mathematically exact — it produces
    bit-identical gradients to full checkpointing.

    Args:
        model: MLX transformer model (with LoRA adapters applied).
        checkpoint_every: Controls which layers get checkpointed.

            - ``None`` (default): sqrt(n) strategy —
              ``stride = max(1, int(math.sqrt(n_layers)))``.
              Best balance of memory savings vs recompute overhead.
            - ``1``: checkpoint every layer (maximum memory savings,
              ~30% extra compute).
            - ``k > 1``: checkpoint every k-th layer.

    Returns:
        Number of layers that were newly checkpointed.
    """
    layers = _get_layers(model)
    n_layers = len(layers)

    if checkpoint_every is None:
        stride = max(1, int(math.sqrt(n_layers)))
    else:
        if isinstance(checkpoint_every, bool) or not isinstance(
            checkpoint_every, int
        ):
            raise ValueError(
                "checkpoint_every must be a positive integer when set, "
                f"got {checkpoint_every!r}"
            )
        if checkpoint_every <= 0:
            raise ValueError(
                "checkpoint_every must be > 0, "
                f"got {checkpoint_every!r}"
            )
        stride = checkpoint_every

    count = 0

    for i, layer in enumerate(layers):
        if i % stride != 0:
            continue  # anchor layer — keep activations

        if getattr(type(layer), "_original_class", None) is not None:
            # Already checkpointed — skip for idempotency.
            continue

        original_cls = type(layer)
        if original_cls not in _checkpointed_class_cache:
            _checkpointed_class_cache[original_cls] = (
                _make_checkpointed_class(original_cls)
            )

        layer.__class__ = _checkpointed_class_cache[original_cls]
        count += 1

    if count > 0:
        logger.info(
            "Gradient checkpointing applied to %d of %d layers (every %d)",
            count,
            n_layers,
            stride,
        )

    return count


def remove_gradient_checkpointing(model: nn.Module) -> int:
    """Restore original ``__class__`` on all checkpointed layers.

    Args:
        model: MLX transformer model with checkpointing active.

    Returns:
        Number of layers restored.
    """
    layers = _get_layers(model)
    count = 0

    for layer in layers:
        original_cls = getattr(type(layer), "_original_class", None)
        if original_cls is not None:
            layer.__class__ = original_cls
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
        True if at least one layer's class has ``_original_class``.
    """
    try:
        layers = _get_layers(model)
    except ValueError:
        return False

    return any(
        getattr(type(layer), "_original_class", None) is not None
        for layer in layers
    )
