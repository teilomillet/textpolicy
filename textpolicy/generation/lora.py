# textpolicy/generation/lora.py
"""
Pure LoRA/QLoRA functions for MLX-LM integration.

Following TextPolicy design principles:
- Pure function composition
- Zero abstraction cost
- MLX compilation optimization
- Memory-efficient training

These functions integrate with our GRPO trainer for efficient
parameter updates using LoRA adapters.
"""

from typing import Dict, Optional, Tuple, Any
import mlx.core as mx # type: ignore
import mlx.nn as nn # type: ignore

# Import LoRA from MLX-LM
try:
    from mlx_lm.lora import LoRALinear # type: ignore
except ImportError:
    try:
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError:
        print("Warning: LoRA not available in this MLX-LM version")
        LoRALinear = None # type: ignore

# Import linear_to_lora_layers for automatic dense/MoE LoRA conversion
_has_linear_to_lora_layers = False
try:
    from mlx_lm.tuner.utils import linear_to_lora_layers as _linear_to_lora_layers
    _has_linear_to_lora_layers = True
except ImportError:
    _linear_to_lora_layers = None  # type: ignore

# Import MoE layer types for detection
try:
    from mlx_lm.models.switch_layers import SwitchLinear as _SwitchLinear
    from mlx_lm.models.switch_layers import SwitchGLU as _SwitchGLU
except ImportError:
    _SwitchLinear = None  # type: ignore
    _SwitchGLU = None  # type: ignore


def apply_lora(
    model: nn.Module,
    lora_layers: int = 8,
    lora_rank: int = 8,
    lora_scale: float = 20.0,
    lora_dropout: float = 0.0,
    lora_keys: Optional[set] = None,
) -> nn.Module:
    """
    Apply LoRA adapters to an MLX model.

    Delegates to ``mlx_lm.tuner.utils.linear_to_lora_layers`` which handles
    both dense (``nn.Linear → LoRALinear``) and MoE
    (``SwitchLinear → LoRASwitchLinear``) layers automatically.  Falls back
    to the manual q_proj/v_proj conversion when ``mlx_lm.tuner.utils`` is
    unavailable.

    Args:
        model: Original MLX model (must have a ``layers`` attribute,
            either directly or via ``model.model.layers``).
        lora_layers: Number of transformer blocks to apply LoRA to
            (counted from the end of the stack).
        lora_rank: LoRA rank parameter (lower = more compression).
        lora_scale: LoRA scaling factor (alpha).
        lora_dropout: LoRA dropout rate.
        lora_keys: Optional set of module key names to restrict which
            sub-modules get LoRA (e.g. ``{"self_attn.q_proj"}``).
            When ``None``, all eligible linear / switch-linear layers
            in the selected blocks are converted.

    Returns:
        Model with LoRA adapters applied (mutated in-place).
    """
    if LoRALinear is None:
        print("Warning: LoRA not available, returning original model")
        return model

    if _has_linear_to_lora_layers:
        config = {
            "rank": lora_rank,
            "scale": lora_scale,
            "dropout": lora_dropout,
            "keys": lora_keys,
        }
        # linear_to_lora_layers accesses model.layers — both the top-level
        # Model wrapper and model.model resolve to the same list for all
        # MLX-LM models, so we can pass the model directly.
        _linear_to_lora_layers(model, num_layers=lora_layers, config=config)
        is_moe = detect_moe_model(model)
        arch = "MoE (SwitchLinear → LoRASwitchLinear)" if is_moe else "dense"
        print(
            f"Applied LoRA to {lora_layers} layers "
            f"(rank={lora_rank}, scale={lora_scale}, arch={arch})"
        )
        if is_moe:
            # SwitchGLU/SwitchMLP only apply mx.stop_gradient to routing
            # indices when model.training is True.  Without this, backprop
            # through mx.gather_mm fails.  Trainer.train() sets this
            # automatically; warn for manual usage.
            if not model.training:
                print(
                    "  ⚠ MoE model: call model.train() before backprop "
                    "(Trainer does this automatically)"
                )
    else:
        _apply_lora_manual(model, lora_layers, lora_rank, lora_scale, lora_dropout)

    return model


def _apply_lora_manual(
    model: nn.Module,
    lora_layers: int,
    lora_rank: int,
    lora_scale: float,
    lora_dropout: float,
) -> None:
    """Fallback: manually convert q_proj/v_proj to LoRALinear (dense only)."""
    layers = model.model.layers if hasattr(model, "model") else model.layers
    for layer_idx in range(max(0, len(layers) - lora_layers), len(layers)):
        layer = layers[layer_idx]

        if hasattr(layer, "self_attn"):
            for proj_name in ("q_proj", "v_proj"):
                proj = getattr(layer.self_attn, proj_name, None)
                if proj is None:
                    continue
                if "LoRA" in type(proj).__name__:
                    continue
                setattr(
                    layer.self_attn,
                    proj_name,
                    LoRALinear.from_base(
                        proj, r=lora_rank, scale=lora_scale, dropout=lora_dropout
                    ),
                )

    print(f"Applied LoRA to {lora_layers} layers (rank={lora_rank}, scale={lora_scale})")


def freeze_base(model: nn.Module) -> nn.Module:
    """
    Freeze base model parameters, keeping only LoRA adapter weights trainable.

    After ``model.freeze()`` every parameter (including LoRA) is frozen.
    This function walks the module tree via ``apply_to_modules`` and calls
    ``module.unfreeze(keys=["lora_a", "lora_b"])`` on every LoRA module,
    restoring gradient flow through the adapters.

    Args:
        model: Model with LoRA adapters already applied.

    Returns:
        The same model with base weights frozen and LoRA weights trainable.
    """
    # 1. Freeze everything
    model.freeze()

    # 2. Walk the module tree and unfreeze LoRA adapter weights.
    #    We detect LoRA modules by checking for the ``lora_a`` attribute
    #    (works with any LoRALinear implementation).
    lora_module_count = 0

    def _unfreeze_lora(_path: str, module: nn.Module) -> nn.Module:
        nonlocal lora_module_count
        if hasattr(module, "lora_a"):
            module.unfreeze(keys=["lora_a", "lora_b"])
            lora_module_count += 1
        return module

    model.apply_to_modules(_unfreeze_lora)

    # 3. Count actual trainable vs total parameters for reporting.
    from mlx.utils import tree_flatten
    trainable_params = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    total_params = sum(p.size for _, p in tree_flatten(model.parameters()))

    if total_params > 0:
        pct = trainable_params / total_params * 100
    else:
        pct = 0.0

    print(
        f"Frozen base model: {trainable_params:,} trainable / "
        f"{total_params:,} total parameters ({pct:.2f}%)"
    )
    print(f"  LoRA modules unfrozen: {lora_module_count}")

    return model


def extract_params(model: nn.Module) -> Dict[str, mx.array]:
    """
    Extract only LoRA adapter parameters from a model for saving.

    Uses ``tree_flatten`` on trainable parameters and filters for names
    containing ``lora`` or ``adapter``.  Returns an empty dict when no
    LoRA parameters are found (never returns dummy data).

    Args:
        model: Model with LoRA adapters.

    Returns:
        Flat dictionary mapping dotted parameter names to ``mx.array`` values.
    """
    from mlx.utils import tree_flatten

    lora_params: Dict[str, mx.array] = {}
    for name, param in tree_flatten(model.trainable_parameters()):
        if "lora" in name.lower() or "adapter" in name.lower():
            lora_params[name] = param
    return lora_params


def merge_weights(model: nn.Module) -> nn.Module:
    """
    Pure function to merge LoRA weights back into the base model.
    
    This creates a new model with the LoRA adaptations permanently
    integrated, useful for deployment.
    
    Args:
        model: Model with trained LoRA adapters
        
    Returns:
        Model with merged weights (no LoRA layers)
    """
    # This is a simplified version - real implementation would
    # properly merge the LoRA matrices into the base weights
    print("Note: LoRA weight merging is placeholder - implement based on MLX LoRA utils")
    return model


# ── MoE detection utilities ──────────────────────────────────────────────


def detect_moe_model(model: nn.Module) -> bool:
    """
    Detect whether *model* is a Mixture-of-Experts architecture.

    Walks the module tree looking for ``SwitchLinear`` or ``SwitchGLU``
    layers (the standard MoE building blocks in MLX-LM).

    Args:
        model: An MLX-LM model (e.g. from ``mlx_lm.load()``).

    Returns:
        ``True`` if at least one MoE layer is found, ``False`` otherwise.
    """
    if _SwitchLinear is None:
        # mlx_lm.models.switch_layers not available — can't be MoE.
        return False

    moe_types = (_SwitchLinear,)
    if _SwitchGLU is not None:
        moe_types = (_SwitchLinear, _SwitchGLU)

    for _name, module in model.named_modules():
        if isinstance(module, moe_types):
            return True
    return False


def get_moe_config(model: nn.Module) -> Optional[Dict[str, Any]]:
    """
    Extract MoE configuration from *model*.

    MLX-LM models store architecture hyper-parameters in ``model.args``
    (a ``SimpleNamespace`` or dataclass).  This function reads the
    standard MoE fields and returns them as a plain dict.

    Args:
        model: An MLX-LM model.

    Returns:
        A dict with MoE configuration keys, or ``None`` for dense models.
    """
    if not detect_moe_model(model):
        return None

    args = getattr(model, "args", None)
    if args is None:
        # Model is MoE but has no .args — return minimal info.
        return {"is_moe": True}

    config: Dict[str, Any] = {"is_moe": True}
    for key in (
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
        "num_dense_layers",
        "moe_intermediate_size",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    return config


def compute_lora_memory_savings(
    model: nn.Module,
    lora_rank: int,
    lora_layers: int
) -> Dict[str, float]:
    """
    Pure function to estimate LoRA memory savings.
    
    Computes the theoretical memory reduction from using LoRA
    instead of full fine-tuning.
    
    Args:
        model: Original model
        lora_rank: LoRA rank parameter
        lora_layers: Number of LoRA layers
        
    Returns:
        Dictionary with memory statistics
    """
    # Estimate parameter counts with error handling
    try:
        if hasattr(model, 'parameters'):
            # Try to count parameters, handling different return types
            params = list(model.parameters())
            total_params = 0
            for p in params:
                if hasattr(p, 'size'):
                    total_params += p.size
                elif hasattr(p, 'shape'):
                    # Calculate size from shape
                    size = 1
                    for dim in p.shape:
                        size *= dim
                    total_params += size
        else:
            # Fallback: rough estimate for 0.6B model
            total_params = 600_000_000
    except Exception:
        # Final fallback
        total_params = 600_000_000
    
    # Rough estimate of LoRA parameters
    # Each LoRA layer adds rank * (input_dim + output_dim) parameters
    # This is a simplified calculation
    estimated_lora_params = lora_layers * lora_rank * 2 * 4096  # Rough estimate
    
    if total_params == 0:
        total_params = 600_000_000  # Prevent division by zero
    
    memory_ratio = estimated_lora_params / total_params
    memory_savings = (1 - memory_ratio) * 100
    
    return {
        "total_parameters": total_params,
        "estimated_lora_parameters": estimated_lora_params,
        "memory_ratio": memory_ratio,
        "memory_savings_percent": memory_savings
    }


# Composed function for creating LoRA-enabled training setup
def create_lora_setup(
    model: nn.Module,
    lora_config: Dict[str, Any],
    auto_reload: bool = True,
    adapter_save_path: str = "./lora_adapters.safetensors"
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Set up LoRA training with automatic adapter management.
    
    When auto_reload=True (default), the returned model automatically 
    handles adapter saving/reloading during training. This is invisible
    to the user - just use the model normally with Trainer.
    
    Args:
        model: Base MLX model
        lora_config: LoRA configuration parameters
        auto_reload: Whether to enable automatic adapter management
        adapter_save_path: Where to save/load adapters
        
    Returns:
        (lora_model, memory_stats): LoRA-enabled model and memory statistics
    """
    # Apply LoRA adapters
    lora_model = apply_lora(
        model=model,
        lora_layers=lora_config.get("lora_layers", 8),
        lora_rank=lora_config.get("lora_rank", 8),
        lora_scale=lora_config.get("lora_scale", 20.0),
        lora_dropout=lora_config.get("lora_dropout", 0.0)
    )
    
    # Freeze base parameters
    lora_model = freeze_base(lora_model)
    
    # Compute memory savings
    memory_stats = compute_lora_memory_savings(
        model=model,
        lora_rank=lora_config.get("lora_rank", 8),
        lora_layers=lora_config.get("lora_layers", 8)
    )
    
    # Add auto-reload metadata to model if enabled
    if auto_reload:
        # Store metadata on the model for Trainer to detect
        lora_model._auto_reload_path = adapter_save_path
        lora_model._is_auto_reload_lora = True
        print(f"LoRA auto-reload enabled: {adapter_save_path}")
    else:
        lora_model._is_auto_reload_lora = False
    
    return lora_model, memory_stats


# Real quantization implementation using MLX-LM
def apply_quantization_to_model(
    model: nn.Module,
    config: dict,
    bits: int = 4,
    group_size: int = 64
) -> nn.Module:
    """
    Pure function to apply real quantization for QLoRA using MLX-LM utilities.
    
    This function quantizes the base model weights to reduce memory
    usage even further when combined with LoRA.
    
    Args:
        model: MLX model to quantize
        config: Model configuration dictionary
        bits: Quantization bits (4, 6, or 8)
        group_size: Quantization group size
        
    Returns:
        Quantized model
    """
    try:
        from mlx_lm.utils import quantize_model
        import inspect
        
        print(f"Applying real {bits}-bit quantization...")
        print(f"  Group size: {group_size}")
        print(f"  Expected memory reduction: ~{8/bits:.1f}x")

        # MLX-LM changed quantize_model kwargs:
        # old: q_group_size/q_bits, new: group_size/bits.
        quantize_params = set(inspect.signature(quantize_model).parameters.keys())
        quantize_kwargs = {
            "model": model,
            "config": config,
            "quant_predicate": None,  # Quantize all eligible layers
        }
        if {"q_group_size", "q_bits"}.issubset(quantize_params):
            quantize_kwargs["q_group_size"] = group_size
            quantize_kwargs["q_bits"] = bits
        elif {"group_size", "bits"}.issubset(quantize_params):
            quantize_kwargs["group_size"] = group_size
            quantize_kwargs["bits"] = bits
        else:
            raise TypeError(
                f"Unsupported quantize_model signature: {sorted(quantize_params)}"
            )

        quantized_model, updated_config = quantize_model(**quantize_kwargs)
        
        print("Real quantization applied successfully")
        return quantized_model
        
    except ImportError:
        print("Warning: MLX-LM quantization not available, skipping quantization")
        return model
    except Exception as e:
        print(f"Warning: Quantization failed: {e}, using original model")
        return model


# Complete QLoRA setup function
def create_qlora_setup(
    model: nn.Module,
    lora_config: Dict[str, Any],
    quantization_config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Pure function to set up QLoRA (quantized LoRA) training.
    
    Combines quantization and LoRA for maximum memory efficiency.
    
    Args:
        model: Base MLX model
        lora_config: LoRA configuration
        quantization_config: Quantization configuration
        
    Returns:
        (qlora_model, memory_stats): QLoRA-enabled model and statistics
    """
    # Create default model config for quantization
    model_config = {
        "model_type": "unknown",
        "vocab_size": 32000,  # Default vocab size
        "hidden_size": 4096,   # Default hidden size
    }
    
    # Apply quantization first using real MLX-LM quantization
    quantized_model = apply_quantization_to_model(
        model=model,
        config=model_config,
        bits=quantization_config.get("bits", 4),
        group_size=quantization_config.get("group_size", 64)
    )
    
    # Then apply LoRA to quantized model
    qlora_model, memory_stats = create_lora_setup(
        model=quantized_model,
        lora_config=lora_config
    )
    
    # Update memory statistics to reflect quantization
    quantization_factor = 8 / quantization_config.get("bits", 4)
    memory_stats["quantization_factor"] = quantization_factor
    memory_stats["total_memory_savings"] = (
        memory_stats["memory_savings_percent"] + 
        (quantization_factor - 1) * 100 / quantization_factor
    )
    
    print(f"QLoRA setup complete - estimated {memory_stats['total_memory_savings']:.1f}% memory savings")
    
    return qlora_model, memory_stats
