# textpolicy/generation/__init__.py
"""
Pure MLX-LM text generation functions for RL training.

Following TextPolicy design principles:
- Pure function composition over classes
- Zero abstraction cost for MLX optimization
- Direct integration with GRPO trainer
- LoRA/QLoRA support for memory efficiency

All functions are pure and composable with our existing trainer system.
"""

# Core MLX-LM generation functions  
from .mlx_generation import (
    load_model,
    generate_tokens,
    compute_logprobs,
    encode,
    decode,
    create_policy,
    compute_reward,
)

# LoRA/QLoRA pure functions
from .lora import (
    apply_lora,
    freeze_base,
    extract_params,
    merge_weights,
    create_lora_setup,
    create_qlora_setup
)

# LoRA utility functions (advanced use only)
from .reload import (
    save_adapters,
    reload_model
)

__all__ = [
    # Core generation functions
    "load_model",
    "generate_tokens", 
    "compute_logprobs",
    "encode",
    "decode", 
    "create_policy",
    "compute_reward",
    
    # LoRA functions
    "apply_lora",
    "freeze_base",
    "extract_params", 
    "merge_weights",
    "create_lora_setup",
    "create_qlora_setup",
    
    # Advanced LoRA utilities
    "save_adapters",
    "reload_model",
]