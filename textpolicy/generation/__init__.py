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
    batch_generate_tokens,
    compute_logprobs,
    compute_logprobs_batched,
    compute_prompt_reuse_stats,
    encode,
    decode,
    create_policy,
    create_batched_policy,
    compute_reward,
)

# LoRA/QLoRA pure functions
from .lora import (
    apply_lora,
    freeze_base,
    extract_params,
    merge_weights,
    create_lora_setup,
    create_qlora_setup,
    detect_moe_model,
    get_moe_config,
)

# LoRA adapter reloading
from .reload import (
    save_adapters,
    reload_model,
    create_reloadable_policy,
    create_training_loop_with_reload,
    create_auto_reload_setup,
)

# LoRA memory planning
from .lora import (
    compute_lora_memory_savings,
    apply_quantization_to_model,
)

__all__ = [
    # Core generation functions
    "load_model",
    "generate_tokens",
    "batch_generate_tokens",
    "compute_logprobs",
    "compute_logprobs_batched",
    "compute_prompt_reuse_stats",
    "encode",
    "decode",
    "create_policy",
    "create_batched_policy",
    "compute_reward",
    
    # LoRA functions
    "apply_lora",
    "freeze_base",
    "extract_params",
    "merge_weights",
    "create_lora_setup",
    "create_qlora_setup",

    # MoE detection
    "detect_moe_model",
    "get_moe_config",
    
    # LoRA adapter reloading
    "save_adapters",
    "reload_model",
    "create_reloadable_policy",
    "create_training_loop_with_reload",
    "create_auto_reload_setup",

    # LoRA memory planning
    "compute_lora_memory_savings",
    "apply_quantization_to_model",
]
