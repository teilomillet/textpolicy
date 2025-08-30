#!/usr/bin/env python3
"""
11: LoRA/QLoRA Training (MLX)

Demonstrates how to configure LoRA or QLoRA with TextPolicy’s trainer.
The script is safe to run without mlx-lm (prints a demo message and exits).

Real run (requires mlx-lm and a model ID):
- Set MODEL_ID (e.g., Qwen/Qwen3-0.6B) in the environment
- Uncomment QLoRA section to quantize before applying LoRA

This example keeps settings small to illustrate usage.
"""

import os


def main():
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
    except Exception:
        print("MLX not available; demo only.")
        return

    try:
        # Core generation + LoRA helpers
        from textpolicy.generation.mlx_generation import load_model, create_policy
        from textpolicy.generation.lora import create_lora_setup, create_qlora_setup
        from textpolicy.training import Trainer
        from textpolicy.algorithms import grpo
    except Exception:
        print("mlx-lm not available; LoRA example is a usage reference only.")
        return

    model_id = os.environ.get("MODEL_ID")
    if not model_id:
        print("Set MODEL_ID to a model (e.g., Qwen/Qwen3-0.6B) to run this example.")
        return

    # 1) Load base model
    print(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id)

    # 2) Configure LoRA or QLoRA
    lora_config = {"lora_layers": 8, "lora_rank": 8, "lora_scale": 20.0}

    # LoRA (default)
    model, memory_stats = create_lora_setup(model, lora_config)

    # QLoRA (optional): quantize, then apply LoRA
    # quant_config = {"bits": 4, "group_size": 64}
    # model, memory_stats = create_qlora_setup(model, lora_config, quant_config)

    print("Memory stats:", memory_stats)

    # 3) Create Trainer (uses GRPO advantages + policy loss)
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optim.Adam(learning_rate=5e-6),
        compile_training=True,
    )

    # 4) Create policy for rollout
    policy_fn = create_policy(model, tokenizer, {"max_tokens": 25, "temperature": 0.7})

    # At this point, connect rollout → buffer → trainer (see examples 08–10)
    # This example focuses on LoRA/QLoRA configuration.
    print("LoRA/QLoRA setup complete. Proceed with rollout and training as in example 08.")


if __name__ == "__main__":
    main()

