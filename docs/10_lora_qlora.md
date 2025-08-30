# LoRA and QLoRA (MLX)

Intent: fine-tune large models efficiently using LoRA/QLoRA on MLX; train with TextPolicy’s Trainer and reuse adapters.

## When to use

- Fine-tuning with limited memory.
- Fast iteration by saving and reloading only adapter weights.

## Core APIs

From `textpolicy.generation`:
- LoRA setup: `create_lora_setup(model, lora_config, auto_reload=True, adapter_save_path=...)`
- QLoRA setup: `create_qlora_setup(model, lora_config, quantization_config)`
- Adapter save/reload: `save_adapters(model, path)`, `reload_model(base_model_path, adapter_path)`
- Reloadable policy: `create_reloadable_policy(base_model_path, initial_adapter_path, generation_params)`
- Auto-reload helper: `create_auto_reload_setup(model_path, adapter_save_path, **generation_params)`

LoRA building blocks (advanced): `apply_lora`, `freeze_base`, `extract_params`, `merge_weights` (merge is a placeholder).

## LoRA setup + Trainer (minimal)

```
from textpolicy.generation.mlx_generation import load_model, create_policy
from textpolicy.generation.lora import create_lora_setup
from textpolicy.training import Trainer
from textpolicy.algorithms import grpo
import mlx.optimizers as optim

# 1) Load base model
model, tokenizer = load_model("Qwen/Qwen3-0.6B")

# 2) Enable LoRA adapters
lora_config = {"lora_layers": 8, "lora_rank": 8, "lora_scale": 20.0}
model, memory_stats = create_lora_setup(model, lora_config)

# 3) Create Trainer
trainer = Trainer(
    model=model,
    advantage_fn=grpo.compute_advantages,
    loss_fn=grpo.policy_loss,
    optimizer=optim.Adam(learning_rate=5e-6),
)

# Policy (for rollout)
policy_fn = create_policy(model, tokenizer, {"max_tokens": 25, "temperature": 0.7})
```

Training proceeds as usual with TextPolicy rollout → buffer → trainer.

## Auto-reload pattern (adapters)

After training, you save adapters and reload a fresh model for rollout. Use the helper to create a reloadable policy and a training wrapper.

```
from textpolicy.generation.reload import (
    create_auto_reload_setup,
    save_adapters,
)

policy_fn, reload_fn, lora_model, tokenizer = create_auto_reload_setup(
    model_path="Qwen/Qwen3-0.6B",
    adapter_save_path="./lora_adapters.safetensors",
    max_tokens=25, temperature=0.7,
)

# After a trainer step, save and reload
# save_adapters(trainer.model, "./lora_adapters.safetensors")
# reload_fn("./lora_adapters.safetensors")
```

This yields a policy tied to a base model path; the policy uses the latest adapters after each reload.

## QLoRA setup

Quantize the base model, then apply LoRA:

```
from textpolicy.generation.lora import create_qlora_setup

lora_config = {"lora_layers": 8, "lora_rank": 8}
quant_config = {"bits": 4, "group_size": 64}
model, memory_stats = create_qlora_setup(model, lora_config, quant_config)
```

Notes:
- Quantization uses MLX-LM utilities when available; otherwise it skips and returns the original model.
- `memory_stats` includes approximate savings.

## Saving and reloading adapters

```
from textpolicy.generation.reload import save_adapters, reload_model

# After training
save_adapters(trainer.model, "./lora_adapters.safetensors")

# Later (fresh process)
model, tokenizer = reload_model(
    base_model_path="Qwen/Qwen3-0.6B",
    adapter_path="./lora_adapters.safetensors",
)
```

## Merging weights

`merge_weights(model)` is a placeholder and depends on MLX-LM internals. Keep adapters separate for deployment unless you have a merge procedure.

## Practical tips

- Freeze base parameters (`freeze_base`) so only adapter weights update.
- Keep adapter files small and save them frequently.
- For rollouts, use a reloadable policy to ensure the generator uses updated adapters.

