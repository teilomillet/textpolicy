# TextPolicy

Reinforcement learning toolkit for text generation on MLX (Apple Silicon).
TextPolicy provides algorithms (GRPO/GSPO), text-generation environments, a rollout runner,
reward functions with a decorator registry, and LoRA/QLoRA utilities.

## Install (uv)

```bash
uv add textpolicy
```

Optional model integration:

```bash
uv add mlx mlx-lm
```

## Quickstart

Working example using a real model and tokenizer (mlx-lm required):

```python
import mlx.core as mx
import textpolicy as tp
from textpolicy import load_model, create_policy
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.rollout import RolloutRunner, create_strategy

# 1) Load model and tokenizer (mlx-lm)
model, tokenizer = load_model("Qwen/Qwen3-0.6B")

# 2) Create a policy (controls generation)
generation_params = {"max_tokens": 25, "temperature": 0.7}
policy_fn = create_policy(model, tokenizer, generation_params)

# 3) Define a reward function (env uses this to score responses)
@tp.reward
def length_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    return float(len(completion.split()))

# 4) Create an environment (requires a tokenizer)
env = TextGenerationEnv(["What is AI?"], length_reward, tokenizer=tokenizer)

# 5) Collect one rollout step
strategy = create_strategy('grpo')
runner = RolloutRunner(env, policy=policy_fn, strategy=strategy, max_steps=1)
buffer = runner.collect()
print(len(buffer.episodes))
```

Docs:
- Quickstart: `docs/QUICKSTART_UV.md`
- LoRA/QLoRA: `docs/10_lora_qlora.md`
- Full index: `docs/index.md`

FAQ:
- Do I need a model? 
    - Yes for generation with `create_policy`. 
    Use `load_model()` (mlx‑lm) to get `(model, tokenizer)`. 
    For reward‑only code (no generation), a model is not required.
- Do I need a tokenizer? 
    - Yes. 
    Both `TextGenerationEnv` and `TextGenerationEnvironment` require a tokenizer. 
    `load_model()` returns one for mlx‑lm models.
- How do I control generation? 
    - Pass `generation_params` to `create_policy` (for example, `max_tokens`, `temperature`, `top_p`, `repetition_penalty`).
- What does `step()` return? 
    - A dict with `observation`, `reward`, `terminated`, `truncated`, `info`. The runner enforces this.

## GPU Training (Tinker)

Run our advantage pipeline on GPUs via the [Tinker](https://tinker.dev) API:

```bash
export TINKER_API_KEY=<your-key>

# Smoke test (5 steps)
uv run python -m textpolicy.tinker.train_math --max-steps 5 --group-size 4

# Full run (500 steps)
uv run python -m textpolicy.tinker.train_math --max-steps 500

# A/B campaign: baseline GRPO vs full pipeline
uv run python scripts/tinker_campaign.py --max-steps 100 --execute
```

Docs:
- Training guide: `docs/12_tinker_gpu.md`
- Campaign methodology: `docs/13_tinker_campaigns.md`

Examples:
- 01–06: reward functions, batch processing, minimal training
- 08: GRPO training with rollout + buffer
- 09–10: length reduction (GRPO/GSPO)
- 11: LoRA/QLoRA configuration
- 12: TinyLoRA + GTPO + HICRA reasoning run (`experiments/countdown_reasoning_lora.py`)
