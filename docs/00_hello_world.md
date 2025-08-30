# 00. Hello, World! (First Principles Guide)

This is your quickest path to “it runs”, plus the why behind it.

## Questions To Ask First

- What is the smallest thing I can run to see a reward?
- How does a policy interact with an environment?
- What does `reset()` give me; what must `step()` return?
- Why does TextPolicy care about MLX arrays and batching?
- How do I avoid slow code (Big-O pitfalls) on Apple Silicon?

## How It Works (First Principles)

- An *environment* emits an observation (`reset()`), and evaluates an action (`step()`) to produce a reward.
- A *policy* maps observation → action. For text, the action is usually token IDs that decode to text.
- A *rollout* is a sequence collected by the runner: `obs → action → step → store`, batched for MLX efficiency.
- A *trainer* uses a buffer of rollouts to improve the policy (e.g., GRPO/GSPO) using pure functions.
- TextPolicy standardizes interfaces so components compose cleanly and efficiently on MLX.

Key contracts:
- `reset()` returns `(observation, info)`.
- `step(action)` returns a dict with keys: `observation`, `reward`, `terminated`, `truncated`, `info`.

Why MLX patterns matter:
- MLX runs fast when you avoid per-item Python work; batch conversions and use MLX ops in hot loops.
- Keep data transfers minimal (host↔device). Convert to MLX once per step/batch.

## Your First Reward (no models required)

```
def length_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    return float(len(completion.split()))
```

## Your First Environment (dummy tokenizer)

```
from textpolicy.environment.text_generation import TextGenerationEnv

class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 256 for c in text]
    def decode(self, ids):
        return ''.join(chr(int(i) % 256) for i in ids)

env = TextGenerationEnv(["Hello"], length_reward, tokenizer=DummyTokenizer())
obs, info = env.reset()
result = env.step("a b c")
print(result["reward"])  # 3.0
```

Notes:
- `TextGenerationEnv` requires a tokenizer to decode actions back to text for your reward.
- `result` is a dict; `result["terminated"]` is `True` for this single-turn task.

## Your First Rollout (trivial policy)

```
import mlx.core as mx  # Requires MLX installed
from textpolicy.rollout.runner import RolloutRunner
from textpolicy.rollout.strategy import create_strategy

def policy(obs_mx, deterministic=False):
    # Return tokens for 'a b c' as MLX array
    return mx.array([97, 32, 98, 32, 99], dtype=mx.int32), {}

strategy = create_strategy('grpo')
runner = RolloutRunner(env, policy=policy, strategy=strategy, max_steps=2)
buffer = runner.collect()
print(len(buffer.episodes), buffer.episodes[0].rewards)
```

What just happened:
- The runner calls `env.reset()` once, then loops `policy -> env.step() -> store`.
- The runner accepts dict or tuple step results and normalizes them internally.
- Rewards are simple word counts from our `length_reward`.

See also:
- examples/01_hello_reward.py for reward basics.
- examples/06_minimal_training.py for the training pipeline shape.
- examples/08_real_rl_training.py for real GRPO training with mlx-lm.

## Next: Real Models with mlx-lm

```
from textpolicy.generation.mlx_generation import load_model, create_policy
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
policy_fn = create_policy(model, tokenizer, {"max_tokens": 25, "temperature": 0.7})
```

Use `RolloutCoordinator` for a full loop with `Trainer` (see docs/05_rollout_training.md and examples/08_real_rl_training.py).

## Performance Must-Knows (Avoid Big-O errors)

- Batch conversions: `mx.array(np_batch)` once vs converting each item.
- Contiguity: `np.ascontiguousarray` before `mx.array` for best throughput.
- Avoid quadratic growth: don’t repeatedly `+` strings or lists in loops; join or preallocate.
- Keep compute in MLX; store in Python lists; convert once per step.

## Common Errors (and why)

- Missing tokenizer in `TextGenerationEnv`: it must decode actions → text to score rewards.
- Missing model/tokenizer in `TextGenerationEnvironment`: constructor raises `ValueError` — it needs both for real generation.
- mlx-lm not installed: `load_model` raises `ImportError` — install `mlx` and `mlx-lm`.

## Learn By Doing

- Start with examples/01_hello_reward.py to master reward functions.
- Progress to examples/06_minimal_training.py for the pipeline shape.
- Run examples/08_real_rl_training.py when you’re ready for real models.
