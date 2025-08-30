# Quickstart with uv

Intent: install the package, import it as `tp`, and run the minimal APIs.

## Install

```
uv add textpolicy
```

Validate the installation:

```
uv run textpolicy validate           # human-readable
uv run textpolicy validate --json    # machine-readable
```

Programmatically:

```
from textpolicy import validate_installation as _v
_v()
```

## Import

```
import textpolicy as tp
```

## Minimal APIs

- Installation check: `tp.validate_installation()`
- Rewards: `tp.reward` (decorator), `tp.length_reward`, `tp.keyword_reward`, `tp.perplexity_reward`, `tp.accuracy_reward`
- Generation (mlx-lm): `tp.load_model`, `tp.create_policy`, `tp.encode`, `tp.decode`
- Environments: `tp.TextGenerationEnv`, `tp.TextGenerationEnvironment`
- Rollout: `textpolicy.rollout.RolloutRunner`, `textpolicy.rollout.create_strategy`
- Training: `tp.Trainer`

## Minimal end-to-end (no models)

Define a simple reward and run one rollout using a dummy tokenizer and a trivial policy.

```
import mlx.core as mx
import textpolicy as tp

# 1) Reward
@tp.reward
def length_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    return float(len(completion.split()))

# 2) Dummy tokenizer
class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 256 for c in text]
    def decode(self, ids):
        return ''.join(chr(int(i) % 256) for i in ids)

# 3) Environment
env = tp.TextGenerationEnv(["Hello"], length_reward, tokenizer=DummyTokenizer())

# 4) Policy: return tokens for 'a b c'
def policy(obs_mx, deterministic=False):
    return mx.array([97, 32, 98, 32, 99], dtype=mx.int32), {}

# 5) Rollout
from textpolicy.rollout import RolloutRunner
from textpolicy.rollout import create_strategy
runner = RolloutRunner(env, policy=policy, strategy=create_strategy('grpo'), max_steps=1)
buffer = runner.collect()
print(len(buffer.episodes))
```

## With mlx-lm (optional)

```
from textpolicy import load_model, create_policy
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
policy_fn = create_policy(model, tokenizer, {"max_tokens": 25, "temperature": 0.7})
```

Use `policy_fn` with the rollout runner as shown above. `TextGenerationEnv` requires a tokenizer.

## Task suites (evaluation)

Use the registry to manage text-generation task suites:

```
from textpolicy.environment import register_task_suite, list_task_suites, get_task_suite
from textpolicy.environment.text_generation import TextGenerationTask

def my_suite():
    return [TextGenerationTask(
        prompt="Summarize RL in two sentences.",
        target_keywords=["agent", "reward"],
        target_length_range=(20, 35),
        difficulty=0.5,
        category="summary",
        evaluation_criteria={"keyword_weight": 0.5, "length_weight": 0.5},
    )]

register_task_suite("my_suite", my_suite)
print(list_task_suites())
```

Pass `task_suite="my_suite"` to `TextGenerationEnvironment` to use a custom suite.

## Step result contract

`Environment.step` must return a dict with keys `observation`, `reward`, `terminated`, `truncated`, `info`. The runner enforces this and raises an error for tuples.

