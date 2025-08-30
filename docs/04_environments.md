# 04. Environments

TextPolicy provides two text-generation environments:

- `TextGenerationEnv`: lightweight env for simple RL tasks using prompts + custom reward.
- `TextGenerationEnvironment`: richer evaluation env requiring a real model/tokenizer.

## Interface

- `reset() -> (observation, info)`
- `step(action) -> {"observation", "reward", "terminated", "truncated", "info"}`

Definition: An *environment* exposes `reset()` and `step()` and defines how actions produce rewards. The rollout runner also accepts gym-style tuples, but environments should implement the dict return for consistency.

## Simple env example

```
from textpolicy.environment.text_generation import TextGenerationEnv

class DummyTokenizer:
    def encode(self, text):
        return [ord(c) % 256 for c in text]
    def decode(self, ids):
        return ''.join(chr(int(i) % 256) for i in ids)

def reward_fn(prompt, completion, example, **kwargs):
    return float(len(completion.split()))

env = TextGenerationEnv(["Hello"], reward_fn, tokenizer=DummyTokenizer())
obs, info = env.reset()
res = env.step("a b c")
```

## Evaluation env

`TextGenerationEnvironment` requires `model` and `tokenizer` and will raise a `ValueError` if missing. It builds a task suite and computes structured rewards (length, keywords, coherence).

Task suites are discoverable via the registry. Use `register_task_suite`, `list_task_suites`, and `get_task_suite` to manage suites.

## Task Suite Registry

Intent: register and discover task suites by name without modifying the environment class.

- API (from `textpolicy.environment`): `register_task_suite`, `list_task_suites`, `get_task_suite`.
- The environment consults the registry first, then falls back to defaults (`basic`, `challenging`).

Example: define and register a custom suite

```
from textpolicy.environment import register_task_suite, list_task_suites, get_task_suite
from textpolicy.environment.text_generation import TextGenerationTask

def my_suite():
    return [
        TextGenerationTask(
            prompt="Summarize reinforcement learning in two sentences.",
            target_keywords=["agent", "reward"],
            target_length_range=(20, 35),
            difficulty=0.5,
            category="summary",
            evaluation_criteria={"keyword_weight": 0.5, "length_weight": 0.5},
        )
    ]

register_task_suite("my_suite", my_suite)
print(list_task_suites())      # includes 'my_suite'
print(len(get_task_suite("my_suite") or []))
```

Use with `TextGenerationEnvironment` by passing `task_suite="my_suite"`.
