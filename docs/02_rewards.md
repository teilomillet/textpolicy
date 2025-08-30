# 02. Rewards

An *RL reward function* scores a (prompt, completion, example) triple and returns a float.

## Signature

```
def reward_fn(prompt: str, completion: str, example: dict, **kwargs) -> float:
    ...
```

Examples: see examples/01_hello_reward.py, 02_reward_decorator.py, 03_batch_processing.py, and 04_multiple_rewards.py.

## Reward registry

TextPolicy ships common rewards and a decorator/registry:

```
from textpolicy.rewards.registry import reward

@reward("my_reward")
def my_reward(prompt, completion, example, **kwargs):
    return float(len(completion.split()))
```

## Batch processing tip

Prefer vectorized operations (e.g., with MLX arrays) and avoid Python loops for large batches.
