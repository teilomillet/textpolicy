# Integration Guide (MLX + Apple Silicon)

This guide outlines best practices for integrating TextPolicy in MLX-based training
workflows and for writing efficient, reliable integration tests.

## Environment Interface

- `reset()` returns `(observation, info)`.
- `step(action)` returns a dict containing: `observation`, `reward`, `terminated`, `truncated`, `info`.
- The rollout runner accepts both dict and tuple step results for compatibility, but environments should implement the dict form for consistency.

## MLX Performance Guidelines

- Minimize host↔device transfers. Convert to MLX arrays in batches (`mx.array(np_batch)`) instead of per item.
- Ensure contiguous memory before MLX conversion (`np.ascontiguousarray`).
- Use MLX ops within hot loops; avoid Python loops over tokens.
- Avoid O(n^2) patterns (e.g., repeated string/list concatenation). Preallocate or join.
- Keep buffers as Python lists for cheap storage; run compute (advantages, loss) in MLX.

## Type Conversion Patterns

- Adapters handle MLX↔NumPy conversions; do not duplicate conversions in policies.
- For discrete actions, return scalar MLX arrays or Python scalars; for sequences, return token ID arrays.
- Text generation envs use a tokenizer with `encode`/`decode`; ensure these methods are available.

## Error Handling

- `TextGenerationEnvironment` requires a valid `model` and `tokenizer`. Constructor raises `ValueError` if missing.
- `TextGenerationEnv` requires a `tokenizer` to decode actions for reward functions.
- When integrating external envs, wrap errors with actionable messages where possible.

## Minimal Integration Test Pattern

Use a lightweight environment with a dummy tokenizer and a trivial policy that returns token IDs.
Keep the number of steps small to ensure fast tests on CI.

```python
import pytest

@pytest.mark.integration
def test_e2e_minimal_rollout_grpo():
    try:
        import mlx.core as mx
    except Exception:
        pytest.skip("MLX not available")

    from textpolicy.environment.text_generation import TextGenerationEnv
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    class DummyTokenizer:
        def encode(self, text):
            return [ord(c) % 256 for c in text]
        def decode(self, ids):
            return ''.join(chr(int(i) % 256) for i in ids)

    def reward_fn(prompt, completion, example, **kwargs):
        return float(len(completion.split()))

    env = TextGenerationEnv(["Hello"], reward_fn, tokenizer=DummyTokenizer())

    def simple_policy(obs_mx):  # Return tokens for 'a b c'
        return mx.array([97, 32, 98, 32, 99], dtype=mx.int32), {}

    strategy = create_strategy('grpo')
    runner = RolloutRunner(env, policy=simple_policy, strategy=strategy, max_steps=2)
    buffer = runner.collect()

    assert len(buffer.episodes) >= 1
    assert all(r > 0 for r in buffer.episodes[0].rewards)
```

## Test Hygiene

- Use `@pytest.mark.integration` and keep runtime under ~30s for fast CI.
- Use deterministic seeds if asserting on values; prefer property-based assertions (e.g., `reward > 0`).
- Avoid network or heavy model downloads; prefer dummy tokenizers/environments.

