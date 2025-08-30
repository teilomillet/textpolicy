# 07. Testing and Integration

## Running tests

```
uv run pytest                 # all tests
uv run pytest -m unit         # fast units
uv run pytest -m integration  # integration only
```

## Markers

- unit, integration, algorithm, reward, slow, requires_model

## Minimal E2E example

See tests/test_integration_e2e_training.py for a tiny rollout test using TextGenerationEnv + dummy tokenizer.

## Writing efficient tests

- Keep integration tests under ~30s.
- Use dummy tokenizers/environments; avoid model downloads.
- Prefer property assertions (e.g., reward > 0) over brittle exact matches.
- Skip tests gracefully when MLX is unavailable.

