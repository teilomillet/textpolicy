# 09. Command-Line Interface

Intent: provide a tiny, dependency-free CLI for quick health checks and CI use.

Overview
- Command: `textpolicy validate`
- Purpose: validate installation and core environment contract
- Exit codes: 0 = ok, 1 = fail, 2 = argument error

Usage
```
# Human-readable report
uv run textpolicy validate

# Machine-readable JSON for CI
uv run textpolicy validate --json

# Via Python module
uv run python -m textpolicy               # defaults to `validate`
uv run python -m textpolicy validate --json
```

What it checks
- MLX availability and version (required for policies)
- Gymnasium availability (for adapters)
- mlx-lm availability (optional)
- Minimal TextGenerationEnv reset/step contract and reward path using a dummy tokenizer

Notes
- This CLI is intentionally small. A config-driven runner can be added later.
- The CLI wraps `textpolicy.validate.validate_installation()` to keep behavior consistent.

