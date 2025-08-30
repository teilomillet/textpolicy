# 01. Setup

## Install

Use uv to run and test locally with the project’s virtualenv:

```
uv run python -c "import textpolicy; print(textpolicy.__name__)"
```

Install optional MLX dependencies for model integration and compiled execution:

```
pip install mlx mlx-lm
```

Notes:
- MLX/Neural Engine features require Apple Silicon.
- Without mlx-lm, TextPolicy still works for rewards and non-model paths; model-specific features will raise clear errors.

## Verify

```
uv run python - << 'PY'
import textpolicy
print('ok:', hasattr(textpolicy, 'length_reward'))
PY
```

