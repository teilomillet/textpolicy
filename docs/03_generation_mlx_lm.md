# 03. Generation with mlx-lm

Use `textpolicy.generation.mlx_generation` to load models and create policies.

Definition: A *policy* is a function that maps an observation to an action and optional extra data for training (e.g., log-probabilities).

## Load model and tokenizer

```
from textpolicy.generation.mlx_generation import load_model
model, tokenizer = load_model("Qwen/Qwen3-0.6B")
```

This function raises an ImportError if mlx-lm is not installed.

## Create a policy for rollouts

```
from textpolicy.generation.mlx_generation import create_policy
policy_fn = create_policy(model, tokenizer, {"max_tokens": 25, "temperature": 0.7})
```

Policy returns `(action_tokens, extra)` where `action_tokens` is an MLX array of token IDs.
Examples: see examples/08_real_rl_training.py and 09_length_reduction_training.py.

## Encode/Decode helpers

```
from textpolicy.generation.mlx_generation import encode, decode
tok = encode(tokenizer, "Hello")
txt = decode(tokenizer, tok)
```

See examples/08_real_rl_training.py for end-to-end usage.
