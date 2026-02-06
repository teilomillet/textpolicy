# 05. Rollouts and Training

Definition: A *rollout* is a collected sequence of (observation, action, reward, next observation, done) transitions produced by interacting a policy with an environment.

Definition: A *buffer* stores rollouts for training. A *strategy* configures action selection and data storage for a specific algorithm.

## RolloutCoordinator (single-process)

```
from textpolicy.rollout import RolloutCoordinator
from textpolicy.generation.mlx_generation import load_model, create_policy
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.algorithms import grpo
from textpolicy.training import Trainer

model, tokenizer = load_model("Qwen/Qwen3-0.6B")
policy_fn = create_policy(model, tokenizer, {"max_tokens": 25})

def create_env():
    prompts = ["What is AI?", "Explain neural networks"]
    def reward_fn(prompt, completion, example, **kwargs):
        return float(len(completion.split()))
    return TextGenerationEnv(prompts, reward_fn, tokenizer=tokenizer)

coordinator = RolloutCoordinator(create_env, lambda: policy_fn, algorithm='grpo', num_workers=0, max_steps=100)
buffer = coordinator.collect()

trainer = Trainer(model=model,
                  advantage_fn=grpo.compute_advantages,
                  loss_fn=grpo.policy_loss,
                  optimizer=None,  # configure your optimizer
                  buffer=buffer)  # on_policy=True by default: clears buffer after each train()
# metrics = trainer.train()
```

See examples/08_real_rl_training.py and 09_length_reduction_training.py.

## RolloutRunner (direct)

```
from textpolicy.rollout.runner import RolloutRunner
runner = RolloutRunner(create_env(), policy_fn, create_strategy('grpo'), max_steps=50)
buf = runner.collect()
```

## Tips

- Keep policies pure and fast; avoid Python work per token.
- Batch MLX conversions where possible.
- Use `env_fn` when multiprocessing to instantiate env/model in each process.

## Step Result Standardization

Intent: use one return shape for `Environment.step`.

- Required: dict with keys `observation`, `reward`, `terminated`, `truncated`, `info`.
- The runner enforces dict-shaped step results and raises a TypeError for tuples.
- Update environments to return dicts only.
