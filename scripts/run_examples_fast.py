#!/usr/bin/env python3
"""
Fast smoke-run for examples 08/09/10 without network or large models.

This script reuses each example's reward and intent but swaps heavy pieces
with lightweight stubs:
- Mock model/tokenizer and trivial policy producing fixed tokens
- Single-process rollout with small max_steps
- One or two trainer steps using GRPO/GSPO loss functions

Goal: validate wiring and runtime behavior in < 8 minutes.
"""

from __future__ import annotations

import sys


def _make_dummy_policy():
    import mlx.core as mx

    def policy(obs_mx, deterministic=False):
        # Return tokens for 'a b c'
        return mx.array([97, 32, 98, 32, 99], dtype=mx.int32), {}

    return policy


def _dummy_tokenizer():
    class T:
        def encode(self, text):
            return [ord(c) % 256 for c in text]

        def decode(self, ids):
            return "".join(chr(int(i) % 256) for i in ids)

    return T()


def _run_fast_training(reward_fn, steps: int = 2):
    import mlx.optimizers as optim
    import mlx.core as mx
    from textpolicy.environment.text_generation import TextGenerationEnv
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy
    from textpolicy.buffer import Buffer
    from textpolicy.training import Trainer
    from textpolicy.algorithms import grpo

    tokenizer = _dummy_tokenizer()
    env = TextGenerationEnv([
        "What is AI?",
        "Explain ML"
    ], reward_fn, max_tokens=10, tokenizer=tokenizer)

    policy = _make_dummy_policy()
    strategy = create_strategy('grpo')

    runner = RolloutRunner(env, policy=policy, strategy=strategy, max_steps=steps)
    rollout_buffer = runner.collect()

    # Minimal buffer+trainer pass
    buffer = Buffer(max_episodes=10)
    for ep in rollout_buffer.episodes:
        buffer.add_episode_from_dict(ep.to_dict())

    # Dummy MLX model compatible with nn/optimizer APIs
    import mlx.nn as nn

    # Use a tiny linear layer to satisfy optimizer/model API. The training
    # loss function handles shape mismatches by falling back to zeros.
    dummy_model = nn.Linear(1, 1)
    trainer = Trainer(
        model=dummy_model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optim.Adam(learning_rate=1e-4),
        buffer=buffer,
        data_selector_fn=grpo.select_all_data,
        compile_training=False,
    )
    metrics = trainer.train()
    return metrics


def run_08_fast():
    import sys
    sys.path.append('examples')
    from importlib import import_module
    stable_reward = import_module('08_real_rl_training').stable_reward
    print("[08] Fast run start")
    m = _run_fast_training(stable_reward, steps=2)
    print("[08] Fast run metrics:", m)


def run_09_fast():
    import sys
    sys.path.append('examples')
    from importlib import import_module
    length_reduction_reward = import_module('09_length_reduction_training').length_reduction_reward
    print("[09] Fast run start")
    m = _run_fast_training(length_reduction_reward, steps=3)
    print("[09] Fast run metrics:", m)


def run_10_fast():
    import sys
    sys.path.append('examples')
    from importlib import import_module
    length_reward = import_module('10_gspo_length_reduction_training').length_reward
    print("[10] Fast run start")
    m = _run_fast_training(length_reward, steps=3)
    print("[10] Fast run metrics:", m)


def main():
    try:
        run_08_fast()
        run_09_fast()
        run_10_fast()
        print("All fast runs completed.")
    except Exception as e:
        print("Fast run failed:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
