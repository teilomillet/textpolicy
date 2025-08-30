import pytest


@pytest.mark.integration
def test_e2e_minimal_rollout_grpo():
    """
    Minimal end-to-end rollout + buffer collection using TextGenerationEnv
    with a dummy tokenizer and a trivial policy. This validates that
    the environment returns dict-shaped step results and the runner
    normalization path works as expected.
    
    Kept intentionally lightweight for CI (no external model downloads).
    """
    try:
        import mlx.core as mx  # type: ignore
    except Exception:
        pytest.skip("MLX not available")

    from textpolicy.environment.text_generation import TextGenerationEnv
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    class DummyTokenizer:
        def encode(self, text):
            return [ord(c) % 256 for c in text]

        def decode(self, ids):
            return "".join(chr(int(i) % 256) for i in ids)

    def reward_fn(prompt, completion, example, **kwargs) -> float:
        # Simple length reward in words
        return float(len(completion.split()))

    # Create simple environment
    env = TextGenerationEnv(["Hello"], reward_fn, tokenizer=DummyTokenizer())

    # Policy returns tokens that decode to 'a b c'
    def simple_policy(obs_mx, deterministic=False):
        return mx.array([97, 32, 98, 32, 99], dtype=mx.int32), {}

    strategy = create_strategy('grpo')
    runner = RolloutRunner(env, policy=simple_policy, strategy=strategy, max_steps=2)

    buffer = runner.collect()
    assert len(buffer.episodes) >= 1
    ep = buffer.episodes[0]
    # Episode stores rewards in `rew`
    assert len(ep.rew) >= 1
    assert all(r > 0 for r in ep.rew)
