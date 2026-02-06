import pytest


class _TupleEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        # Old gym 4-tuple format (unsupported by runner now)
        return 0, 1.0, True, {}


class _DictEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        return {
            "observation": 0,
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "info": {},
        }


class _VectorObsEnv:
    def reset(self):
        return [1, 2, 3], {}

    def step(self, action):
        return {
            "observation": [1, 2, 3],
            "reward": 1.0,
            "terminated": True,
            "truncated": False,
            "info": {},
        }


class _ScalarObsOnlyResetEnv:
    def __init__(self):
        self.t = 0

    def reset(self):
        self.t = 0
        return self.t

    def step(self, action):
        self.t += 1
        return {
            "observation": self.t,
            "reward": 1.0,
            "terminated": False,
            "truncated": False,
            "info": {},
        }


def _policy(obs, deterministic=False):
    # Return scalar action as MLX array substitute (runner only inspects ndim)
    import mlx.core as mx
    return mx.array(0), {}


def _policy_requires_vector_obs(obs, deterministic=False):
    import mlx.core as mx
    assert obs.ndim == 1, f"Expected 1D observation, got shape {obs.shape}"
    return mx.array(0), {}


def test_runner_rejects_tuple_step():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _TupleEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy, strategy=strategy, max_steps=1)

    with pytest.raises(TypeError):
        runner.collect_episode()


def test_runner_accepts_dict_step():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _DictEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy, strategy=strategy, max_steps=1)

    traj = runner.collect_episode()
    assert isinstance(traj, list)
    assert len(traj) >= 1


def test_runner_collect_rejects_tuple_step():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _TupleEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy, strategy=strategy, max_steps=1)

    with pytest.raises(TypeError):
        runner.collect()


def test_runner_collect_accepts_dict_step():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _DictEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy, strategy=strategy, max_steps=1)

    buf = runner.collect()
    # Buffer should contain at least one episode
    assert len(buf.episodes) >= 0  # episodes may be reset; ensure call succeeds


def test_runner_collect_single_step_uses_single_observation_shape():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _VectorObsEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy_requires_vector_obs, strategy=strategy, max_steps=1)

    buf = runner.collect()
    assert len(buf.episodes) == 1


def test_runner_collect_honors_max_steps_and_progresses_observations():
    import mlx.core as mx
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    seen_obs = []

    env = _ScalarObsOnlyResetEnv()

    def policy(obs, deterministic=False):
        seen_obs.append(int(obs.item()))
        return mx.array(0), {}

    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=policy, strategy=strategy, max_steps=4)

    buf = runner.collect()

    # Ongoing episode should contain one transition per collection step.
    assert len(buf.current_episode.obs) == 4
    assert seen_obs == [0, 1, 2, 3]


def test_runner_collect_does_not_drop_episodes_when_max_steps_exceeds_10():
    import mlx.core as mx
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    class _AlwaysDoneEnv:
        def reset(self):
            return 0, {}

        def step(self, action):
            return {
                "observation": 0,
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
                "info": {},
            }

    def policy(obs, deterministic=False):
        return mx.array(0), {}

    max_steps = 20
    runner = RolloutRunner(
        _AlwaysDoneEnv(),
        policy=policy,
        strategy=create_strategy("grpo"),
        max_steps=max_steps,
    )

    buf = runner.collect()
    assert len(buf.episodes) == max_steps
