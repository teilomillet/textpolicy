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


class _MissingKeysEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        # Missing required keys: terminated, truncated, info
        return {
            "observation": 0,
            "reward": 1.0,
        }


class _NonBoolTerminatedEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        return {
            "observation": 0,
            "reward": 1.0,
            "terminated": 1,  # invalid type
            "truncated": False,
            "info": {},
        }


class _CounterRewardDoneEnv:
    def __init__(self):
        self.reward_counter = 0

    def reset(self):
        return 0, {}

    def step(self, action):
        self.reward_counter += 1
        return {
            "observation": 0,
            "reward": float(self.reward_counter),
            "terminated": True,
            "truncated": False,
            "info": {},
        }


class _CorrectnessInfoEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        return {
            "observation": 0,
            "reward": -0.5,
            "terminated": True,
            "truncated": False,
            "info": {"is_correct": True},
        }


def _counter_reward_env_fn():
    return _CounterRewardDoneEnv()


def _counter_policy_fn():
    def policy(obs, deterministic=False):
        import mlx.core as mx
        return mx.array(0), {}

    return policy


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


def test_runner_collect_preserves_is_correct_from_info():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _CorrectnessInfoEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy, strategy=strategy, max_steps=1)

    buf = runner.collect()
    assert len(buf.episodes) == 1
    episode = buf.episodes[0]
    assert episode.is_correct == [True]


def test_runner_collect_rejects_missing_required_step_keys():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    runner = RolloutRunner(
        _MissingKeysEnv(),
        policy=_policy,
        strategy=create_strategy("grpo"),
        max_steps=1,
    )

    with pytest.raises(KeyError, match="missing required keys"):
        runner.collect()


def test_runner_collect_rejects_non_bool_terminated_truncated():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    runner = RolloutRunner(
        _NonBoolTerminatedEnv(),
        policy=_policy,
        strategy=create_strategy("grpo"),
        max_steps=1,
    )

    with pytest.raises(TypeError, match="must be booleans"):
        runner.collect()


def test_runner_collect_single_step_uses_single_observation_shape():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _VectorObsEnv()
    strategy = create_strategy("grpo")
    runner = RolloutRunner(env, policy=_policy_requires_vector_obs, strategy=strategy, max_steps=1)

    buf = runner.collect()
    assert len(buf.episodes) == 1


def test_runner_collect_repeated_calls_return_fresh_buffer_and_data():
    from textpolicy.rollout.runner import RolloutRunner
    from textpolicy.rollout.strategy import create_strategy

    env = _CounterRewardDoneEnv()
    runner = RolloutRunner(
        env,
        policy=_policy,
        strategy=create_strategy("grpo"),
        max_steps=3,
    )

    buf1 = runner.collect()
    rewards1 = [float(ep.rew[0]) for ep in buf1.episodes]

    buf2 = runner.collect()
    rewards2 = [float(ep.rew[0]) for ep in buf2.episodes]

    assert buf1 is not buf2
    assert rewards1 == [1.0, 2.0, 3.0]
    assert rewards2 == [4.0, 5.0, 6.0]
    assert [float(ep.rew[0]) for ep in buf1.episodes] == [1.0, 2.0, 3.0]


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


@pytest.mark.integration
def test_rollout_coordinator_multiprocess_collect_returns_fresh_batches():
    from textpolicy.rollout.rollout import RolloutCoordinator

    coordinator = RolloutCoordinator(
        env_fn=_counter_reward_env_fn,
        policy_fn=_counter_policy_fn,
        algorithm="grpo",
        num_workers=1,
        max_steps=3,
    )
    try:
        batch1 = coordinator.collect()
        rewards1 = [float(ep.rew[0]) for ep in batch1.episodes]

        batch2 = coordinator.collect()
        rewards2 = [float(ep.rew[0]) for ep in batch2.episodes]
    finally:
        coordinator.close()

    assert batch1 is not batch2
    assert rewards1 == [1.0, 2.0, 3.0]
    assert rewards2 == [4.0, 5.0, 6.0]
    # Verify batch1 was NOT mutated by the second collect() call.
    assert [float(ep.rew[0]) for ep in batch1.episodes] == [1.0, 2.0, 3.0]
