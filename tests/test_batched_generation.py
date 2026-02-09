"""
Tests for batched text generation across episodes (Issue #26).
"""

from __future__ import annotations

import types
from typing import Any, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import pytest

import textpolicy.generation.mlx_generation as mlx_generation
from textpolicy.generation.mlx_generation import (
    _create_batched_decode_mask,
    _create_batched_prefill_mask,
    batch_generate_tokens,
    create_batched_policy,
    create_policy,
    generate_tokens,
)
from textpolicy.rollout.rollout import RolloutCoordinator
from textpolicy.rollout.runner import RolloutRunner
from textpolicy.rollout.strategy import create_strategy


class _DummyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    eos_token_ids = [2]

    def encode(self, text: str) -> List[int]:
        return [3 + (ord(c) % 11) for c in text][:16]

    def decode(self, token_ids: List[int]) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"<tok_{token_id}>"


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 32, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def __call__(self, x, mask=None, cache=None):  # noqa: ARG002
        return self.head(self.embed(x))


class _AlwaysEOSModel:
    def __init__(self, vocab_size: int = 32, eos_id: int = 2):
        self.vocab_size = vocab_size
        self.eos_id = eos_id

    def __call__(self, x, mask=None, cache=None):  # noqa: ARG002
        bsz, seq_len = x.shape
        vocab_idx = mx.arange(self.vocab_size, dtype=mx.int32).reshape(1, 1, -1)
        eos_logits = mx.where(
            vocab_idx == self.eos_id,
            mx.array(20.0, dtype=mx.float32),
            mx.array(-20.0, dtype=mx.float32),
        )
        return mx.broadcast_to(eos_logits, (bsz, seq_len, self.vocab_size))


class _NeverEOSModel:
    def __init__(self, vocab_size: int = 32, eos_id: int = 2, preferred_id: int = 3):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.preferred_id = preferred_id

    def __call__(self, x, mask=None, cache=None):  # noqa: ARG002
        bsz, seq_len = x.shape
        vocab_idx = mx.arange(self.vocab_size, dtype=mx.int32).reshape(1, 1, -1)
        logits = mx.where(
            vocab_idx == self.preferred_id,
            mx.array(20.0, dtype=mx.float32),
            mx.array(-20.0, dtype=mx.float32),
        )
        logits = mx.where(
            vocab_idx == self.eos_id,
            mx.array(-40.0, dtype=mx.float32),
            logits,
        )
        return mx.broadcast_to(logits, (bsz, seq_len, self.vocab_size))


@pytest.mark.unit
class TestPromptCacheConstruction:
    def test_uses_legacy_cache_module_layout(self, monkeypatch):
        model = object()
        monkeypatch.setattr(mlx_generation, "HAS_MLX_LM", True)

        def fake_import_module(name: str):
            if name == "mlx_lm.cache":
                return types.SimpleNamespace(
                    make_prompt_cache=lambda incoming_model: {
                        "layout": "cache",
                        "model": incoming_model,
                    }
                )
            raise ImportError(name)

        monkeypatch.setattr(mlx_generation.importlib, "import_module", fake_import_module)

        cache_obj = mlx_generation._make_prompt_cache_if_available(model)
        assert cache_obj["layout"] == "cache"
        assert cache_obj["model"] is model

    def test_falls_back_to_cache_prompt_layout(self, monkeypatch):
        model = object()
        monkeypatch.setattr(mlx_generation, "HAS_MLX_LM", True)
        import_calls: List[str] = []

        def fake_import_module(name: str):
            import_calls.append(name)
            if name == "mlx_lm.cache":
                raise ImportError(name)
            if name == "mlx_lm.cache_prompt":
                return types.SimpleNamespace(
                    make_prompt_cache=lambda incoming_model: {
                        "layout": "cache_prompt",
                        "model": incoming_model,
                    }
                )
            raise ImportError(name)

        monkeypatch.setattr(mlx_generation.importlib, "import_module", fake_import_module)

        cache_obj = mlx_generation._make_prompt_cache_if_available(model)
        assert cache_obj["layout"] == "cache_prompt"
        assert cache_obj["model"] is model
        assert import_calls[:2] == ["mlx_lm.cache", "mlx_lm.cache_prompt"]


@pytest.mark.unit
class TestBatchedMaskCreation:
    def test_single_sequence_no_padding_is_causal(self):
        mask = _create_batched_prefill_mask([4], max_prompt_len=4)
        expected = mx.array(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        ).reshape(1, 1, 4, 4)
        assert mask.shape == (1, 1, 4, 4)
        assert mx.array_equal(mask, expected)

    def test_prefill_mask_blocks_left_padding(self):
        # Prompt length 2 in window 4 -> positions 0,1 are padding.
        mask = _create_batched_prefill_mask([2], max_prompt_len=4)
        assert not bool(mx.any(mask[0, 0, :, :2]).item())
        assert bool(mask[0, 0, 3, 2].item())
        assert bool(mask[0, 0, 3, 3].item())

    def test_decode_mask_blocks_padding_kv_positions(self):
        mask = _create_batched_decode_mask([2], max_prompt_len=4, decode_offset=1)
        assert mask.shape == (1, 1, 1, 5)
        assert not bool(mask[0, 0, 0, 0].item())
        assert not bool(mask[0, 0, 0, 1].item())
        assert bool(mask[0, 0, 0, 2].item())
        assert bool(mask[0, 0, 0, 4].item())

    def test_each_sequence_has_its_own_padding_boundary(self):
        mask = _create_batched_prefill_mask([4, 2], max_prompt_len=4)
        assert mask.shape == (2, 1, 4, 4)
        assert bool(mask[0, 0, 3, 0].item())  # no pad on seq 0
        assert not bool(mask[1, 0, 3, 0].item())  # pad on seq 1
        assert not bool(mask[1, 0, 3, 1].item())
        assert bool(mask[1, 0, 3, 2].item())


@pytest.mark.unit
class TestBatchedGeneration:
    def test_n1_structure(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model,
            tok,
            [mx.array([3, 4, 5], dtype=mx.int32)],
            max_tokens=4,
            temperature=0.0,
        )
        assert len(out) == 1
        resp, info = out[0]
        assert resp.ndim == 1
        assert "logprob" in info
        assert info["logprob"].ndim == 1
        assert info["logprob"].shape[0] == resp.shape[0]

    def test_equal_length_prompts(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompts = [
            mx.array([3, 4, 5], dtype=mx.int32),
            mx.array([6, 7, 8], dtype=mx.int32),
        ]
        out = batch_generate_tokens(model, tok, prompts, max_tokens=3, temperature=0.0)
        assert len(out) == 2
        for resp, info in out:
            assert info["logprob"].shape[0] == resp.shape[0]

    def test_variable_length_prompts(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompts = [
            mx.array([3], dtype=mx.int32),
            mx.array([4, 5, 6, 7], dtype=mx.int32),
            mx.array([8, 9], dtype=mx.int32),
        ]
        out = batch_generate_tokens(model, tok, prompts, max_tokens=3, temperature=0.0)
        assert len(out) == 3
        for resp, info in out:
            assert info["logprob"].shape[0] == resp.shape[0]

    def test_logprobs_are_non_positive_and_finite(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model,
            tok,
            [mx.array([3, 4], dtype=mx.int32), mx.array([5, 6], dtype=mx.int32)],
            max_tokens=4,
            temperature=0.5,
        )
        for _, info in out:
            lp = info["logprob"]
            if lp.shape[0] == 0:
                continue
            assert bool(mx.all(lp <= 0).item())
            assert not bool(mx.any(mx.isnan(lp)).item())
            assert not bool(mx.any(mx.isinf(lp)).item())

    def test_no_tokens_after_eos(self):
        model = _AlwaysEOSModel(eos_id=2)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model,
            tok,
            [mx.array([3, 4], dtype=mx.int32), mx.array([5], dtype=mx.int32)],
            max_tokens=5,
            temperature=0.0,
        )
        for resp, info in out:
            assert resp.shape[0] == 1
            assert int(resp[0].item()) == tok.eos_token_id
            assert info["logprob"].shape[0] == 1

    def test_max_tokens_when_eos_never_emitted(self):
        model = _NeverEOSModel(eos_id=2, preferred_id=3)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model,
            tok,
            [mx.array([3], dtype=mx.int32), mx.array([4, 5], dtype=mx.int32)],
            max_tokens=4,
            temperature=0.0,
        )
        for resp, info in out:
            assert resp.shape[0] == 4
            assert info["logprob"].shape[0] == 4
            assert tok.eos_token_id not in resp.tolist()


class _SingleTurnEnv:
    def __init__(self):
        self._prompts = [
            [3, 4, 5],
            [6, 7],
            [8, 9, 10],
        ]
        self._idx = 0

    def reset(self):
        prompt = self._prompts[self._idx % len(self._prompts)]
        self._idx += 1
        return mx.array(prompt, dtype=mx.int32), {"episode": self._idx - 1}

    def step(self, action):
        length = len(action) if isinstance(action, list) else 1
        return {
            "observation": mx.array([], dtype=mx.int32),
            "reward": float(length),
            "terminated": True,
            "truncated": False,
            "info": {},
        }


def _dummy_policy(obs, deterministic=False):  # noqa: ARG001
    return mx.array([1], dtype=mx.int32), {"logprob": mx.array([-0.5], dtype=mx.float32)}


@pytest.mark.unit
class TestBatchedRollout:
    def test_buffer_episode_count(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env,
            policy=_dummy_policy,
            strategy=create_strategy("grpo"),
            max_steps=5,
        )
        batched_policy = create_batched_policy(
            model,
            tok,
            {"max_tokens": 3, "temperature": 0.0},
        )
        buffer = runner.collect_batched(batched_policy, batch_size=2)
        assert len(buffer.episodes) == 5

    def test_logprob_length_matches_response_length(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env,
            policy=_dummy_policy,
            strategy=create_strategy("grpo"),
            max_steps=4,
        )
        batched_policy = create_batched_policy(
            model,
            tok,
            {"max_tokens": 4, "temperature": 0.0},
        )
        buffer = runner.collect_batched(batched_policy, batch_size=2)
        for ep in buffer.episodes:
            assert ep.logprob is not None
            act = ep.act[0]
            logp = ep.logprob[0]
            act_len = len(act.tolist()) if hasattr(act, "tolist") else len(act)
            logp_len = len(logp.tolist()) if hasattr(logp, "tolist") else len(logp)
            assert logp_len == act_len

    def test_batch_size_one_path(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env,
            policy=_dummy_policy,
            strategy=create_strategy("grpo"),
            max_steps=3,
        )
        batched_policy = create_batched_policy(
            model,
            tok,
            {"max_tokens": 2, "temperature": 0.0},
        )
        buffer = runner.collect_batched(batched_policy, batch_size=1)
        assert len(buffer.episodes) == 3

    def test_collect_batched_exposes_generation_profile(self):
        model = _TinyLM()
        tok = _DummyTokenizer()
        env = _SingleTurnEnv()
        runner = RolloutRunner(
            env,
            policy=_dummy_policy,
            strategy=create_strategy("grpo"),
            max_steps=3,
            profile=True,
        )
        batched_policy = create_batched_policy(
            model,
            tok,
            {
                "max_tokens": 3,
                "temperature": 0.0,
                "profile_decode_stats": True,
            },
        )
        _ = runner.collect_batched(batched_policy, batch_size=2)
        gen_profile = runner.get_generation_profile()
        assert "decode_tps" in gen_profile
        assert "shared_prefill_used" in gen_profile


@pytest.mark.unit
class TestGreedyDeterminismAndParity:
    """H7: Greedy decode is deterministic and invariant to batch composition.

    These tests verify the core Amdahl optimisation correctness guarantee:
    left-padding and attention masking must not alter model output for any
    individual prompt.
    """

    def test_greedy_deterministic(self):
        """Running greedy decode twice on the same prompt yields identical tokens."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt = [mx.array([3, 4, 5], dtype=mx.int32)]

        r1 = batch_generate_tokens(model, tok, prompt, max_tokens=5, temperature=0.0)
        r2 = batch_generate_tokens(model, tok, prompt, max_tokens=5, temperature=0.0)

        assert mx.array_equal(r1[0][0], r2[0][0]), (
            "Greedy decode should be deterministic across calls"
        )

    def test_single_vs_batch_parity(self):
        """A prompt produces the same greedy tokens whether alone or in a batch.

        This is THE key invariant for batched generation: left-padding and
        attention masking must not change the output for any individual prompt.
        """
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt_a = mx.array([3, 4, 5], dtype=mx.int32)
        prompt_b = mx.array([6, 7, 8, 9], dtype=mx.int32)

        # Prompt A alone (batch of 1)
        solo = batch_generate_tokens(
            model, tok, [prompt_a], max_tokens=5, temperature=0.0
        )
        solo_tokens = solo[0][0]

        # Prompt A in a batch with B (batch of 2 — B has different length)
        batched = batch_generate_tokens(
            model, tok, [prompt_a, prompt_b], max_tokens=5, temperature=0.0
        )
        batched_tokens = batched[0][0]

        assert mx.array_equal(solo_tokens, batched_tokens), (
            f"Prompt A should produce identical greedy tokens whether batched alone "
            f"or with other prompts.\n"
            f"  Solo:    {solo_tokens.tolist()}\n"
            f"  Batched: {batched_tokens.tolist()}"
        )

    def test_single_vs_triple_batch_parity(self):
        """Same prompt in batch of 1 vs batch of 3 (with variable lengths)."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        target = mx.array([10, 11, 12], dtype=mx.int32)

        solo = batch_generate_tokens(
            model, tok, [target], max_tokens=4, temperature=0.0
        )
        trio = batch_generate_tokens(
            model, tok, [
                target,
                mx.array([1], dtype=mx.int32),
                mx.array([5, 6, 7, 8, 9], dtype=mx.int32),
            ],
            max_tokens=4, temperature=0.0,
        )

        assert mx.array_equal(solo[0][0], trio[0][0]), (
            "Target prompt should produce identical tokens regardless of batch size"
        )

    def test_logprobs_match_across_batch_sizes(self):
        """Logprobs for a prompt should be identical solo vs batched."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt = mx.array([3, 4, 5], dtype=mx.int32)

        solo = batch_generate_tokens(
            model, tok, [prompt], max_tokens=4, temperature=0.0
        )
        duo = batch_generate_tokens(
            model, tok, [prompt, mx.array([7, 8], dtype=mx.int32)],
            max_tokens=4, temperature=0.0,
        )

        solo_lp = solo[0][1]["logprob"]
        duo_lp = duo[0][1]["logprob"]
        assert mx.allclose(solo_lp, duo_lp, atol=1e-5).item(), (
            f"Logprobs should match.\n"
            f"  Solo:    {solo_lp.tolist()}\n"
            f"  Batched: {duo_lp.tolist()}"
        )


@pytest.mark.unit
class TestMaskValidationEdgeCases:
    """H8: Mask creation rejects invalid inputs with clear errors."""

    def test_prefill_mask_zero_max_len(self):
        with pytest.raises(ValueError, match="positive"):
            _create_batched_prefill_mask([1], max_prompt_len=0)

    def test_prefill_mask_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _create_batched_prefill_mask([], max_prompt_len=5)

    def test_prefill_mask_zero_prompt_length(self):
        with pytest.raises(ValueError, match=">= 1"):
            _create_batched_prefill_mask([0], max_prompt_len=5)

    def test_decode_mask_negative_offset(self):
        with pytest.raises(ValueError, match=">= 0"):
            _create_batched_decode_mask([1], max_prompt_len=5, decode_offset=-1)

    def test_decode_mask_zero_max_len(self):
        with pytest.raises(ValueError, match="positive"):
            _create_batched_decode_mask([1], max_prompt_len=0, decode_offset=0)

    def test_decode_mask_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            _create_batched_decode_mask([], max_prompt_len=5, decode_offset=0)

    def test_decode_mask_zero_prompt_length(self):
        with pytest.raises(ValueError, match=">= 1"):
            _create_batched_decode_mask([0], max_prompt_len=5, decode_offset=0)


@pytest.mark.unit
def test_rollout_coordinator_batch_size_routes_to_batched_path():
    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_policy(
        model,
        tok,
        generation_params={"max_tokens": 2, "temperature": 0.0},
    )
    coordinator = RolloutCoordinator(
        env_fn=_SingleTurnEnv,
        policy_fn=lambda: policy,
        algorithm="grpo",
        num_workers=0,
        max_steps=4,
        max_episodes=4,
        batch_size=2,
    )
    try:
        buffer = coordinator.collect()
        assert len(buffer.episodes) == 4
    finally:
        coordinator.close()


# ── Repetition penalty tests ────────────────────────────────────────


@pytest.mark.unit
def test_batch_generate_rejects_invalid_repetition_penalty():
    """Negative / zero repetition_penalty must raise ValueError."""
    model = _TinyLM()
    tok = _DummyTokenizer()
    prompts = [mx.array([3, 4], dtype=mx.int32)]

    for bad_value in [-1.0, 0.0, -0.5]:
        with pytest.raises(ValueError, match="repetition_penalty must be a positive"):
            batch_generate_tokens(
                model, tok, prompts, max_tokens=2, repetition_penalty=bad_value,
            )


@pytest.mark.unit
def test_create_batched_policy_forwards_repetition_penalty():
    """Regression guard: create_batched_policy must pass repetition_penalty
    through to batch_generate_tokens (the exact bug fixed in this PR)."""
    import unittest.mock as mock

    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_batched_policy(
        model, tok,
        generation_params={
            "max_tokens": 2,
            "temperature": 0.0,
            "repetition_penalty": 1.3,
        },
    )

    prompts = [mx.array([3, 4], dtype=mx.int32)]
    with mock.patch(
        "textpolicy.generation.mlx_generation.batch_generate_tokens",
        wraps=batch_generate_tokens,
    ) as spy:
        policy(prompts)
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs.get("repetition_penalty") == 1.3, (
            "repetition_penalty was not forwarded to batch_generate_tokens"
        )


@pytest.mark.unit
def test_batch_generate_tokens_unknown_backend_raises():
    model = _TinyLM()
    tok = _DummyTokenizer()
    prompts = [mx.array([3, 4], dtype=mx.int32)]
    with pytest.raises(ValueError, match="Unknown batched decode backend"):
        batch_generate_tokens(
            model,
            tok,
            prompts,
            max_tokens=2,
            temperature=0.0,
            backend="does-not-exist",
        )


@pytest.mark.unit
def test_custom_batched_decode_backend_dispatch():
    calls = {"count": 0}
    backend_name = "unit-test-backend"

    def _backend(
        model,  # noqa: ARG001
        tokenizer,  # noqa: ARG001
        prompt_token_lists,
        max_tokens,  # noqa: ARG001
        temperature,  # noqa: ARG001
        top_p,  # noqa: ARG001
        repetition_penalty,  # noqa: ARG001
        repetition_context_size,  # noqa: ARG001
    ):
        calls["count"] += 1
        return [
            (
                mx.array([9], dtype=mx.int32),
                {"logprob": mx.array([-0.1], dtype=mx.float32)},
            )
            for _ in prompt_token_lists
        ]

    mlx_generation.register_batched_decode_backend(backend_name, _backend)
    model = _TinyLM()
    tok = _DummyTokenizer()
    prompts = [
        mx.array([3, 4], dtype=mx.int32),
        mx.array([5, 6], dtype=mx.int32),
    ]
    out = batch_generate_tokens(
        model,
        tok,
        prompts,
        max_tokens=3,
        temperature=0.0,
        backend=backend_name,
    )

    assert calls["count"] == 1
    assert len(out) == 2
    for resp, info in out:
        assert resp.tolist() == [9]
        assert info["logprob"].tolist() == pytest.approx([-0.1], abs=1e-6)


@pytest.mark.unit
def test_create_batched_policy_forwards_decode_backend():
    calls = {"count": 0}
    backend_name = "unit-test-policy-backend"

    def _backend(
        model,  # noqa: ARG001
        tokenizer,  # noqa: ARG001
        prompt_token_lists,
        max_tokens,  # noqa: ARG001
        temperature,  # noqa: ARG001
        top_p,  # noqa: ARG001
        repetition_penalty,  # noqa: ARG001
        repetition_context_size,  # noqa: ARG001
    ):
        calls["count"] += 1
        return [
            (
                mx.array([7], dtype=mx.int32),
                {"logprob": mx.array([-0.2], dtype=mx.float32)},
            )
            for _ in prompt_token_lists
        ]

    mlx_generation.register_batched_decode_backend(backend_name, _backend)

    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_batched_policy(
        model,
        tok,
        generation_params={
            "max_tokens": 2,
            "temperature": 0.0,
            "decode_backend": backend_name,
        },
    )

    results = policy([mx.array([3, 4], dtype=mx.int32)])
    assert calls["count"] == 1
    assert results[0][0].tolist() == [7]


@pytest.mark.unit
def test_mlx_native_backend_dispatch_uses_batch_generator(monkeypatch):
    calls: Dict[str, Any] = {}

    class _FakeBatchGenerator:
        def __init__(
            self,
            model,  # noqa: ARG002
            max_tokens=128,  # noqa: ARG002
            stop_tokens=None,
            sampler=None,  # noqa: ARG002
            logits_processors=None,  # noqa: ARG002
            completion_batch_size=32,  # noqa: ARG002
            prefill_batch_size=8,  # noqa: ARG002
            prefill_step_size=2048,  # noqa: ARG002
            prompt_progress_callback=None,  # noqa: ARG002
            max_kv_size=None,  # noqa: ARG002
        ):
            calls["stop_tokens"] = set(stop_tokens or set())
            self._uids: List[int] = []
            self._done = False

        def insert(
            self,
            prompts,
            max_tokens=None,  # noqa: ARG002
            caches=None,  # noqa: ARG002
            samplers=None,  # noqa: ARG002
            logits_processors=None,  # noqa: ARG002
        ):
            calls["insert_prompts"] = prompts
            self._uids = list(range(len(prompts)))
            return self._uids

        def next(self):
            if self._done:
                return []
            self._done = True
            responses = []
            for uid in self._uids:
                lp = [-10.0] * 32
                lp[2] = -0.1
                responses.append(
                    types.SimpleNamespace(
                        uid=uid,
                        token=2,
                        logprobs=mx.array(lp, dtype=mx.float32),
                        finish_reason="stop",
                    )
                )
            return responses

        def stats(self):
            n = len(self._uids)
            return types.SimpleNamespace(
                prompt_time=0.01,
                generation_time=0.02,
                generation_tokens=n,
                generation_tps=float(n) / 0.02 if n > 0 else 0.0,
            )

        def close(self):
            return None

    monkeypatch.setattr(
        mlx_generation,
        "_import_mlx_generate_module",
        lambda: types.SimpleNamespace(BatchGenerator=_FakeBatchGenerator),
    )

    model = _TinyLM()
    tok = _DummyTokenizer()
    profile: Dict[str, float] = {}
    out = batch_generate_tokens(
        model,
        tok,
        [mx.array([3, 4], dtype=mx.int32), mx.array([5, 6], dtype=mx.int32)],
        max_tokens=2,
        temperature=0.0,
        backend="mlx_native",
        profile_collector=profile,
    )

    assert calls["stop_tokens"] == {tok.eos_token_id}
    assert len(calls["insert_prompts"]) == 2
    assert len(out) == 2
    for resp, info in out:
        assert resp.tolist() == [2]
        assert info["logprob"].tolist() == pytest.approx([-0.1], abs=1e-6)
    assert profile.get("backend_mlx_native") == 1.0


@pytest.mark.unit
def test_mlx_speculative_backend_uses_target_logprobs(monkeypatch):
    calls: Dict[str, Any] = {}

    def _fake_speculative_generate_step(
        prompt,  # noqa: ARG001
        model,  # noqa: ARG001
        draft_model,  # noqa: ARG001
        **kwargs,
    ):
        calls["kwargs"] = kwargs
        lp_a = [-10.0] * 32
        lp_b = [-10.0] * 32
        lp_a[4] = -0.3
        lp_b[2] = -0.2
        yield 4, mx.array(lp_a, dtype=mx.float32), True
        yield 2, mx.array(lp_b, dtype=mx.float32), False

    monkeypatch.setattr(
        mlx_generation,
        "_import_mlx_generate_module",
        lambda: types.SimpleNamespace(speculative_generate_step=_fake_speculative_generate_step),
    )

    model = _TinyLM()
    tok = _DummyTokenizer()
    profile: Dict[str, float] = {}
    out = batch_generate_tokens(
        model,
        tok,
        [mx.array([3, 4], dtype=mx.int32)],
        max_tokens=8,
        temperature=0.0,
        backend="mlx_speculative",
        backend_options={"draft_model": object(), "num_draft_tokens": 5},
        profile_collector=profile,
    )

    assert out[0][0].tolist() == [4, 2]
    assert out[0][1]["logprob"].tolist() == pytest.approx([-0.3, -0.2], abs=1e-6)
    assert calls["kwargs"]["num_draft_tokens"] == 5
    assert profile.get("backend_mlx_speculative") == 1.0
    assert profile.get("speculative_tokens_from_draft") == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_mlx_speculative_backend_without_draft_falls_back(monkeypatch):
    calls = {"count": 0}

    def _fallback(*args, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        return [
            (
                mx.array([9], dtype=mx.int32),
                {"logprob": mx.array([-0.4], dtype=mx.float32)},
            )
        ]

    monkeypatch.setattr(mlx_generation, "_batch_generate_tokens_mlx", _fallback)
    monkeypatch.setattr(
        mlx_generation,
        "_import_mlx_generate_module",
        lambda: types.SimpleNamespace(speculative_generate_step=lambda *a, **k: iter(())),
    )

    model = _TinyLM()
    tok = _DummyTokenizer()
    out = batch_generate_tokens(
        model,
        tok,
        [mx.array([3, 4], dtype=mx.int32)],
        max_tokens=2,
        temperature=0.0,
        backend="mlx_speculative",
    )

    assert calls["count"] == 1
    assert out[0][0].tolist() == [9]


@pytest.mark.unit
def test_create_batched_policy_loads_speculative_draft_once(monkeypatch):
    calls = {"load": 0, "batch": 0, "draft_obj_ids": []}
    draft_model_obj = object()

    def _fake_load(path_or_hf_repo, **kwargs):  # noqa: ARG001
        calls["load"] += 1
        return draft_model_obj, None

    def _fake_batch_generate_tokens(
        model,  # noqa: ARG001
        tokenizer,  # noqa: ARG001
        prompt_token_lists,
        max_tokens=50,  # noqa: ARG001
        temperature=0.7,  # noqa: ARG001
        top_p=0.9,  # noqa: ARG001
        repetition_penalty=None,  # noqa: ARG001
        repetition_context_size=20,  # noqa: ARG001
        backend="mlx",  # noqa: ARG001
        profile_collector=None,  # noqa: ARG001
        backend_options=None,
    ):
        calls["batch"] += 1
        calls["draft_obj_ids"].append(id(backend_options.get("draft_model")))
        return [
            (
                mx.array([2], dtype=mx.int32),
                {"logprob": mx.array([-0.1], dtype=mx.float32)},
            )
            for _ in prompt_token_lists
        ]

    monkeypatch.setattr(mlx_generation, "HAS_MLX_LM", True)
    monkeypatch.setattr(mlx_generation, "load", _fake_load)
    monkeypatch.setattr(
        mlx_generation,
        "batch_generate_tokens",
        _fake_batch_generate_tokens,
    )

    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_batched_policy(
        model,
        tok,
        generation_params={
            "max_tokens": 2,
            "temperature": 0.0,
            "decode_backend": "mlx_speculative",
            "draft_model_id": "dummy/draft-model",
        },
    )
    _ = policy([mx.array([3, 4], dtype=mx.int32)])
    _ = policy([mx.array([5, 6], dtype=mx.int32)])

    assert calls["load"] == 1
    assert calls["batch"] == 2
    assert calls["draft_obj_ids"][0] == calls["draft_obj_ids"][1]


@pytest.mark.unit
def test_batch_generate_tokens_profile_collector_populated():
    model = _TinyLM()
    tok = _DummyTokenizer()
    profile = {}
    _ = batch_generate_tokens(
        model,
        tok,
        [mx.array([3, 4], dtype=mx.int32)],
        max_tokens=3,
        temperature=0.0,
        profile_collector=profile,
    )
    expected_keys = {
        "prefill_s",
        "decode_s",
        "total_s",
        "decode_tps",
        "tokens_generated",
        "shared_prefill_used",
        "used_cache",
        "avg_active_batch",
    }
    assert expected_keys.issubset(profile.keys()), (
        f"Missing decode profile keys: {expected_keys - set(profile.keys())}"
    )


@pytest.mark.unit
def test_create_batched_policy_sets_last_decode_profile_when_enabled():
    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_batched_policy(
        model,
        tok,
        generation_params={
            "max_tokens": 2,
            "temperature": 0.0,
            "profile_decode_stats": True,
        },
    )
    _ = policy([mx.array([3, 4], dtype=mx.int32)])
    profile = getattr(policy, "_tp_last_decode_profile", None)
    assert isinstance(profile, dict)
    assert "decode_tps" in profile


class _NarrowGapModel:
    """Model with a small logit gap: token 3 gets +1.0, token 4 gets +0.8,
    everything else -10.  A repetition_penalty of 2.0 on token 3 flips
    its logit from +1.0 to +0.5, making token 4 (+0.8) the argmax."""

    def __init__(self, vocab_size: int = 32):
        self.vocab_size = vocab_size

    def __call__(self, x, mask=None, cache=None):  # noqa: ARG002
        bsz, seq_len = x.shape
        base = mx.full((1, 1, self.vocab_size), -10.0)
        base = base.at[:, :, 3].add(11.0)   # token 3 → +1.0
        base = base.at[:, :, 4].add(10.8)   # token 4 → +0.8
        return mx.broadcast_to(base, (bsz, seq_len, self.vocab_size))


@pytest.mark.unit
def test_repetition_penalty_affects_sampling():
    """Repetition penalty should discourage re-sampling the same token.

    Uses a deterministic model with a narrow logit gap (token 3: +1.0,
    token 4: +0.8).  Penalty=2.0 divides token 3's logit to +0.5, making
    token 4 the new argmax.
    """
    model = _NarrowGapModel()
    tok = _DummyTokenizer()
    prompt = mx.array([3, 3, 3, 3], dtype=mx.int32)  # token 3 in context

    results_no_penalty = batch_generate_tokens(
        model, tok, [prompt], max_tokens=4, temperature=0.0,
    )
    results_with_penalty = batch_generate_tokens(
        model, tok, [prompt], max_tokens=4, temperature=0.0,
        repetition_penalty=2.0,
    )

    tokens_no = results_no_penalty[0][0].tolist()
    tokens_with = results_with_penalty[0][0].tolist()
    # Without penalty: all token 3 (highest logit).
    # With penalty: token 3 is penalized, so token 4 wins.
    assert all(t == 3 for t in tokens_no), f"Expected all 3s, got {tokens_no}"
    assert tokens_with[0] == 4, (
        f"Expected token 4 (penalty flips argmax from 3), got {tokens_with[0]}"
    )


@pytest.mark.unit
def test_create_policy_forwards_repetition_penalty():
    """Unbatched create_policy should forward repetition_penalty to generate_tokens."""
    import unittest.mock as mock

    model = _TinyLM()
    tok = _DummyTokenizer()
    policy = create_policy(model, tok, generation_params={"repetition_penalty": 1.5})

    # Mock generate_tokens without wraps — the unbatched path calls
    # mlx_lm.stream_generate internally which needs a real tokenizer.
    # We only need to verify the call signature, not execute generation.
    dummy_result = (mx.array([1], dtype=mx.int32), {"logprob": mx.array([0.0])})
    prompt = mx.array([1, 2], dtype=mx.int32)
    with mock.patch(
        "textpolicy.generation.mlx_generation.generate_tokens",
        return_value=dummy_result,
    ) as spy:
        policy(prompt)
        spy.assert_called_once()
        _, kwargs = spy.call_args
        assert kwargs.get("repetition_penalty") == 1.5, (
            "repetition_penalty was not forwarded to generate_tokens"
        )


# ── Opt 1: Same-prompt grouping tests ───────────────────────────────


@pytest.mark.unit
class TestGroupSizePromptCycling:
    """Verify TextGenerationEnv.group_size controls prompt cycling."""

    def test_group_size_1_cycles_every_episode(self):
        """Default group_size=1 gives round-robin prompt cycling."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        tok = _DummyTokenizer()
        prompts = ["A", "B", "C"]

        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=lambda prompt, completion, example, **kw: 0.0,
            tokenizer=tok,
            group_size=1,
        )
        seen = []
        for _ in range(6):
            _, info = env.reset()
            seen.append(info["prompt_text"])
            env.step("x")  # advance episode counter
        assert seen == ["A", "B", "C", "A", "B", "C"]

    def test_group_size_equals_batch_repeats_prompt(self):
        """group_size=3 repeats each prompt 3 times before advancing."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        tok = _DummyTokenizer()
        prompts = ["A", "B"]

        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=lambda prompt, completion, example, **kw: 0.0,
            tokenizer=tok,
            group_size=3,
        )
        seen = []
        for _ in range(6):
            _, info = env.reset()
            seen.append(info["prompt_text"])
            env.step("x")
        assert seen == ["A", "A", "A", "B", "B", "B"]

    def test_group_size_invalid_raises(self):
        from textpolicy.environment.text_generation import TextGenerationEnv

        tok = _DummyTokenizer()
        with pytest.raises(ValueError, match="group_size must be >= 1"):
            TextGenerationEnv(
                prompts=["X"],
                reward_fn=lambda prompt, completion, example, **kw: 0.0,
                tokenizer=tok,
                group_size=0,
            )

    def test_clone_preserves_group_size(self):
        from textpolicy.environment.text_generation import TextGenerationEnv

        tok = _DummyTokenizer()
        env = TextGenerationEnv(
            prompts=["A", "B"],
            reward_fn=lambda prompt, completion, example, **kw: 0.0,
            tokenizer=tok,
            group_size=4,
        )
        cloned = env.clone()
        assert cloned.group_size == 4


# ── Opt 2: Shared KV-cache prefill tests ────────────────────────────


@pytest.mark.unit
class TestSharedPrefill:
    """Verify that identical-prompt batches produce correct results."""

    def test_identical_prompts_same_greedy_output(self):
        """When all prompts are the same, all greedy outputs should be identical."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt = mx.array([3, 4, 5], dtype=mx.int32)
        out = batch_generate_tokens(
            model, tok,
            [prompt, prompt, prompt],
            max_tokens=5,
            temperature=0.0,
        )
        assert len(out) == 3
        for resp, _ in out[1:]:
            assert mx.array_equal(out[0][0], resp), (
                "Identical prompts should produce identical greedy tokens"
            )

    def test_identical_prompts_logprobs_match(self):
        """Logprobs should be identical across identical-prompt batch entries."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt = mx.array([3, 4, 5], dtype=mx.int32)
        out = batch_generate_tokens(
            model, tok,
            [prompt, prompt],
            max_tokens=4,
            temperature=0.0,
        )
        lp0 = out[0][1]["logprob"]
        lp1 = out[1][1]["logprob"]
        assert mx.allclose(lp0, lp1, atol=1e-5).item(), (
            f"Logprobs should match for identical prompts.\n"
            f"  Seq 0: {lp0.tolist()}\n  Seq 1: {lp1.tolist()}"
        )

    def test_identical_vs_solo_parity(self):
        """Identical-prompt batch should produce same result as solo generation."""
        model = _TinyLM()
        tok = _DummyTokenizer()
        prompt = mx.array([3, 4, 5], dtype=mx.int32)
        solo = batch_generate_tokens(
            model, tok, [prompt], max_tokens=5, temperature=0.0,
        )
        duo = batch_generate_tokens(
            model, tok, [prompt, prompt], max_tokens=5, temperature=0.0,
        )
        assert mx.array_equal(solo[0][0], duo[0][0]), (
            "Solo and identical-batch should produce the same tokens"
        )


# ── Opt 4: Batch compaction tests ────────────────────────────────────


class _StaggeredEOSModel:
    """Emits EOS when the first input token is even; preferred_id when odd.

    This creates a mixed-EOS scenario: sequences with even-valued first
    prompt tokens finish immediately while odd-valued ones run to
    max_tokens, exercising batch compaction (Opt 4).

    Robust to compaction: behaviour depends on input content, not batch
    position, so it works correctly after finished sequences are removed.
    """

    def __init__(self, vocab_size: int = 32, eos_id: int = 2, preferred_id: int = 3):
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.preferred_id = preferred_id

    def __call__(self, x, mask=None, cache=None):  # noqa: ARG002
        bsz, seq_len = x.shape
        first_tok = x[:, 0]  # [bsz] — first token determines behaviour
        is_even = ((first_tok % 2) == 0).reshape(bsz, 1, 1)
        vocab_idx = mx.arange(self.vocab_size).reshape(1, 1, self.vocab_size)
        logits = mx.where(
            is_even & (vocab_idx == self.eos_id),
            mx.array(40.0),
            mx.where(
                vocab_idx == self.preferred_id,
                mx.array(20.0),
                mx.array(-20.0),
            ),
        )
        return mx.broadcast_to(logits, (bsz, seq_len, self.vocab_size))


@pytest.mark.unit
class TestBatchCompaction:
    """Verify that batch compaction correctly handles staggered EOS."""

    def test_staggered_eos_correct_lengths(self):
        """Even-first-token seq stops at 1 token; odd-first-token runs to max."""
        model = _StaggeredEOSModel(eos_id=2, preferred_id=3)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model, tok,
            # First token 6 (even) → EOS; first token 7 (odd) → preferred
            [mx.array([6, 6], dtype=mx.int32), mx.array([7, 8], dtype=mx.int32)],
            max_tokens=6,
            temperature=0.0,
        )
        assert len(out) == 2
        # Seq 0 (even): should have 1 token (EOS)
        assert out[0][0].shape[0] == 1
        assert int(out[0][0][0].item()) == 2
        # Seq 1 (odd): should have max_tokens tokens (never EOS)
        assert out[1][0].shape[0] == 6
        assert all(t == 3 for t in out[1][0].tolist())

    def test_staggered_logprobs_correct_lengths(self):
        """Logprob arrays must match token lengths after compaction."""
        model = _StaggeredEOSModel(eos_id=2, preferred_id=3)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model, tok,
            # Even first token → EOS (1 token); odd first token → 8 tokens
            [mx.array([6], dtype=mx.int32), mx.array([7], dtype=mx.int32)],
            max_tokens=8,
            temperature=0.0,
        )
        for resp, info in out:
            assert info["logprob"].shape[0] == resp.shape[0]

    def test_staggered_three_sequences(self):
        """Batch of 3: even-first seqs finish early, odd-first run to max.

        Uses equal-length prompts to avoid left-padding changing the first
        token (pad_id=0 is even and would trigger EOS for shorter prompts).
        """
        model = _StaggeredEOSModel(eos_id=2, preferred_id=3)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model, tok,
            [
                mx.array([4, 10], dtype=mx.int32),   # even → EOS
                mx.array([7, 8], dtype=mx.int32),    # odd → preferred
                mx.array([9, 11], dtype=mx.int32),   # odd → preferred
            ],
            max_tokens=5,
            temperature=0.0,
        )
        assert len(out) == 3
        assert out[0][0].shape[0] == 1   # EOS on step 1
        assert out[1][0].shape[0] == 5   # full generation
        assert out[2][0].shape[0] == 5   # full generation


# ── Opt 3 & 4: Vectorised EOS and early exit tests ──────────────────


@pytest.mark.unit
class TestVectorisedEOSAndEarlyExit:
    """Verify vectorised EOS detection and early-exit behaviour."""

    def test_early_eos_stops_all_sequences(self):
        """AlwaysEOSModel should cause immediate stop for all sequences."""
        model = _AlwaysEOSModel(eos_id=2)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model, tok,
            [mx.array([3, 4], dtype=mx.int32), mx.array([5, 6], dtype=mx.int32)],
            max_tokens=10,
            temperature=0.0,
        )
        for resp, info in out:
            assert resp.shape[0] == 1, "Should stop after 1 token (EOS)"
            assert int(resp[0].item()) == 2

    def test_mixed_eos_timing(self):
        """Sequences that hit EOS should stop; others continue to max_tokens."""
        # AlwaysEOSModel emits EOS immediately for all sequences — test
        # that result packaging is correct with identical-length outputs.
        model = _AlwaysEOSModel(eos_id=2)
        tok = _DummyTokenizer()
        out = batch_generate_tokens(
            model, tok,
            [mx.array([3], dtype=mx.int32)] * 4,
            max_tokens=8,
            temperature=0.0,
        )
        for resp, info in out:
            # All should have exactly 1 token (EOS on first step)
            assert resp.shape[0] == 1
            assert info["logprob"].shape[0] == 1


# ── Clone cache helper tests ────────────────────────────────────────


@pytest.mark.unit
class TestCloneSingleCacheToBatch:
    """Unit tests for _clone_single_cache_to_batch."""

    def test_batch_size_1_is_noop(self):
        from textpolicy.generation.mlx_generation import _clone_single_cache_to_batch

        class FakeCache:
            keys = mx.zeros((1, 2, 4, 8))
            values = mx.ones((1, 2, 4, 8))

        cache = [FakeCache()]
        result = _clone_single_cache_to_batch(cache, batch_size=1)
        assert result is cache  # should be the exact same object

    def test_tiles_keys_and_values(self):
        from textpolicy.generation.mlx_generation import _clone_single_cache_to_batch

        class FakeCache:
            keys = mx.ones((1, 2, 4, 8))
            values = mx.ones((1, 2, 4, 8)) * 2.0

        cache = [FakeCache()]
        result = _clone_single_cache_to_batch(cache, batch_size=3)
        assert result[0].keys.shape == (3, 2, 4, 8)
        assert result[0].values.shape == (3, 2, 4, 8)
        # Each batch entry should match the original
        assert mx.allclose(result[0].keys[0], result[0].keys[2]).item()
        assert mx.allclose(result[0].values[0], result[0].values[1]).item()

    def test_handles_unpopulated_cache(self):
        from textpolicy.generation.mlx_generation import _clone_single_cache_to_batch

        class EmptyCache:
            keys = None
            values = None

        cache = [EmptyCache()]
        result = _clone_single_cache_to_batch(cache, batch_size=4)
        assert result[0].keys is None
