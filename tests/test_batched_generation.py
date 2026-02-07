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
