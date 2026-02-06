"""
Real-model performance litmus tests (Issue #44).

These tests intentionally use a real model (`Qwen/Qwen3-0.6B` by default)
to prevent synthetic benchmark blind spots.
"""

from __future__ import annotations

import math
import os
import platform
import statistics
import time
from typing import Any, Callable, List, Tuple

import mlx.core as mx
import mlx.optimizers as optim
import pytest

from textpolicy.algorithms import grpo
from textpolicy.algorithms.grpo import _pack_episodes
from textpolicy.generation.mlx_generation import (
    _create_batched_decode_mask,
    _create_batched_prefill_mask,
    _make_prompt_cache_if_available,
    _model_forward_with_optional_mask_and_cache,
    batch_generate_tokens,
    load_model,
)
from textpolicy.training.trainer import Trainer


pytestmark = [pytest.mark.requires_model, pytest.mark.slow]

_MODEL_ENV = "TEXTPOLICY_REAL_MODEL"
_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
_PROMPT_TEXTS = [
    "Summarize reinforcement learning in one sentence.",
    "List two risks of overfitting in language models.",
    "Explain gradient clipping in plain language.",
    "Why does batching improve throughput on Apple Silicon?",
]


class _NoEOSTokenizerProxy:
    """Tokenizer wrapper that disables EOS stopping for fixed-length rollouts."""

    def __init__(self, base_tokenizer: Any):
        self._base = base_tokenizer
        self.pad_token_id = int(getattr(base_tokenizer, "pad_token_id", 0))
        self.eos_token_ids: List[int] = []
        self.eos_token_id = -1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _require_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Timing threshold is calibrated for Apple Silicon only.")


def _encode_prompt(tokenizer: Any, text: str) -> mx.array:
    if not hasattr(tokenizer, "encode"):
        pytest.skip("Tokenizer does not expose encode().")

    token_ids = tokenizer.encode(text)
    if not token_ids:
        fallback_id = getattr(tokenizer, "bos_token_id", None)
        if fallback_id is None:
            fallback_id = 1
        token_ids = [int(fallback_id)]
    return mx.array(token_ids, dtype=mx.int32)


def _gather_arrays(obj: Any, out: List[mx.array]) -> None:
    if isinstance(obj, mx.array):
        out.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _gather_arrays(value, out)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _gather_arrays(value, out)


def _eval_nested(obj: Any) -> None:
    arrays: List[mx.array] = []
    _gather_arrays(obj, arrays)
    if arrays:
        mx.eval(*arrays)


def _median_ms(fn: Callable[[], Any], repeats: int = 6, warmup: int = 1) -> float:
    for _ in range(warmup):
        _eval_nested(fn())

    samples_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        output = fn()
        _eval_nested(output)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(statistics.median(samples_ms))


def _make_prompt_batch(tokenizer: Any, n: int) -> List[mx.array]:
    prompts = []
    for i in range(n):
        prompts.append(_encode_prompt(tokenizer, _PROMPT_TEXTS[i % len(_PROMPT_TEXTS)]))
    return prompts


def _build_real_rollout_batch(
    model: Any,
    tokenizer: Any,
    n_episodes: int = 4,
    max_tokens: int = 32,
) -> dict:
    prompt_batch = _make_prompt_batch(tokenizer, n_episodes)
    no_eos_tokenizer = _NoEOSTokenizerProxy(tokenizer)
    generated = batch_generate_tokens(
        model,
        no_eos_tokenizer,
        prompt_batch,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    _eval_nested(generated)

    episodes = []
    for prompt_tokens, (response_tokens, info) in zip(prompt_batch, generated):
        logprobs = info["logprob"]
        episodes.append(
            {
                "obs": prompt_tokens.tolist(),
                "act": response_tokens.tolist(),
                "rew": [float(response_tokens.shape[0])],
                "logprob": logprobs.tolist(),
            }
        )

    return _pack_episodes(episodes)


def _make_trainer(model: Any, *, profile: bool = False) -> Trainer:
    return Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,
        loss_fn=grpo.policy_loss,
        optimizer=optim.Adam(learning_rate=0.0),
        compile_training=False,
        profile=profile,
    )


@pytest.fixture(scope="module")
def real_model() -> Tuple[Any, Any]:
    model_name = os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)
    try:
        model, tokenizer = load_model(model_name)
    except Exception as exc:
        pytest.skip(f"Could not load real model '{model_name}': {exc}")
    return model, tokenizer


def test_kv_cache_active(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    cache_obj = _make_prompt_cache_if_available(model)
    assert cache_obj is not None, "KV-cache should be available for Qwen3-0.6B."

    prompt = _encode_prompt(tokenizer, "Cache litmus test.")
    prompt_batch = prompt.reshape(1, -1)
    prompt_len = int(prompt_batch.shape[1])
    mask = _create_batched_prefill_mask([prompt_len], prompt_len)

    logits, used_cache = _model_forward_with_optional_mask_and_cache(
        model,
        prompt_batch,
        mask=mask,
        cache_obj=cache_obj,
    )
    mx.eval(logits)
    assert used_cache, "Model forward should accept and use cache."


def test_cached_decode_faster(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    cache_obj = _make_prompt_cache_if_available(model)
    assert cache_obj is not None, "KV-cache should be available for Qwen3-0.6B."

    seed_token = int(_encode_prompt(tokenizer, "hello")[0].item())
    prompt_len = 128
    prompt_batch = mx.full((1, prompt_len), seed_token, dtype=mx.int32)
    prefill_mask = _create_batched_prefill_mask([prompt_len], prompt_len)

    prefill_logits, used_cache = _model_forward_with_optional_mask_and_cache(
        model,
        prompt_batch,
        mask=prefill_mask,
        cache_obj=cache_obj,
    )
    mx.eval(prefill_logits)
    assert used_cache, "Expected cache usage during prefill."

    decode_state = {"offset": 1}
    token_batch = mx.array([[seed_token]], dtype=mx.int32)

    def cached_decode() -> mx.array:
        decode_mask = _create_batched_decode_mask(
            [prompt_len],
            prompt_len,
            decode_offset=decode_state["offset"],
        )
        logits, decode_used_cache = _model_forward_with_optional_mask_and_cache(
            model,
            token_batch,
            mask=decode_mask,
            cache_obj=cache_obj,
        )
        decode_state["offset"] += 1
        if not decode_used_cache:
            raise AssertionError("Decode path stopped using cache.")
        return logits

    def full_forward() -> mx.array:
        return model(prompt_batch)

    cached_ms = _median_ms(cached_decode, repeats=8, warmup=1)
    full_ms = _median_ms(full_forward, repeats=8, warmup=1)
    speedup = full_ms / cached_ms

    assert speedup >= 3.0, (
        f"Expected cached decode to be >=3x faster than full forward "
        f"(cached={cached_ms:.2f}ms, full={full_ms:.2f}ms, speedup={speedup:.2f}x)."
    )


def test_real_model_logprobs_valid(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    prompts = [_encode_prompt(tokenizer, "Hello, how are you?")]

    results = batch_generate_tokens(
        model,
        tokenizer,
        prompts,
        max_tokens=20,
        temperature=0.0,
    )
    _eval_nested(results)
    response, info = results[0]
    logprobs = info["logprob"]

    assert response.shape[0] == logprobs.shape[0]
    assert logprobs.shape[0] > 0
    assert bool(mx.all(logprobs <= 0).item())
    assert not bool(mx.any(mx.isnan(logprobs)).item())
    assert not bool(mx.any(mx.isinf(logprobs)).item())


def test_real_model_training_step_finite_loss(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer = _make_trainer(model, profile=False)
    metrics = trainer.train(batch)
    loss = float(metrics["loss"])

    assert math.isfinite(loss)
    assert -10.0 < loss < 100.0


@pytest.mark.apple_silicon
def test_training_step_under_5s(real_model: Tuple[Any, Any]) -> None:
    _require_apple_silicon()
    model, tokenizer = real_model
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer = _make_trainer(model, profile=False)
    trainer.train(batch)  # warmup

    t0 = time.perf_counter()
    metrics = trainer.train(batch)
    elapsed_s = time.perf_counter() - t0

    assert math.isfinite(float(metrics["loss"]))
    assert elapsed_s < 5.0, f"Training step took {elapsed_s:.2f}s (expected < 5.0s)."


@pytest.mark.apple_silicon
def test_rollout_under_30s(real_model: Tuple[Any, Any]) -> None:
    _require_apple_silicon()
    model, tokenizer = real_model
    prompts = _make_prompt_batch(tokenizer, n=4)
    no_eos_tokenizer = _NoEOSTokenizerProxy(tokenizer)

    t0 = time.perf_counter()
    results = batch_generate_tokens(
        model,
        no_eos_tokenizer,
        prompts,
        max_tokens=128,
        temperature=0.0,
    )
    _eval_nested(results)
    elapsed_s = time.perf_counter() - t0

    assert len(results) == 4
    assert elapsed_s < 30.0, f"Rollout took {elapsed_s:.2f}s (expected < 30.0s)."


def test_profiling_overhead_under_25pct(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer_unprofiled = _make_trainer(model, profile=False)
    trainer_profiled = _make_trainer(model, profile=True)

    trainer_unprofiled.train(batch)  # warmup
    trainer_profiled.train(batch)    # warmup

    unprofiled_ms = _median_ms(lambda: trainer_unprofiled.train(batch), repeats=4, warmup=0)
    profiled_ms = _median_ms(lambda: trainer_profiled.train(batch), repeats=4, warmup=0)
    overhead = (profiled_ms - unprofiled_ms) / max(unprofiled_ms, 1e-9)

    assert overhead < 0.25, (
        f"Profiling overhead {overhead * 100:.1f}% exceeds 25% "
        f"(profiled={profiled_ms:.2f}ms, unprofiled={unprofiled_ms:.2f}ms)."
    )
