"""
Real-model performance litmus tests (Issue #44).

These tests intentionally use a real model (`arcee-ai/Trinity-Nano-Preview` by default)
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
from textpolicy.generation.lora import apply_quantization_to_model, apply_lora, freeze_base
from textpolicy.generation.mlx_generation import (
    _create_batched_decode_mask,
    _create_batched_prefill_mask,
    _get_eos_configs_for_model,
    _make_prompt_cache_if_available,
    _model_forward_with_optional_mask_and_cache,
    _prepare_tokenizer,
    batch_generate_tokens,
    load_model,
)
from textpolicy.training.trainer import Trainer


pytestmark = [pytest.mark.requires_model, pytest.mark.slow]

_MODEL_ENV = "TEXTPOLICY_REAL_MODEL"
_DEFAULT_MODEL = "arcee-ai/Trinity-Nano-Preview"
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


@pytest.fixture(scope="module")
def real_model_with_lora(real_model: Tuple[Any, Any]) -> Tuple[Any, Any]:
    """Model with LoRA adapters for training tests (matches experiment usage).

    Mutates the shared ``real_model`` in-place (apply_lora + freeze_base).
    A separate load would double memory to ~12 GB and cause OOM-induced
    slowdowns on Apple Silicon.  LoRA is transparent for inference, so tests
    that only generate (KV-cache, decode, logprobs) are unaffected.  The
    rollout timing test explicitly requests this fixture to avoid order
    dependence.
    """
    model, tokenizer = real_model
    from textpolicy.generation.lora import apply_lora, freeze_base

    apply_lora(model, lora_layers=4, lora_rank=2, lora_scale=8.0)
    freeze_base(model)
    return model, tokenizer


def _get_model_memory_bytes(model: Any) -> int:
    """Sum nbytes over all model parameters."""
    from mlx.utils import tree_flatten

    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


@pytest.fixture(scope="module")
def quantized_model() -> Tuple[Any, Any, dict]:
    """Independent 4-bit quantized model load for benchmark comparisons.

    Loads a fresh model copy via ``mlx_lm.load(return_config=True)`` so the
    FP16 ``real_model`` fixture stays unmodified for timing baselines.
    """
    try:
        from mlx_lm import load as mlx_lm_load
    except ImportError:
        pytest.skip("mlx_lm is required for quantized model tests.")

    model_name = os.environ.get(_MODEL_ENV, _DEFAULT_MODEL)
    try:
        tokenizer_config, model_config = _get_eos_configs_for_model(model_name, None)
        model, tokenizer, config = mlx_lm_load(
            path_or_hf_repo=model_name,
            tokenizer_config=tokenizer_config,
            model_config=model_config,
            lazy=False,
            return_config=True,
        )
        _prepare_tokenizer(tokenizer, verbose=False)
    except Exception as exc:
        pytest.skip(f"Could not load model '{model_name}': {exc}")

    fp16_bytes = _get_model_memory_bytes(model)

    try:
        model = apply_quantization_to_model(model, config, bits=4, group_size=64)
    except Exception as exc:
        pytest.skip(f"Quantization failed: {exc}")

    q4_bytes = _get_model_memory_bytes(model)
    stats = {
        "fp16_bytes": fp16_bytes,
        "q4_bytes": q4_bytes,
        "compression_ratio": fp16_bytes / max(q4_bytes, 1),
    }
    return model, tokenizer, stats


@pytest.fixture(scope="module")
def quantized_model_with_qlora(
    quantized_model: Tuple[Any, Any, dict],
) -> Tuple[Any, Any]:
    """4-bit quantized model with LoRA adapters for training tests (QLoRA)."""
    model, tokenizer, _stats = quantized_model
    apply_lora(model, lora_layers=4, lora_rank=2, lora_scale=8.0)
    freeze_base(model)
    return model, tokenizer


def test_kv_cache_active(real_model: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model
    cache_obj = _make_prompt_cache_if_available(model)
    assert cache_obj is not None, "KV-cache should be available for the test model."

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
    assert cache_obj is not None, "KV-cache should be available for the test model."

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


def test_real_model_training_step_finite_loss(real_model_with_lora: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model_with_lora
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer = _make_trainer(model, profile=False)
    metrics = trainer.train(batch)
    loss = float(metrics["loss"])

    assert math.isfinite(loss)
    assert -10.0 < loss < 100.0


@pytest.mark.apple_silicon
def test_training_step_under_5s(real_model_with_lora: Tuple[Any, Any]) -> None:
    _require_apple_silicon()
    model, tokenizer = real_model_with_lora
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer = _make_trainer(model, profile=False)
    trainer.train(batch)  # warmup

    t0 = time.perf_counter()
    metrics = trainer.train(batch)
    elapsed_s = time.perf_counter() - t0

    assert math.isfinite(float(metrics["loss"]))
    assert elapsed_s < 5.0, f"Training step took {elapsed_s:.2f}s (expected < 5.0s)."


@pytest.mark.apple_silicon
def test_rollout_under_30s(real_model_with_lora: Tuple[Any, Any]) -> None:
    _require_apple_silicon()
    model, tokenizer = real_model_with_lora
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


def test_profiling_overhead_under_25pct(real_model_with_lora: Tuple[Any, Any]) -> None:
    model, tokenizer = real_model_with_lora
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


# ---------------------------------------------------------------------------
# Quantized-model litmus tests (Issue #43)
# ---------------------------------------------------------------------------


def test_quantized_model_generates_valid_text(
    quantized_model: Tuple[Any, Any, dict],
) -> None:
    """L1: 4-bit quantized model produces valid logprobs and non-empty output."""
    model, tokenizer, stats = quantized_model
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
    assert logprobs.shape[0] > 0, "Quantized model produced empty output."
    assert bool(mx.all(logprobs <= 0).item()), "Logprobs must be ≤ 0."
    assert not bool(mx.any(mx.isnan(logprobs)).item()), "NaN in logprobs."
    assert not bool(mx.any(mx.isinf(logprobs)).item()), "Inf in logprobs."

    fp16_mb = stats["fp16_bytes"] / 1e6
    q4_mb = stats["q4_bytes"] / 1e6
    print(
        f"\n  Memory: FP16={fp16_mb:.1f}MB → Q4={q4_mb:.1f}MB "
        f"({stats['compression_ratio']:.2f}x compression)"
    )


@pytest.mark.apple_silicon
def test_quantized_decode_step_faster(
    real_model: Tuple[Any, Any],
    quantized_model: Tuple[Any, Any, dict],
) -> None:
    """L2: Per-decode-step latency is lower with 4-bit quantization."""
    _require_apple_silicon()

    fp16_model, tokenizer_fp16 = real_model
    q4_model, tokenizer_q4, stats = quantized_model

    # --- shared setup: prefill then measure single-token decode steps ---
    seed_token = int(_encode_prompt(tokenizer_fp16, "hello")[0].item())
    prompt_len = 128
    prompt_batch = mx.full((1, prompt_len), seed_token, dtype=mx.int32)
    prefill_mask = _create_batched_prefill_mask([prompt_len], prompt_len)
    token_batch = mx.array([[seed_token]], dtype=mx.int32)

    def _setup_and_measure(model: Any) -> float:
        cache = _make_prompt_cache_if_available(model)
        assert cache is not None
        logits, _ = _model_forward_with_optional_mask_and_cache(
            model, prompt_batch, mask=prefill_mask, cache_obj=cache,
        )
        mx.eval(logits)

        offset = {"v": 1}

        def decode_step() -> mx.array:
            mask = _create_batched_decode_mask(
                [prompt_len], prompt_len, decode_offset=offset["v"],
            )
            out, _ = _model_forward_with_optional_mask_and_cache(
                model, token_batch, mask=mask, cache_obj=cache,
            )
            offset["v"] += 1
            return out

        return _median_ms(decode_step, repeats=8, warmup=1)

    fp16_ms = _setup_and_measure(fp16_model)
    q4_ms = _setup_and_measure(q4_model)
    speedup = fp16_ms / q4_ms

    fp16_mb = stats["fp16_bytes"] / 1e6
    q4_mb = stats["q4_bytes"] / 1e6
    print(
        f"\n  Decode: FP16={fp16_ms:.2f}ms, Q4={q4_ms:.2f}ms "
        f"({speedup:.2f}x speedup)"
        f"\n  Memory: FP16={fp16_mb:.1f}MB → Q4={q4_mb:.1f}MB"
    )

    assert speedup >= 1.3, (
        f"Expected ≥1.3x decode speedup from quantization "
        f"(FP16={fp16_ms:.2f}ms, Q4={q4_ms:.2f}ms, speedup={speedup:.2f}x)."
    )


@pytest.mark.apple_silicon
def test_quantized_rollout_faster(
    real_model: Tuple[Any, Any],
    quantized_model: Tuple[Any, Any, dict],
) -> None:
    """L3: Full batch rollout is faster with 4-bit quantization."""
    _require_apple_silicon()

    fp16_model, tokenizer_fp16 = real_model
    q4_model, tokenizer_q4, stats = quantized_model

    no_eos_fp16 = _NoEOSTokenizerProxy(tokenizer_fp16)
    no_eos_q4 = _NoEOSTokenizerProxy(tokenizer_q4)

    def rollout_fp16() -> Any:
        prompts = _make_prompt_batch(tokenizer_fp16, n=4)
        return batch_generate_tokens(
            fp16_model, no_eos_fp16, prompts, max_tokens=64, temperature=0.0,
        )

    def rollout_q4() -> Any:
        prompts = _make_prompt_batch(tokenizer_q4, n=4)
        return batch_generate_tokens(
            q4_model, no_eos_q4, prompts, max_tokens=64, temperature=0.0,
        )

    fp16_ms = _median_ms(rollout_fp16, repeats=4, warmup=1)
    q4_ms = _median_ms(rollout_q4, repeats=4, warmup=1)
    speedup = fp16_ms / q4_ms

    print(
        f"\n  Rollout: FP16={fp16_ms:.1f}ms, Q4={q4_ms:.1f}ms "
        f"({speedup:.2f}x speedup)"
    )

    assert speedup >= 1.2, (
        f"Expected ≥1.2x rollout speedup from quantization "
        f"(FP16={fp16_ms:.1f}ms, Q4={q4_ms:.1f}ms, speedup={speedup:.2f}x)."
    )


def test_qlora_training_reduces_loss(
    quantized_model_with_qlora: Tuple[Any, Any],
) -> None:
    """L4: QLoRA training on a 4-bit model produces finite, reasonable losses."""
    model, tokenizer = quantized_model_with_qlora
    batch = _build_real_rollout_batch(model, tokenizer, n_episodes=4, max_tokens=32)

    trainer = _make_trainer(model, profile=False)
    metrics_1 = trainer.train(batch)
    loss_1 = float(metrics_1["loss"])

    metrics_2 = trainer.train(batch)
    loss_2 = float(metrics_2["loss"])

    assert math.isfinite(loss_1), f"Step 1 loss is not finite: {loss_1}"
    assert math.isfinite(loss_2), f"Step 2 loss is not finite: {loss_2}"
    assert -10.0 < loss_1 < 100.0, f"Step 1 loss out of range: {loss_1}"
    assert -10.0 < loss_2 < 100.0, f"Step 2 loss out of range: {loss_2}"

    print(f"\n  QLoRA losses: step1={loss_1:.4f}, step2={loss_2:.4f}")
