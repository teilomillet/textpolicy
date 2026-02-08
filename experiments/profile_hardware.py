#!/usr/bin/env python3
"""
Hardware profiling for TextPolicy training on Apple Silicon.

Runs real model loading + LoRA setup + rollout + training at multiple sequence
lengths to discover where the O(n^2) attention wall hits for your specific
chip/memory configuration.

Produces:
- Scaling table: gen/train time + peak memory at each sequence length
- Bottleneck analysis: Amdahl's law breakdown showing which sub-phases dominate
- Little's Law throughput: tokens/second at each stage to identify stalls
- Scaling regime detection: are you compute-bound or memory-bandwidth-bound?
- Actionable recommendation: where to focus optimization effort

Usage:
    uv run python experiments/profile_hardware.py
    uv run python experiments/profile_hardware.py --model arcee-ai/Trinity-Nano-Preview
    uv run python experiments/profile_hardware.py --quick
    uv run python experiments/profile_hardware.py --json --output results/my_hardware.json
    uv run python experiments/profile_hardware.py --max-seq-lengths 256 512 1024 2048 4096
"""

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.training import create_tinylora_reasoning_setup
from textpolicy.utils.memory import clear_memory, get_memory_stats


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass
class ProfileConfig:
    """Configuration for hardware profiling."""

    model_id: str = "arcee-ai/Trinity-Nano-Preview"
    seq_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    group_size: int = 8
    steps_per_probe: int = 2
    num_problems: int = 20
    dataset_seed: int = 42
    reference_run_steps: int = 210  # GTPO convergence target
    gradient_checkpointing: bool = False
    micro_batch_size: Optional[int] = None
    json_output: bool = False
    output_path: Optional[str] = None
    quick: bool = False
    timeout_per_step: float = 600.0  # seconds


# ── Hardware detection ───────────────────────────────────────────────────────


def detect_hardware() -> Dict[str, Any]:
    """Detect Apple Silicon chip name and total memory via system_profiler."""
    info: Dict[str, Any] = {
        "chip": "Unknown",
        "total_memory_gb": 0.0,
        "os": "Unknown",
    }

    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if "Chip" in line and ":" in line:
                info["chip"] = line.split(":", 1)[1].strip()
            elif "Memory" in line and ":" in line and "GB" in line:
                parts = line.split(":", 1)[1].strip()
                try:
                    info["total_memory_gb"] = float(parts.replace("GB", "").strip())
                except ValueError:
                    pass

        result2 = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["os"] = f"macOS {result2.stdout.strip()}"
    except Exception:
        pass

    return info


# ── Scaling probe ────────────────────────────────────────────────────────────


@dataclass
class ProbeResult:
    """Result from a single scaling probe point."""

    seq_length: int
    group_size: int
    gen_time_s: float
    train_time_s: float
    total_time_s: float
    peak_memory_mb: float
    status: str  # "OK", "OOM", "TIMEOUT", "ERROR", "SKIPPED"
    error_msg: Optional[str] = None
    # Sub-phase breakdowns (populated when profiling is enabled)
    rollout_phases: Optional[Dict[str, float]] = None
    trainer_phases: Optional[Dict[str, float]] = None
    # Throughput metrics for Little's Law
    total_tokens_generated: int = 0
    # Number of fully completed probe steps
    steps_completed: int = 0


def run_probe(
    seq_length: int,
    group_size: int,
    steps: int,
    model: Any,
    tokenizer: Any,
    trainer: Any,
    problems: List[Dict],
    prompts: List[str],
    timeout: float,
) -> ProbeResult:
    """Run a scaling probe at a specific sequence length.

    Executes `steps` training iterations and records generation time,
    training time, peak memory, and sub-phase breakdowns from the
    Trainer and RolloutCoordinator profiling infrastructure.
    """
    gen_times: List[float] = []
    train_times: List[float] = []
    rollout_phase_accum: Dict[str, float] = {}
    trainer_phase_accum: Dict[str, float] = {}
    total_tokens = 0
    completed_steps = 0
    rollout: Optional[RolloutCoordinator] = None

    try:
        # Reset peak memory tracking before probe
        mx.reset_peak_memory()
        # Reset trainer step count so each probe starts from step 0.
        # Currently cosmetic (metrics_fn is None), but prevents subtle
        # issues if a learning rate scheduler or metrics_fn is added later.
        trainer._step_count = 0

        def create_env():
            return TextGenerationEnv(
                prompts=prompts,
                reward_fn=countdown_reward,
                max_tokens=seq_length,
                tokenizer=tokenizer,
                examples=problems,
                group_size=group_size,
            )

        policy_fn = create_policy(
            model,
            tokenizer,
            generation_params={
                "max_tokens": seq_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
        )

        rollout = RolloutCoordinator(
            env_fn=create_env,
            policy_fn=lambda: policy_fn,
            algorithm="grpo",
            num_workers=0,
            max_steps=group_size,
            max_episodes=group_size,
            batch_size=group_size,
            model=model,
            tokenizer=tokenizer,
            generation_params={
                "max_tokens": seq_length,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
            profile=True,  # Enable rollout sub-phase timing
        )

        buffer = Buffer(max_episodes=group_size)
        trainer.link_buffer(buffer, data_selector_fn=grpo.select_recent_data)

        for step_i in range(steps):
            step_start = time.perf_counter()

            buffer.clear()

            # Generation phase
            gen_start = time.perf_counter()
            rollout_buffer = rollout.collect()
            for ep in rollout_buffer.episodes:
                buffer.add_episode_from_dict(ep.to_dict())
            # Evaluate model parameters to flush any pending lazy computation
            # from the generation phase.  mx.eval() with no args is a no-op;
            # explicit arrays are required.  The rollout runner's internal
            # barriers should already handle this, but we add a defence-in-depth
            # barrier to ensure generation cost is not leaked into train timing.
            mx.eval(model.parameters())
            gen_time = time.perf_counter() - gen_start
            gen_times.append(gen_time)

            # Per-step timeout is checked at phase boundaries.
            if time.perf_counter() - step_start > timeout:
                return ProbeResult(
                    seq_length=seq_length,
                    group_size=group_size,
                    gen_time_s=_mean(gen_times),
                    train_time_s=_mean(train_times),
                    total_time_s=_mean(gen_times) + _mean(train_times),
                    peak_memory_mb=mx.get_peak_memory() / 1024 / 1024,
                    status="TIMEOUT",
                    error_msg=(
                        f"Exceeded {timeout:.0f}s timeout during generation "
                        f"at step {step_i + 1}"
                    ),
                    rollout_phases=_average_phases(
                        rollout_phase_accum, max(1, len(gen_times))
                    ),
                    trainer_phases=_average_phases(
                        trainer_phase_accum, max(1, len(train_times))
                    ),
                    total_tokens_generated=total_tokens,
                    steps_completed=completed_steps,
                )

            # Accumulate rollout sub-phase timing
            for phase, secs in rollout.get_rollout_timing().items():
                rollout_phase_accum[phase] = rollout_phase_accum.get(phase, 0.0) + secs

            # Count tokens generated in this step.
            # Episode.act is a list of actions (one per env step).
            # In single-turn text generation, each action is itself a
            # list of response token IDs, so we sum their lengths.
            for ep in rollout_buffer.episodes:
                for action in ep.act:
                    if isinstance(action, (list, tuple)):
                        total_tokens += len(action)
                    else:
                        total_tokens += 1  # scalar action

            # Training phase
            train_start = time.perf_counter()
            metrics = trainer.train()
            # Flush any pending lazy computation from the training phase.
            # trainer.train() has internal mx.eval(loss, grads) barriers, but
            # metrics computation may leave lazy arrays.  Evaluating model
            # parameters ensures training cost is fully attributed here.
            mx.eval(model.parameters())
            train_time = time.perf_counter() - train_start
            train_times.append(train_time)

            # Accumulate trainer sub-phase timing from metrics
            for key, value in metrics.items():
                if key.startswith("timing/") and key.endswith("_s"):
                    phase = key[len("timing/"):-2]
                    trainer_phase_accum[phase] = (
                        trainer_phase_accum.get(phase, 0.0) + float(value)
                    )
            completed_steps += 1

            if time.perf_counter() - step_start > timeout:
                return ProbeResult(
                    seq_length=seq_length,
                    group_size=group_size,
                    gen_time_s=_mean(gen_times),
                    train_time_s=_mean(train_times),
                    total_time_s=_mean(gen_times) + _mean(train_times),
                    peak_memory_mb=mx.get_peak_memory() / 1024 / 1024,
                    status="TIMEOUT",
                    error_msg=(
                        f"Exceeded {timeout:.0f}s timeout during training "
                        f"at step {step_i + 1}"
                    ),
                    rollout_phases=_average_phases(
                        rollout_phase_accum, max(1, len(gen_times))
                    ),
                    trainer_phases=_average_phases(
                        trainer_phase_accum, max(1, len(train_times))
                    ),
                    total_tokens_generated=total_tokens,
                    steps_completed=completed_steps,
                )

        peak_mem = mx.get_peak_memory() / 1024 / 1024

        return ProbeResult(
            seq_length=seq_length,
            group_size=group_size,
            gen_time_s=_mean(gen_times),
            train_time_s=_mean(train_times),
            total_time_s=_mean(gen_times) + _mean(train_times),
            peak_memory_mb=peak_mem,
            status="OK",
            rollout_phases=_average_phases(rollout_phase_accum, steps),
            trainer_phases=_average_phases(trainer_phase_accum, steps),
            total_tokens_generated=total_tokens,
            steps_completed=completed_steps,
        )

    except Exception as e:
        error_str = str(e)
        status = "OOM" if "memory" in error_str.lower() else "ERROR"
        return ProbeResult(
            seq_length=seq_length,
            group_size=group_size,
            gen_time_s=_mean(gen_times),
            train_time_s=_mean(train_times),
            total_time_s=_mean(gen_times) + _mean(train_times),
            peak_memory_mb=mx.get_peak_memory() / 1024 / 1024,
            status=status,
            error_msg=error_str[:200],
            rollout_phases=_average_phases(rollout_phase_accum, max(1, len(gen_times))),
            trainer_phases=_average_phases(trainer_phase_accum, max(1, len(train_times))),
            total_tokens_generated=total_tokens,
            steps_completed=completed_steps,
        )
    finally:
        if rollout is not None:
            try:
                rollout.close()
            except Exception:
                pass
        clear_memory()


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _average_phases(accum: Dict[str, float], steps: int) -> Dict[str, float]:
    """Convert accumulated phase totals to per-step averages."""
    if steps <= 0:
        return dict(accum)
    return {k: v / steps for k, v in accum.items()}


# ── Analysis & extrapolation ─────────────────────────────────────────────────


def fit_scaling_exponent(
    results: List[ProbeResult],
) -> Tuple[Optional[float], Optional[float]]:
    """Fit time = a * n^b to the training times of OK probes.

    Uses log-linear regression: log(time) = log(a) + b * log(n).

    Returns:
        (a, b) coefficients, or (None, None) if insufficient data.
    """
    ok_results = [r for r in results if r.status == "OK" and r.train_time_s > 0]
    if len(ok_results) < 2:
        return None, None

    log_n = [math.log(r.seq_length) for r in ok_results]
    log_t = [math.log(r.train_time_s) for r in ok_results]

    n = len(log_n)
    sum_x = sum(log_n)
    sum_y = sum(log_t)
    sum_xy = sum(x * y for x, y in zip(log_n, log_t))
    sum_xx = sum(x * x for x in log_n)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None, None

    b = (n * sum_xy - sum_x * sum_y) / denom
    log_a = (sum_y - b * sum_x) / n
    a = math.exp(log_a)

    return a, b


def extrapolate_time(a: float, b: float, seq_length: int) -> float:
    """Predict training time per step at a given sequence length."""
    return a * (seq_length**b)


def extrapolate_memory(
    results: List[ProbeResult], seq_length: int
) -> Optional[float]:
    """Rough linear extrapolation of peak memory vs seq_length."""
    ok = [r for r in results if r.status == "OK"]
    if len(ok) < 2:
        return None
    n = len(ok)
    xs = [r.seq_length for r in ok]
    ys = [r.peak_memory_mb for r in ok]
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    sum_xx = sum(x * x for x in xs)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return None
    d = (n * sum_xy - sum_x * sum_y) / denom
    c = (sum_y - d * sum_x) / n
    return c + d * seq_length


# ── Bottleneck analysis ──────────────────────────────────────────────────────


@dataclass
class BottleneckAnalysis:
    """Amdahl's Law + Little's Law analysis for a single probe point."""

    seq_length: int
    # Amdahl: phase → (seconds, fraction, max_speedup_if_eliminated)
    amdahl_phases: List[Tuple[str, float, float, float]]
    # The single biggest bottleneck
    bottleneck_phase: str
    bottleneck_fraction: float
    amdahl_limit: float  # max theoretical speedup from eliminating bottleneck
    # Little's Law: throughput metrics
    tokens_per_second_gen: float  # generation throughput
    tokens_per_second_train: float  # training throughput (forward+backward)
    # Regime classification
    regime: str  # "compute-bound", "memory-bandwidth-bound", "balanced"


def analyze_bottlenecks(results: List[ProbeResult]) -> List[BottleneckAnalysis]:
    """Run Amdahl's Law and Little's Law analysis on each successful probe.

    Amdahl's Law tells you: "Even if you make phase X infinitely fast,
    the overall speedup is bounded by 1/(1-f) where f is the fraction
    of time spent in X."  This focuses optimization effort.

    Little's Law (L = lambda * W) tells you: "The number of tokens
    in-flight = throughput * latency."  When throughput drops as
    seq_length grows, you know the GPU is stalling.
    """
    analyses: List[BottleneckAnalysis] = []

    for r in results:
        if r.status != "OK":
            continue

        # Build unified phase list from rollout + trainer sub-phases
        phases: List[Tuple[str, float]] = []
        total_time = r.total_time_s

        if r.rollout_phases:
            for phase, secs in r.rollout_phases.items():
                if phase == "total":
                    continue
                phases.append((f"rollout/{phase}", secs))

        if r.trainer_phases:
            for phase, secs in r.trainer_phases.items():
                if phase == "total":
                    continue
                phases.append((f"train/{phase}", secs))

        # If sub-phases don't cover all time, add an "other" bucket
        sub_total = sum(s for _, s in phases)
        if total_time > 0 and sub_total < total_time * 0.95:
            phases.append(("overhead", total_time - sub_total))

        # Sort by time descending
        phases.sort(key=lambda x: x[1], reverse=True)

        # Compute Amdahl fractions
        amdahl_phases: List[Tuple[str, float, float, float]] = []
        for name, secs in phases:
            if total_time <= 0:
                continue
            frac = secs / total_time
            # Amdahl limit: if this phase became instant, max speedup = 1/(1-f)
            limit = 1.0 / (1.0 - frac) if frac < 0.9999 else float("inf")
            amdahl_phases.append((name, secs, frac, limit))

        bottleneck = amdahl_phases[0] if amdahl_phases else ("unknown", 0, 0, 1.0)

        # Little's Law: tokens/second
        tokens_gen = r.total_tokens_generated
        steps_completed = max(1, r.steps_completed)
        total_gen_time = r.gen_time_s * steps_completed
        total_train_time = r.train_time_s * steps_completed
        tps_gen = tokens_gen / total_gen_time if total_gen_time > 0 else 0.0
        # For training, total tokens pass through forward + backward
        tps_train = tokens_gen / total_train_time if total_train_time > 0 else 0.0

        # Regime classification based on fraction of time in loss_and_grad
        # (the O(n^2) attention bottleneck) vs generation (memory-bandwidth).
        loss_grad_frac = 0.0
        gen_frac = 0.0
        if total_time > 0:
            if r.trainer_phases:
                loss_grad_frac = r.trainer_phases.get("loss_and_grad", 0.0) / total_time
            if r.rollout_phases:
                gen_frac = r.rollout_phases.get("generation", 0.0) / total_time

        if loss_grad_frac > 0.4:
            regime = "compute-bound"
        elif loss_grad_frac > 0.2:
            regime = "transitioning"
        elif gen_frac > 0.7:
            regime = "memory-bandwidth-bound"
        else:
            regime = "balanced"

        analyses.append(
            BottleneckAnalysis(
                seq_length=r.seq_length,
                amdahl_phases=amdahl_phases,
                bottleneck_phase=bottleneck[0],
                bottleneck_fraction=bottleneck[2],
                amdahl_limit=bottleneck[3],
                tokens_per_second_gen=tps_gen,
                tokens_per_second_train=tps_train,
                regime=regime,
            )
        )

    return analyses


# ── Analytical Memory Model ──────────────────────────────────────────────────


@dataclass
class ModelArchitecture:
    """Architecture parameters extracted from model.args."""

    num_layers: int
    num_full_attn_layers: int
    num_sliding_attn_layers: int
    sliding_window: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts_per_tok: int
    vocab_size: int
    bytes_per_param: int = 2  # bfloat16


def extract_architecture(model: Any) -> ModelArchitecture:
    """Extract architecture parameters from a loaded MLX model.

    Works with any model that has a `.args` attribute (all mlx_lm models).
    Specifically calibrated for arcee-ai/Trinity-Nano-Preview (afmoe).
    """
    args = getattr(model, "args", None)
    if args is None:
        raise ValueError("Model has no .args attribute — cannot extract architecture")

    layer_types = getattr(args, "layer_types", None)
    if layer_types is None:
        layer_types = ["full_attention"] * args.num_hidden_layers

    num_full = sum(1 for lt in layer_types if "full" in lt)
    num_sliding = sum(1 for lt in layer_types if "sliding" in lt)

    return ModelArchitecture(
        num_layers=args.num_hidden_layers,
        num_full_attn_layers=num_full,
        num_sliding_attn_layers=num_sliding,
        sliding_window=getattr(args, "sliding_window", None) or 131072,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=getattr(
            args, "num_key_value_heads", args.num_attention_heads
        ),
        head_dim=getattr(
            args, "head_dim", args.hidden_size // args.num_attention_heads
        ),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        moe_intermediate_size=getattr(args, "moe_intermediate_size", 0),
        num_experts_per_tok=getattr(args, "num_experts_per_tok", 0),
        vocab_size=args.vocab_size,
    )


def _raw_activation_gb(
    arch: ModelArchitecture, G: int, seq_len: int
) -> float:
    """Compute raw activation memory in GB from architecture (before calibration).

    Memory components:
    1. KV cache (generation): G × L × 2 × Hkv × d × seq × 2 bytes
    2. Linear activations (forward + backward tape): ~6 × D × seq per layer
    3. Attention scores (the O(n²) term):
       - Full attention: G × H × seq² × 4 bytes (float32 softmax)
       - Sliding window: G × H × seq × min(seq, W) × 4 bytes
    """
    L = arch.num_layers
    H = arch.num_attention_heads
    Hkv = arch.num_kv_heads
    d = arch.head_dim
    D = arch.hidden_size
    W = arch.sliding_window

    kv_cache = G * L * 2 * Hkv * d * seq_len * 2
    linear = G * L * 6 * D * seq_len * 2
    full_attn = G * H * seq_len * seq_len * 4 * arch.num_full_attn_layers
    context = min(seq_len, W)
    sliding_attn = G * H * seq_len * context * 4 * arch.num_sliding_attn_layers

    return (kv_cache + linear + full_attn + sliding_attn) / (1024**3)


def predict_peak_memory_gb(
    arch: ModelArchitecture,
    base_memory_gb: float,
    G: int,
    seq_len: int,
    calibration_offset: float = 0.0,
    calibration_scale: float = 1.0,
) -> float:
    """Predict peak memory for a (G, seq_len) configuration.

    Formula:  peak = (base_memory + offset) + scale × raw_activation(G, seq_len)

    The offset captures runtime overhead (MLX framework, gradient tape baseline).
    The scale adjusts for unknowns in the activation formula.
    The raw_activation formula captures the O(n²) attention scaling from
    first principles.
    """
    return (
        base_memory_gb
        + calibration_offset
        + calibration_scale * _raw_activation_gb(arch, G, seq_len)
    )


def calibrate_analytical_model(
    arch: ModelArchitecture,
    base_memory_gb: float,
    results: List[ProbeResult],
) -> Tuple[float, float, Dict[str, Any]]:
    """Fit (base_offset, scale) from empirical probe measurements.

    The formula is:  predicted = (base_memory + offset) + scale × raw_activation

    The offset absorbs runtime overhead (MLX framework, Python, gradient tape
    baseline, OS) that isn't captured by the model weight footprint.
    The scale adjusts for unknowns in the activation formula.

    With 3+ data points: fits offset+scale via least-squares.
    With <=2 points or poor fit: falls back to a conservative calibration.
    With 1 data point: fixes scale=1.0, solves for offset.
    With 0 data points: returns defaults (offset=0, scale=1.0).
    """
    def _fit_errors(
        offset_gb: float, scale_value: float
    ) -> Tuple[float, float, float]:
        """Return (mape_pct, max_abs_error_gb, max_underpredict_gb)."""
        if not raws:
            return 0.0, 0.0, 0.0
        preds = [offset_gb + scale_value * x for x in raws]
        abs_err = [abs(p - m) for p, m in zip(preds, measured)]
        under = [max(0.0, m - p) for p, m in zip(preds, measured)]
        mape_terms = []
        for p, m in zip(preds, measured):
            # Stabilize denominator for tiny values (~<256MB activation delta).
            denom = max(abs(m), 0.25)
            mape_terms.append(abs(p - m) / denom)
        mape_pct = 100.0 * sum(mape_terms) / len(mape_terms)
        return mape_pct, max(abs_err), max(under)

    ok = [r for r in results if r.status == "OK"]
    if not ok:
        return 0.0, 1.0, {
            "method": "default_no_data",
            "fallback_used": True,
            "num_points": 0,
            "fit_mape_pct": 0.0,
            "max_abs_error_gb": 0.0,
            "max_underpredict_gb": 0.0,
            "reasons": ["No successful probes available for calibration."],
        }

    raws = [_raw_activation_gb(arch, r.group_size, r.seq_length) for r in ok]
    measured = [r.peak_memory_mb / 1024 - base_memory_gb for r in ok]

    if len(ok) == 1:
        # One data point: fix scale=1.0, solve for offset
        offset = max(0.0, measured[0] - raws[0])
        mape_pct, max_abs_err, max_under = _fit_errors(offset, 1.0)
        return offset, 1.0, {
            "method": "single_point",
            "fallback_used": True,
            "num_points": 1,
            "fit_mape_pct": mape_pct,
            "max_abs_error_gb": max_abs_err,
            "max_underpredict_gb": max_under,
            "reasons": ["Only one successful probe; using conservative single-point fit."],
        }

    # Least-squares fit: measured_i = offset + scale * raw_i
    # This is standard linear regression: y = a + b*x
    n = len(ok)
    sum_x = sum(raws)
    sum_y = sum(measured)
    sum_xy = sum(x * y for x, y in zip(raws, measured))
    sum_xx = sum(x * x for x in raws)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        # Degenerate case: all same raw value
        avg_measured = sum_y / n
        offset = max(0.0, avg_measured - raws[0])
        mape_pct, max_abs_err, max_under = _fit_errors(offset, 1.0)
        return offset, 1.0, {
            "method": "degenerate_single_scale",
            "fallback_used": True,
            "num_points": n,
            "fit_mape_pct": mape_pct,
            "max_abs_error_gb": max_abs_err,
            "max_underpredict_gb": max_under,
            "reasons": ["Degenerate regression (insufficient x-variance); using scale=1."],
        }

    scale = (n * sum_xy - sum_x * sum_y) / denom
    offset = (sum_y - scale * sum_x) / n

    # Clamp to physically meaningful range
    scale = max(0.1, scale)
    offset = max(0.0, offset)

    mape_pct, max_abs_err, max_under = _fit_errors(offset, scale)
    fallback_reasons: List[str] = []
    if n < 3:
        fallback_reasons.append("Fewer than 3 successful probes.")
    if mape_pct > 15.0:
        fallback_reasons.append(f"High fit error ({mape_pct:.1f}% MAPE).")
    if max_under > 0.5:
        fallback_reasons.append(
            f"Model underpredicts observed memory by up to {max_under:.2f} GB."
        )

    if fallback_reasons:
        # Conservative fallback:
        # 1) keep slope at least 1.0 to avoid shallow extrapolation
        # 2) raise offset so predictions are >= all observed points
        # 3) add extra safety margin for out-of-sample uncertainty
        conservative_scale = max(1.0, scale)
        conservative_offset = max(
            0.0,
            max(m - conservative_scale * x for x, m in zip(raws, measured)),
        )
        margin_gb = max(0.25, 0.05 * max(0.0, max(measured)))
        conservative_offset += margin_gb
        cons_mape, cons_abs, cons_under = _fit_errors(
            conservative_offset, conservative_scale
        )
        return conservative_offset, conservative_scale, {
            "method": "conservative_fallback",
            "fallback_used": True,
            "num_points": n,
            "fit_mape_pct": cons_mape,
            "max_abs_error_gb": cons_abs,
            "max_underpredict_gb": cons_under,
            "base_fit_mape_pct": mape_pct,
            "base_max_underpredict_gb": max_under,
            "reasons": fallback_reasons,
        }

    return offset, scale, {
        "method": "least_squares",
        "fallback_used": False,
        "num_points": n,
        "fit_mape_pct": mape_pct,
        "max_abs_error_gb": max_abs_err,
        "max_underpredict_gb": max_under,
        "reasons": [],
    }


def _build_empirical_oom_boundaries(results: List[ProbeResult]) -> Dict[int, int]:
    """Return earliest OOM seq_length per observed group size."""
    boundaries: Dict[int, int] = {}
    for r in results:
        if r.status != "OOM":
            continue
        prev = boundaries.get(r.group_size)
        if prev is None or r.seq_length < prev:
            boundaries[r.group_size] = r.seq_length
    return boundaries


def _empirical_seq_cap_for_group(
    group_size: int, oom_boundaries: Dict[int, int]
) -> Optional[int]:
    """Infer a safe seq cap for a target group size from observed OOM boundaries.

    If we observed OOM at group size G0 and target G >= G0, enforce seq_len < OOM_G0.
    """
    cap: Optional[int] = None
    for observed_g, observed_oom_seq in oom_boundaries.items():
        if group_size < observed_g:
            continue
        candidate = max(1, observed_oom_seq - 1)
        cap = candidate if cap is None else min(cap, candidate)
    return cap


@dataclass
class OptimalConfig:
    """A memory-constrained optimal configuration."""

    label: str
    group_size: int
    max_seq_len: int
    predicted_peak_gb: float
    estimated_step_time_s: float
    estimated_run_hours: float
    strategy: str  # "quality", "balanced", "time"


def find_optimal_configs(
    arch: ModelArchitecture,
    base_memory_gb: float,
    memory_budget_gb: float,
    calibration_offset: float,
    calibration_scale: float,
    a: Optional[float],
    b: Optional[float],
    reference_steps: int,
    oom_boundaries: Optional[Dict[int, int]] = None,
) -> List[OptimalConfig]:
    """Solve for optimal (G, seq_len) configs under the memory constraint.

    Tests three strategies:
    - Quality (G=16): more GRPO comparisons per group → better advantages
    - Balanced (G=8): the GRPO default
    - Time-optimal (G=4): faster steps, can fit longer sequences
    """
    configs: List[OptimalConfig] = []

    for label, G, strategy in [
        ("Quality-optimal (G=16)", 16, "quality"),
        ("Balanced (G=8)", 8, "balanced"),
        ("Time-optimal (G=4)", 4, "time"),
    ]:
        # Binary search for max seq_len fitting memory budget
        lo, hi = 64, 16384
        max_seq = 0
        if oom_boundaries:
            cap = _empirical_seq_cap_for_group(G, oom_boundaries)
            if cap is not None:
                hi = min(hi, cap)
        while lo <= hi:
            mid = (lo + hi) // 2
            pred = predict_peak_memory_gb(
                arch, base_memory_gb, G, mid, calibration_offset, calibration_scale
            )
            if pred <= memory_budget_gb:
                max_seq = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if max_seq > 0:
            pred_peak = predict_peak_memory_gb(
                arch, base_memory_gb, G, max_seq, calibration_offset, calibration_scale
            )
        else:
            pred_peak = base_memory_gb + calibration_offset

        # Extrapolate step time from empirical scaling fit.
        # Generation scales ~linearly with G; training scales with seq².
        step_time = 0.0
        if max_seq > 0 and a is not None and b is not None:
            train_time = extrapolate_time(a, b, max_seq)
            # Generation time scales roughly linearly with G and seq_len
            gen_time_est = 0.06 * G * max_seq / 256  # rough baseline
            step_time = gen_time_est + train_time

        run_hours = (step_time * reference_steps) / 3600 if step_time > 0 else 0.0

        configs.append(
            OptimalConfig(
                label=label,
                group_size=G,
                max_seq_len=max_seq,
                predicted_peak_gb=pred_peak,
                estimated_step_time_s=step_time,
                estimated_run_hours=run_hours,
                strategy=strategy,
            )
        )

    return configs


# ── Recommendation engine ───────────────────────────────────────────────────


@dataclass
class Recommendation:
    sweet_spot_seq_length: int
    sweet_spot_group_size: int
    estimated_run_time_hours: float
    max_feasible_seq_length: int
    notes: List[str]
    optimization_targets: List[str]  # Where to focus effort


def build_recommendation(
    results: List[ProbeResult],
    total_memory_gb: float,
    reference_steps: int,
    bottleneck_analyses: List[BottleneckAnalysis],
) -> Recommendation:
    """Recommend the best training configuration based on probe results."""
    ok_results = [r for r in results if r.status == "OK"]

    if not ok_results:
        return Recommendation(
            sweet_spot_seq_length=0,
            sweet_spot_group_size=8,
            estimated_run_time_hours=0.0,
            max_feasible_seq_length=0,
            notes=["No successful probes. Try a smaller model or reduce group_size."],
            optimization_targets=[],
        )

    max_feasible = max(r.seq_length for r in ok_results)

    # Sweet spot: largest seq_length where total_time_s < 3x the shortest
    min_total = min(r.total_time_s for r in ok_results)
    sweet_spot_candidates = [
        r for r in ok_results if r.total_time_s < 3 * min_total
    ]
    sweet_spot = max(sweet_spot_candidates, key=lambda r: r.seq_length)

    est_hours = (sweet_spot.total_time_s * reference_steps) / 3600

    notes: List[str] = []
    optimization_targets: List[str] = []

    # Check if gradient checkpointing might help
    failed = [r for r in results if r.status in ("OOM", "ERROR")]
    if failed:
        next_seq = min(r.seq_length for r in failed)
        notes.append(
            f"To unlock {next_seq}+ tokens: try gradient_checkpointing=True "
            f"and micro_batch_size=4 (GC+M=4 gives -33.7% peak memory at "
            f"seq=1024)"
        )

    # Check if peak memory exceeds or approaches physical RAM
    memory_mb = total_memory_gb * 1024
    if ok_results:
        highest_ok = max(ok_results, key=lambda r: r.seq_length)
        if highest_ok.peak_memory_mb > memory_mb:
            notes.append(
                f"WARNING: Peak memory {highest_ok.peak_memory_mb / 1024:.1f} GB exceeds "
                f"physical RAM ({total_memory_gb:.0f} GB) at seq_length="
                f"{highest_ok.seq_length}. Swapping to disk — expect severe slowdown."
            )
        elif highest_ok.peak_memory_mb > memory_mb * 0.8:
            notes.append(
                f"Running close to memory limit "
                f"({highest_ok.peak_memory_mb:.0f} MB / {memory_mb:.0f} MB)"
            )
            if highest_ok.peak_memory_mb > memory_mb * 0.7:
                notes.append(
                    "Consider micro_batch_size=4 to reduce peak memory, or "
                    "combine with gradient_checkpointing=True for best savings"
                )

    if max_feasible < 4096:
        notes.append(
            "To unlock 4096+ tokens: more memory or smaller model needed"
        )

    # Derive optimization targets from bottleneck analysis
    if bottleneck_analyses:
        # Use the sweet spot's analysis, or the largest feasible
        sweet_analysis = None
        for ba in bottleneck_analyses:
            if ba.seq_length == sweet_spot.seq_length:
                sweet_analysis = ba
                break
        if sweet_analysis is None:
            sweet_analysis = bottleneck_analyses[-1]

        bn = sweet_analysis.bottleneck_phase
        bf = sweet_analysis.bottleneck_fraction
        bl = sweet_analysis.amdahl_limit

        if bf > 0.5:
            optimization_targets.append(
                f"Primary bottleneck: {bn} ({bf:.0%} of step time, "
                f"Amdahl limit {bl:.1f}x)"
            )

        if sweet_analysis.regime == "compute-bound":
            optimization_targets.append(
                "Train phase dominates: consider mx.compile, "
                "gradient_checkpointing=True, micro_batch_size=4, "
                "or quantized training"
            )
        elif sweet_analysis.regime == "memory-bandwidth-bound":
            optimization_targets.append(
                "Generation phase dominates: consider KV-cache reuse, "
                "prompt caching, or batched generation"
            )

        # Check if loss_and_grad dominates trainer time
        for phase_name, secs, frac, limit in sweet_analysis.amdahl_phases:
            if "loss_and_grad" in phase_name and frac > 0.3:
                optimization_targets.append(
                    f"Forward+backward pass is {frac:.0%} of step: "
                    f"mx.compile or reduced precision would help most"
                )
                break

    return Recommendation(
        sweet_spot_seq_length=sweet_spot.seq_length,
        sweet_spot_group_size=sweet_spot.group_size,
        estimated_run_time_hours=est_hours,
        max_feasible_seq_length=max_feasible,
        notes=notes,
        optimization_targets=optimization_targets,
    )


# ── Output formatting ───────────────────────────────────────────────────────


def format_human_output(
    hw_info: Dict[str, Any],
    model_id: str,
    model_memory_mb: float,
    lora_memory_mb: float,
    total_memory_gb: float,
    results: List[ProbeResult],
    a: Optional[float],
    b: Optional[float],
    recommendation: Recommendation,
    reference_steps: int,
    bottleneck_analyses: List[BottleneckAnalysis],
    steps_per_probe: int = 2,
    optimal_configs: Optional[List[OptimalConfig]] = None,
    arch: Optional[ModelArchitecture] = None,
    calibration_offset: Optional[float] = None,
    calibration_scale: Optional[float] = None,
    calibration_diagnostics: Optional[Dict[str, Any]] = None,
    oom_boundaries: Optional[Dict[int, int]] = None,
) -> str:
    """Format the profiling results as a human-readable report."""
    lines: List[str] = []

    # Header
    lines.append("")
    lines.append("=" * 60)
    lines.append("  TEXTPOLICY HARDWARE PROFILE")
    chip = hw_info.get("chip", "Unknown")
    mem = hw_info.get("total_memory_gb", total_memory_gb)
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
    lines.append(f"  Chip: {chip}  |  Memory: {mem:.0f} GB  |  Model: {model_short}")
    lines.append("=" * 60)

    # Model memory footprint
    lines.append("")
    lines.append("  Model Memory Footprint")
    lines.append("  " + "-" * 22)
    lines.append(f"  Base model:       ~{model_memory_mb / 1024:.1f} GB")
    lines.append(f"  + LoRA overhead:   +{lora_memory_mb / 1024:.2f} GB")
    avail = total_memory_gb - model_memory_mb / 1024
    lines.append(f"  Available for activations: ~{max(0, avail):.1f} GB")

    # Scaling results table
    lines.append("")
    ok_results = [r for r in results if r.status == "OK"]
    baseline_train = ok_results[0].train_time_s if ok_results else 1.0
    group_size = results[0].group_size if results else 8

    lines.append(f"  Scaling Results (G={group_size}, {steps_per_probe} steps each)")
    lines.append("  " + "-" * 40)
    header = (
        f"  {'Seq Len':>8} | {'Gen/step':>9} | {'Train/step':>10} | "
        f"{'Total/step':>10} | {'Peak Mem':>9} | {'Scaling':>8} | {'Status':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * len(header.strip()))

    for r in results:
        if r.status == "OK":
            scaling = (
                f"{r.train_time_s / baseline_train:.1f}x"
                if baseline_train > 0
                else "-"
            )
            row = (
                f"  {r.seq_length:>8} | {r.gen_time_s:>8.1f}s | {r.train_time_s:>9.1f}s | "
                f"{r.total_time_s:>9.1f}s | {r.peak_memory_mb / 1024:>7.1f} GB | "
                f"{scaling:>8} | {'OK':>7}"
            )
        else:
            row = (
                f"  {r.seq_length:>8} | {'--':>9} | {'--':>10} | "
                f"{'--':>10} | {'--':>9} | {'--':>8} | {r.status:>7}"
            )
        lines.append(row)

    # Scaling exponent
    if b is not None:
        lines.append("")
        lines.append(f"  Observed training scaling exponent: n^{b:.2f}")
        if b > 1.5:
            lines.append("  (super-linear -- attention-dominated)")
        elif b > 0.8:
            lines.append("  (roughly linear -- memory-bandwidth-dominated)")
        else:
            lines.append("  (sub-linear)")

    # ── Bottleneck Analysis (Amdahl's Law) ───────────────────────────
    if bottleneck_analyses:
        lines.append("")
        lines.append("  " + "=" * 56)
        lines.append("  BOTTLENECK ANALYSIS (Amdahl's Law)")
        lines.append("  " + "=" * 56)
        lines.append("")
        lines.append(
            "  Amdahl's Law: if phase X takes fraction f of total time,")
        lines.append(
            "  then even making X infinitely fast gives at most 1/(1-f) speedup.")
        lines.append("")

        for ba in bottleneck_analyses:
            lines.append(f"  seq_length={ba.seq_length}  [{ba.regime}]")
            lines.append("  " + "-" * 50)
            for name, secs, frac, limit in ba.amdahl_phases:
                if frac < 0.01:
                    continue  # skip trivial phases
                limit_str = f"{limit:.1f}x" if limit < 100 else ">>10x"
                bar_len = int(frac * 30)
                bar = "#" * bar_len + "." * (30 - bar_len)
                lines.append(
                    f"    {name:<24s} {secs:>6.2f}s  "
                    f"{frac:>5.1%}  [{bar}]  limit {limit_str}"
                )
            lines.append("")

        # Show how bottleneck shifts across sequence lengths
        if len(bottleneck_analyses) >= 2:
            lines.append("  Bottleneck shift across sequence lengths:")
            lines.append("  " + "-" * 50)
            for ba in bottleneck_analyses:
                lines.append(
                    f"    {ba.seq_length:>6} tokens: {ba.bottleneck_phase:<24s} "
                    f"({ba.bottleneck_fraction:.0%})"
                )
            lines.append("")

    # ── Throughput Analysis (Little's Law) ────────────────────────────
    if bottleneck_analyses:
        lines.append("  " + "=" * 56)
        lines.append("  THROUGHPUT ANALYSIS (Little's Law)")
        lines.append("  " + "=" * 56)
        lines.append("")
        lines.append(
            "  Little's Law: L = lambda * W  (items in system = throughput * latency)")
        lines.append(
            "  Dropping tok/s as seq_length grows = GPU stalling on memory/compute.")
        lines.append("")

        header_tp = (
            f"  {'Seq Len':>8} | {'Gen tok/s':>10} | {'Train tok/s':>11} | "
            f"{'Total tokens':>13} | {'Regime':>20}"
        )
        lines.append(header_tp)
        lines.append("  " + "-" * len(header_tp.strip()))

        for ba in bottleneck_analyses:
            lines.append(
                f"  {ba.seq_length:>8} | {ba.tokens_per_second_gen:>9.0f} | "
                f"{ba.tokens_per_second_train:>10.0f} | "
                f"{ba.tokens_per_second_gen * 1:>13.0f} | "  # placeholder
                f"{ba.regime:>20}"
            )
        # replace the placeholder with actual total tokens
        # rebuild those lines properly
        lines_to_fix = len(bottleneck_analyses)
        for i, ba in enumerate(bottleneck_analyses):
            r_match = [r for r in results if r.seq_length == ba.seq_length and r.status == "OK"]
            total_tok = r_match[0].total_tokens_generated if r_match else 0
            idx = len(lines) - lines_to_fix + i
            lines[idx] = (
                f"  {ba.seq_length:>8} | {ba.tokens_per_second_gen:>9.0f} | "
                f"{ba.tokens_per_second_train:>10.0f} | "
                f"{total_tok:>13} | "
                f"{ba.regime:>20}"
            )

        # Throughput scaling insight
        if len(bottleneck_analyses) >= 2:
            first_tps = bottleneck_analyses[0].tokens_per_second_train
            last_tps = bottleneck_analyses[-1].tokens_per_second_train
            if first_tps > 0 and last_tps > 0:
                seq_ratio = (
                    bottleneck_analyses[-1].seq_length
                    / bottleneck_analyses[0].seq_length
                )
                tps_ratio = last_tps / first_tps
                lines.append("")
                lines.append(
                    f"  Seq length grew {seq_ratio:.0f}x, "
                    f"training throughput changed to {tps_ratio:.2f}x"
                )
                if tps_ratio < 0.5:
                    lines.append(
                        "  --> Severe throughput collapse: "
                        "compute cost growing faster than data size"
                    )
                elif tps_ratio < 0.9:
                    lines.append(
                        "  --> Moderate throughput loss: scaling is super-linear"
                    )
                else:
                    lines.append(
                        "  --> Throughput maintained: scaling is near-linear"
                    )

        lines.append("")

    # Extrapolation table
    if a is not None and b is not None:
        lines.append("  Extrapolated Estimates (based on observed scaling)")
        lines.append("  " + "-" * 50)
        header2 = (
            f"  {'Target':>16} | {'Seq Len':>8} | {'Train/step':>10} | "
            f"{f'{reference_steps}-step run':>14} | {'Feasible?':>10}"
        )
        lines.append(header2)
        lines.append("  " + "-" * len(header2.strip()))

        targets = [
            ("Countdown PoC", 256),
            ("Mid-scale GTPO", 1024),
            ("Paper GTPO", 4096),
            ("Paper HICRA", 16384),
        ]

        for label, seq_len in targets:
            t = extrapolate_time(a, b, seq_len)
            run_hours = (t * reference_steps) / 3600
            mem_est = extrapolate_memory(results, seq_len)
            measured_ok = next((r for r in ok_results if r.seq_length == seq_len), None)
            if measured_ok is not None:
                t = measured_ok.train_time_s
                run_hours = (t * reference_steps) / 3600

            if measured_ok is not None:
                # Exact successful measurement at this target is authoritative.
                feasible = "YES (measured)"
            else:
                feasible = "YES"
                failed_at = [
                    r for r in results if r.seq_length <= seq_len and r.status != "OK"
                ]
                if failed_at:
                    feasible = "NO"
                elif mem_est is not None and mem_est > total_memory_gb * 1024:
                    feasible = f"NO (est. >{total_memory_gb:.0f}GB)"
                elif run_hours > 48:
                    feasible = "NO (>48h)"

            if run_hours < 1:
                time_str = f"{run_hours * 60:.0f} min"
            elif run_hours < 24:
                time_str = f"{run_hours:.1f} hrs"
            else:
                time_str = f"{run_hours / 24:.1f} days"

            row = (
                f"  {label:>16} | {seq_len:>8} | {t:>9.1f}s | "
                f"{time_str:>14} | {feasible:>10}"
            )
            lines.append(row)

    # ── Analytical Memory Model ──────────────────────────────────
    cal_off = calibration_offset or 0.0
    cal_sc = calibration_scale or 1.0
    if arch is not None and calibration_scale is not None:
        lines.append("  " + "=" * 56)
        lines.append("  ANALYTICAL MEMORY MODEL")
        lines.append("  " + "=" * 56)
        lines.append("")
        lines.append(
            f"  Architecture: {arch.num_layers} layers "
            f"({arch.num_full_attn_layers} full attn + "
            f"{arch.num_sliding_attn_layers} sliding, window={arch.sliding_window})"
        )
        lines.append(
            f"  Attention: {arch.num_attention_heads} heads, "
            f"{arch.num_kv_heads} KV heads (GQA "
            f"{arch.num_attention_heads // max(1, arch.num_kv_heads)}:1), "
            f"head_dim={arch.head_dim}"
        )
        if arch.num_experts_per_tok > 0:
            lines.append(
                f"  MoE: top-{arch.num_experts_per_tok}, "
                f"expert FFN dim={arch.moe_intermediate_size}"
            )
        lines.append(
            f"  Calibration: offset={cal_off:.2f} GB, scale={cal_sc:.3f}"
        )
        if calibration_diagnostics:
            method = str(calibration_diagnostics.get("method", "unknown"))
            fit_mape_pct = calibration_diagnostics.get("fit_mape_pct")
            lines.append(f"  Calibration mode: {method}")
            if isinstance(fit_mape_pct, (int, float)):
                lines.append(f"  Fit error (MAPE): {fit_mape_pct:.1f}%")
            reasons = calibration_diagnostics.get("reasons", [])
            if reasons:
                lines.append("  Guardrail triggers:")
                for reason in reasons:
                    lines.append(f"    - {reason}")
        if oom_boundaries:
            lines.append("  Empirical OOM boundaries:")
            for g, seq in sorted(oom_boundaries.items()):
                lines.append(f"    - G={g}: observed OOM at seq_length={seq}")
        lines.append("")

        # Predicted vs measured comparison
        if ok_results:
            lines.append("  Predicted vs Measured Memory:")
            lines.append("  " + "-" * 50)
            pred_header = (
                f"  {'Seq Len':>8} | {'Predicted':>10} | {'Measured':>10} | {'Error':>8}"
            )
            lines.append(pred_header)
            lines.append("  " + "-" * len(pred_header.strip()))
            for r in ok_results:
                pred = predict_peak_memory_gb(
                    arch,
                    model_memory_mb / 1024,
                    r.group_size,
                    r.seq_length,
                    cal_off,
                    cal_sc,
                )
                meas = r.peak_memory_mb / 1024
                err = ((pred - meas) / meas * 100) if meas > 0 else 0
                lines.append(
                    f"  {r.seq_length:>8} | {pred:>8.1f} GB | {meas:>8.1f} GB | "
                    f"{err:>+6.1f}%"
                )
            lines.append("")

        # Optimal configs
        if optimal_configs:
            lines.append("  Memory-Constrained Optimal Configs:")
            lines.append(
                f"  (budget: {total_memory_gb * 0.9:.0f} GB = "
                f"90% of {total_memory_gb:.0f} GB physical RAM)"
            )
            lines.append("  " + "-" * 50)
            cfg_header = (
                f"  {'Strategy':>25} | {'G':>3} | {'Max Seq':>8} | "
                f"{'Peak Mem':>9} | {'Est. Run':>12}"
            )
            lines.append(cfg_header)
            lines.append("  " + "-" * len(cfg_header.strip()))
            for oc in optimal_configs:
                if oc.estimated_run_hours > 0:
                    if oc.estimated_run_hours < 1:
                        run_str = f"{oc.estimated_run_hours * 60:.0f} min"
                    else:
                        run_str = f"{oc.estimated_run_hours:.1f} hrs"
                else:
                    run_str = "--"
                lines.append(
                    f"  {oc.label:>25} | {oc.group_size:>3} | "
                    f"{oc.max_seq_len:>6}  | "
                    f"{oc.predicted_peak_gb:>7.1f} GB | "
                    f"{run_str:>12}"
                )
            lines.append("")

    # Recommendation
    lines.append("")
    lines.append("=" * 60)
    lines.append("  RECOMMENDATION")
    lines.append("=" * 60)

    if recommendation.sweet_spot_seq_length > 0:
        lines.append(
            f"  Your sweet spot: {recommendation.sweet_spot_seq_length} tokens, "
            f"G={recommendation.sweet_spot_group_size}"
        )
        if recommendation.estimated_run_time_hours < 1:
            time_est = f"~{recommendation.estimated_run_time_hours * 60:.0f} minutes"
        else:
            time_est = f"~{recommendation.estimated_run_time_hours:.1f} hours"
        lines.append(
            f"  Estimated training time ({reference_steps} steps): {time_est}"
        )
    else:
        lines.append("  No feasible configuration found.")

    lines.append("")
    for note in recommendation.notes:
        lines.append(f"  {note}")

    if recommendation.optimization_targets:
        lines.append("")
        lines.append("  WHERE TO FOCUS OPTIMIZATION:")
        for target in recommendation.optimization_targets:
            lines.append(f"    -> {target}")

    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def build_json_output(
    hw_info: Dict[str, Any],
    model_id: str,
    model_memory_mb: float,
    lora_memory_mb: float,
    results: List[ProbeResult],
    a: Optional[float],
    b: Optional[float],
    recommendation: Recommendation,
    reference_steps: int,
    bottleneck_analyses: List[BottleneckAnalysis],
    optimal_configs: Optional[List[OptimalConfig]] = None,
    arch: Optional[ModelArchitecture] = None,
    calibration_offset: Optional[float] = None,
    calibration_scale: Optional[float] = None,
    calibration_diagnostics: Optional[Dict[str, Any]] = None,
    oom_boundaries: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """Build the machine-readable JSON output."""
    out: Dict[str, Any] = {
        "hardware": hw_info,
        "model": model_id,
        "model_memory_mb": model_memory_mb,
        "lora_overhead_mb": lora_memory_mb,
        "scaling_exponent": {"a": a, "b": b},
        "probes": [asdict(r) for r in results],
        "bottleneck_analyses": [asdict(ba) for ba in bottleneck_analyses],
        "recommendation": asdict(recommendation),
        "reference_steps": reference_steps,
    }
    if arch is not None:
        out["architecture"] = asdict(arch)
    if calibration_offset is not None or calibration_scale is not None:
        out["calibration"] = {
            "offset_gb": calibration_offset,
            "scale": calibration_scale,
        }
    if calibration_diagnostics is not None:
        out["calibration_diagnostics"] = calibration_diagnostics
    if oom_boundaries:
        out["empirical_oom_boundaries"] = oom_boundaries
    if optimal_configs:
        out["optimal_configs"] = [asdict(oc) for oc in optimal_configs]
    # gradient_checkpointing is surfaced via ProfileConfig → run_profile header
    return out


# ── Main pipeline ────────────────────────────────────────────────────────────


def run_profile(config: ProfileConfig) -> None:
    """Execute the full profiling pipeline."""

    # 1. Hardware detection
    print("Detecting hardware...")
    hw_info = detect_hardware()
    total_memory_gb = hw_info.get("total_memory_gb", 0.0)
    print(f"  Chip: {hw_info['chip']}")
    print(f"  Memory: {total_memory_gb:.0f} GB")
    print(f"  OS: {hw_info['os']}")

    # 2. Model loading + baseline memory
    print(f"\nLoading model: {config.model_id}")
    mem_before = get_memory_stats()
    base_model, tokenizer = load_model(config.model_id)

    optimizer = optim.Adam(learning_rate=5e-6)
    trainer, memory_stats = create_tinylora_reasoning_setup(
        model=base_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lora_config={
            "lora_layers": 4,
            "lora_rank": 2,
            "lora_scale": 8.0,
            "lora_dropout": 0.0,
        },
        # Profiling should measure compute/memory behavior, not disk writes.
        # Disable auto-reload to avoid per-step adapter save I/O.
        auto_reload=False,
        compile_training=False,
        gradient_checkpointing=config.gradient_checkpointing,
        micro_batch_size=config.micro_batch_size,
        profile=True,
        max_grad_norm=0.5,
    )
    model = trainer.model

    mem_after = get_memory_stats()
    mem_before_mlx = mem_before.get("mlx_memory_mb", 0.0)
    mem_after_mlx = mem_after.get("mlx_memory_mb", 0.0)
    model_memory_mb = max(0.0, mem_after_mlx - mem_before_mlx)
    lora_overhead = memory_stats.get("trainable_params", 0) * 4 / 1024 / 1024

    print(f"  Model memory: ~{model_memory_mb / 1024:.1f} GB")
    print(
        f"  Memory savings from LoRA: "
        f"{memory_stats.get('memory_savings_percent', 0.0):.1f}%"
    )
    if config.gradient_checkpointing:
        print("  Gradient checkpointing: ENABLED")
    if config.micro_batch_size is not None:
        print(f"  Micro-batch size: {config.micro_batch_size}")

    # 3. Prepare dataset
    print(f"\nGenerating countdown problems (seed={config.dataset_seed})...")
    problems = generate_countdown_problems(config.num_problems, seed=config.dataset_seed)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    # 4. Scaling probe loop
    seq_lengths = config.seq_lengths
    if config.quick:
        seq_lengths = [s for s in seq_lengths if s <= 512]
        if not seq_lengths:
            seq_lengths = config.seq_lengths[:2]

    print(f"\nRunning scaling probes: {seq_lengths}")
    print(f"  Group size: G={config.group_size}")
    print(f"  Steps per probe: {config.steps_per_probe}")
    print()

    results: List[ProbeResult] = []
    stop_early = False

    for seq_len in seq_lengths:
        if stop_early:
            results.append(
                ProbeResult(
                    seq_length=seq_len,
                    group_size=config.group_size,
                    gen_time_s=0.0,
                    train_time_s=0.0,
                    total_time_s=0.0,
                    peak_memory_mb=0.0,
                    status="SKIPPED",
                    error_msg="Skipped due to earlier failure",
                )
            )
            continue

        print(f"  Probing seq_length={seq_len}...", end=" ", flush=True)
        probe_start = time.perf_counter()

        result = run_probe(
            seq_length=seq_len,
            group_size=config.group_size,
            steps=config.steps_per_probe,
            model=model,
            tokenizer=tokenizer,
            trainer=trainer,
            problems=problems,
            prompts=prompts,
            timeout=config.timeout_per_step,
        )
        results.append(result)

        elapsed = time.perf_counter() - probe_start
        if result.status == "OK":
            print(
                f"OK  gen={result.gen_time_s:.1f}s  train={result.train_time_s:.1f}s  "
                f"peak={result.peak_memory_mb / 1024:.1f}GB  ({elapsed:.0f}s total)"
            )
        else:
            print(f"{result.status}: {result.error_msg or 'unknown'}")
            if result.status == "OOM":
                stop_early = True

    # 5. Analysis
    a, b = fit_scaling_exponent(results)
    bottleneck_analyses = analyze_bottlenecks(results)

    # 5b. Analytical memory model
    print("\nBuilding analytical memory model...")
    arch = extract_architecture(model)
    base_memory_gb = model_memory_mb / 1024
    cal_offset, cal_scale, cal_diag = calibrate_analytical_model(
        arch, base_memory_gb, results
    )
    print(
        f"  Calibration: offset={cal_offset:.2f} GB, scale={cal_scale:.3f} "
        f"({cal_diag.get('method', 'unknown')})"
    )
    fit_mape_pct = cal_diag.get("fit_mape_pct")
    if isinstance(fit_mape_pct, (int, float)):
        print(f"  Fit error (MAPE): {fit_mape_pct:.1f}%")
    reasons = cal_diag.get("reasons", [])
    if reasons:
        print("  Guardrail triggers:")
        for reason in reasons:
            print(f"    - {reason}")

    # Use 90% of physical RAM as budget (leave room for OS + other processes)
    memory_budget_gb = total_memory_gb * 0.9
    oom_boundaries = _build_empirical_oom_boundaries(results)
    if oom_boundaries:
        print("  Applying empirical OOM boundaries:")
        for g, seq in sorted(oom_boundaries.items()):
            print(f"    - G={g}: seq_length < {seq}")
    optimal_configs = find_optimal_configs(
        arch, base_memory_gb, memory_budget_gb, cal_offset, cal_scale,
        a, b, config.reference_run_steps, oom_boundaries=oom_boundaries,
    )
    for oc in optimal_configs:
        print(f"  {oc.label}: max {oc.max_seq_len} tokens")

    # 6. Recommendation
    recommendation = build_recommendation(
        results, total_memory_gb, config.reference_run_steps, bottleneck_analyses
    )

    # 7. Output
    if config.json_output:
        output = build_json_output(
            hw_info=hw_info,
            model_id=config.model_id,
            model_memory_mb=model_memory_mb,
            lora_memory_mb=lora_overhead,
            results=results,
            a=a,
            b=b,
            recommendation=recommendation,
            reference_steps=config.reference_run_steps,
            bottleneck_analyses=bottleneck_analyses,
            optimal_configs=optimal_configs,
            arch=arch,
            calibration_offset=cal_offset,
            calibration_scale=cal_scale,
            calibration_diagnostics=cal_diag,
            oom_boundaries=oom_boundaries,
        )
        json_str = json.dumps(output, indent=2)

        if config.output_path:
            out_path = Path(config.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str)
            print(f"\nJSON results saved to {out_path}")
        else:
            print(json_str)
    else:
        report = format_human_output(
            hw_info=hw_info,
            model_id=config.model_id,
            model_memory_mb=model_memory_mb,
            lora_memory_mb=lora_overhead,
            total_memory_gb=total_memory_gb,
            results=results,
            a=a,
            b=b,
            recommendation=recommendation,
            reference_steps=config.reference_run_steps,
            bottleneck_analyses=bottleneck_analyses,
            steps_per_probe=config.steps_per_probe,
            optimal_configs=optimal_configs,
            arch=arch,
            calibration_offset=cal_offset,
            calibration_scale=cal_scale,
            calibration_diagnostics=cal_diag,
            oom_boundaries=oom_boundaries,
        )
        print(report)

        if config.output_path:
            out_path = Path(config.output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report)
            print(f"Report saved to {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile your hardware for TextPolicy training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python experiments/profile_hardware.py
  uv run python experiments/profile_hardware.py --quick
  uv run python experiments/profile_hardware.py --json --output results/hw.json
  uv run python experiments/profile_hardware.py --max-seq-lengths 256 512 1024 2048 4096
        """,
    )
    parser.add_argument(
        "--model",
        default="arcee-ai/Trinity-Nano-Preview",
        help="Model ID to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--max-seq-lengths",
        nargs="+",
        type=int,
        default=[256, 512, 1024, 2048],
        help="Sequence lengths to test (default: 256 512 1024 2048)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Group size / episodes per step (default: %(default)s)",
    )
    parser.add_argument(
        "--steps-per-probe",
        type=int,
        default=2,
        help="Training steps per probe point (default: %(default)s)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only test 256 and 512 tokens",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to file",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=210,
        help="Reference training run length for time estimates (default: %(default)s)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable gradient checkpointing — recomputes activations during "
            "backward pass instead of caching them. ~20-30%% extra compute, "
            "but significant memory savings. Best combined with "
            "--micro-batch-size (GC+M=4 gives -33.7%% peak memory at "
            "seq=1024)"
        ),
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Split episodes into groups of N for forward/backward, "
            "accumulating gradients. Reduces peak activation memory roughly "
            "by factor N. Recommended starting point: 4 (gives -37.5%% peak "
            "memory, -25%% step time at seq=1024). Combine with "
            "--gradient-checkpointing for best results"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout per probe step in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    cfg = ProfileConfig(
        model_id=args.model,
        seq_lengths=sorted(args.max_seq_lengths),
        group_size=args.group_size,
        steps_per_probe=args.steps_per_probe,
        gradient_checkpointing=args.gradient_checkpointing,
        micro_batch_size=args.micro_batch_size,
        quick=args.quick,
        json_output=args.json_output,
        output_path=args.output,
        reference_run_steps=args.reference_steps,
        timeout_per_step=args.timeout,
    )

    run_profile(cfg)
