#!/usr/bin/env python3
"""
LoRA reasoning run on the Countdown task (GTPO + HICRA).

This experiment composes:
- TinyLoRA-style PEFT defaults
- GTPO entropy-weighted token credit
- HICRA planning-token amplification

Usage:
    uv run python experiments/countdown_reasoning_lora.py --steps 500
    uv run python experiments/countdown_reasoning_lora.py --model arcee-ai/Trinity-Nano-Preview --steps 10 --output results/test_reasoning
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

import mlx.core as mx
import mlx.optimizers as optim

from textpolicy.algorithms import grpo
from textpolicy.analysis import EmergenceLogger, load_strategic_grams
from textpolicy.buffer import Buffer
from textpolicy.environment.text_generation import TextGenerationEnv
from textpolicy.generation.mlx_generation import create_policy, load_model
from textpolicy.generation.reload import save_adapters
from textpolicy.rollout import RolloutCoordinator
from textpolicy.tasks.countdown import (
    countdown_reward,
    format_countdown_prompt,
    generate_countdown_problems,
)
from textpolicy.algorithms.grpo import compute_advantages, policy_loss
from textpolicy.generation.lora import create_lora_setup
from textpolicy.training import Trainer, build_gtpo_transform


@dataclass
class ReasoningConfig:
    """Configuration for LoRA reasoning experiment."""

    # Model
    model_id: str = "arcee-ai/Trinity-Nano-Preview"

    # TinyLoRA-style defaults
    lora_layers: int = 4
    lora_rank: int = 2
    lora_scale: float = 8.0
    lora_dropout: float = 0.0

    # Reasoning shaping
    strategic_grams_path: Optional[str] = None

    # GTPO (arXiv 2508.04349 Eq. 3, 5, 6)
    alpha_1: float = 1.0
    alpha_2: float = 0.1
    reward_threshold: float = 0.5
    hicra_gamma: float = 0.3

    # Training
    learning_rate: float = 5e-6
    max_steps: int = 500
    max_grad_norm: float = 0.5
    compile_training: Union[bool, str] = False
    gradient_checkpointing: bool = False
    micro_batch_size: Optional[int] = None
    profile_training: bool = False

    # Generation
    max_completion_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Dataset
    num_problems: int = 50
    dataset_seed: int = 42

    # Rollout
    episodes_per_step: int = 8
    batch_size: int = 8

    # Output
    output_dir: str = "results/countdown_reasoning_lora"

    # Wandb (opt-in: set wandb_project to enable)
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


# Memory optimization quick-reference:
#   --gradient-checkpointing        recompute selected activations to reduce memory
#   --micro-batch-size 4            good first value for lower memory/logit pressure
#   Combine both when memory is still tight; benchmark on your hardware.
#   See docs/06_performance.md for examples.


# RolloutRunner uses max(10, max_steps) for buffer capacity, so it
# dynamically grows with episodes_per_step.  No artificial cap needed.


def save_config(config: ReasoningConfig, output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Config saved to {config_path}")


def save_checkpoint(model, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(checkpoint_dir / "lora_adapters.safetensors")
    save_adapters(model, adapter_path)


def print_summary(output_dir: Path) -> None:
    steps_path = output_dir / "emergence" / "steps.jsonl"
    if not steps_path.exists():
        print("No steps.jsonl found — nothing to summarize.")
        return

    steps = []
    with open(steps_path) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    if not steps:
        print("No step records found.")
        return

    first = steps[0]
    last = steps[-1]
    best_accuracy = 0.0
    best_step = 0
    for s in steps:
        total = s.get("total_count", 0)
        if total > 0:
            acc = s.get("correct_count", 0) / total
            if acc > best_accuracy:
                best_accuracy = acc
                best_step = s["step"]

    print("\n" + "=" * 50)
    print("REASONING EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"  Steps completed:  {last['step'] + 1}")
    print(f"  First step reward: {first['mean_reward']:.3f}")
    print(f"  Final step reward: {last['mean_reward']:.3f}")
    print(f"  Best accuracy:     {best_accuracy:.1%} (step {best_step})")
    print(f"  Final planning ratio: {last['planning_token_ratio']:.4f}")
    print("=" * 50)


def print_amdahl_summary(
    phase_totals: Dict[str, float],
    trainer_phase_totals: Dict[str, float],
    rollout_phase_totals: Optional[Dict[str, float]] = None,
) -> None:
    total = sum(phase_totals.values())
    if total <= 0.0:
        return

    print("\n" + "=" * 50)
    print("AMDAHL BOTTLENECK SUMMARY (END-TO-END)")
    print("=" * 50)
    ranked = sorted(phase_totals.items(), key=lambda kv: kv[1], reverse=True)
    for phase, seconds in ranked:
        fraction = seconds / total
        max_speedup = float("inf") if fraction >= 0.999999 else 1.0 / (1.0 - fraction)
        name = phase.replace("_s", "")
        speedup_str = "inf" if max_speedup == float("inf") else f"{max_speedup:.2f}x"
        print(
            f"  {name:16s} {seconds:8.2f}s  "
            f"({fraction * 100:5.1f}%)  Amdahl limit if perfect: {speedup_str}"
        )

    if rollout_phase_totals:
        rollout_total = rollout_phase_totals.get("total", 0.0)
        if rollout_total > 0.0:
            print("\n  Rollout-internal split (for rollout_collect phase):")
            r_ranked = sorted(
                (
                    (phase, secs)
                    for phase, secs in rollout_phase_totals.items()
                    if phase != "total"
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for phase, seconds in r_ranked:
                pct = (seconds / rollout_total) * 100.0
                print(f"    {phase:16s} {seconds:8.2f}s  ({pct:5.1f}%)")

    if trainer_phase_totals:
        trainer_total = trainer_phase_totals.get("total", 0.0)
        if trainer_total > 0.0:
            print("\n  Trainer-internal split (for train phase):")
            t_ranked = sorted(
                (
                    (phase, secs)
                    for phase, secs in trainer_phase_totals.items()
                    if phase != "total"
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for phase, seconds in t_ranked:
                pct = (seconds / trainer_total) * 100.0
                print(f"    {phase:16s} {seconds:8.2f}s  ({pct:5.1f}%)")
    print("=" * 50)


def init_wandb(config: ReasoningConfig) -> bool:
    """Initialize wandb run if configured. Returns True if wandb is active."""
    if not config.wandb_project or not HAS_WANDB:
        if config.wandb_project and not HAS_WANDB:
            print("Warning: --wandb-project set but wandb not installed. pip install wandb")
        return False

    tags = ["gtpo", "hicra", "tinylora", "countdown"]

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
        tags=tags,
        notes=(
            "GTPO (arXiv 2508.04349) + HICRA (arXiv 2509.03646) "
            "with TinyLoRA adapters (arXiv 2602.04118) on Countdown task."
        ),
    )

    # Group metrics by prefix for dashboard panels.
    for prefix in ["entropy", "reward", "hicra", "train", "policy", "accuracy", "completions"]:
        wandb.define_metric(f"{prefix}/*", step_metric="step")

    return True


def compute_episode_stats(episodes: List[Any]) -> Dict[str, Any]:
    """Compute per-rollout distribution stats from raw episodes.

    Returns min/max lengths, reward extremes, and reward-diversity fraction
    recommended by TRL GRPOTrainer and the GTPO paper (arXiv 2508.04349
    Appendix E.1: entropy consolidation, E.2: response length analysis).
    """
    rewards: List[float] = []
    lengths: List[int] = []
    planning_ratios: List[float] = []

    for ep in episodes:
        if isinstance(ep, dict):
            rew = ep.get("rew", [])
            act = ep.get("act", [])
        else:
            rew = ep.rew
            act = ep.act
        reward_val = float(rew[0]) if rew else 0.0
        comp_len = len(act[0]) if act else 0
        rewards.append(reward_val)
        lengths.append(comp_len)

    stats: Dict[str, Any] = {}
    if lengths:
        stats["min_length"] = min(lengths)
        stats["max_length"] = max(lengths)
    if rewards:
        stats["reward_min"] = min(rewards)
        stats["reward_max"] = max(rewards)
        # frac_zero_std: fraction of the group with identical reward → no
        # advantage signal.  When high, effective batch size shrinks
        # (TRL GRPOTrainer metric: frac_reward_zero_std).
        if len(rewards) > 1:
            reward_set = set(rewards)
            stats["frac_zero_std"] = 1.0 if len(reward_set) == 1 else 0.0
        else:
            stats["frac_zero_std"] = 1.0
    return stats


def log_wandb_step(
    step: int,
    step_stats: Dict[str, Any],
    train_metrics: Dict[str, Any],
    episode_stats: Dict[str, Any],
    config: ReasoningConfig,
    use_wandb: bool,
) -> None:
    """Log structured metrics to wandb for a single training step."""
    if not use_wandb:
        return

    log: Dict[str, Any] = {"step": step}

    # ── Entropy dynamics (GTPO collapse detection) ────────────────────
    # arXiv 2508.04349 Appendix E.1: track entropy consolidation via
    # coefficient of variation.  A declining ratio signals the entropy
    # distribution is converging across successful sequences.
    log["entropy/mean"] = step_stats["entropy_mean"]
    log["entropy/std"] = step_stats["entropy_std"]
    ent_mean = step_stats["entropy_mean"]
    if ent_mean > 0:
        log["entropy/collapse_indicator"] = step_stats["entropy_std"] / ent_mean

    # ── Reward signal ─────────────────────────────────────────────────
    log["reward/mean"] = step_stats["mean_reward"]
    log["reward/std"] = step_stats["std_reward"]
    if "reward_min" in episode_stats:
        log["reward/min"] = episode_stats["reward_min"]
        log["reward/max"] = episode_stats["reward_max"]
    if "frac_zero_std" in episode_stats:
        log["reward/frac_zero_std"] = episode_stats["frac_zero_std"]

    # ── HICRA planning token coverage ─────────────────────────────────
    # arXiv 2509.03646 tracks planning ratio distribution, not just mean.
    log["hicra/planning_token_ratio"] = step_stats["planning_token_ratio"]

    # ── Training loss + gradient norm ─────────────────────────────────
    log["train/loss"] = train_metrics["loss"]
    if "grad_norm" in train_metrics:
        log["train/grad_norm"] = train_metrics["grad_norm"]
    log["train/learning_rate"] = config.learning_rate

    # ── Accuracy ──────────────────────────────────────────────────────
    total = step_stats["total_count"]
    log["accuracy/correct"] = step_stats["correct_count"]
    log["accuracy/total"] = total
    if total > 0:
        log["accuracy/rate"] = step_stats["correct_count"] / total

    # ── Completion lengths ────────────────────────────────────────────
    # TRL GRPOTrainer logs mean/min/max (arXiv 2508.04349 Appendix E.2).
    log["completions/mean_length"] = step_stats["mean_completion_length"]
    if "min_length" in episode_stats:
        log["completions/min_length"] = episode_stats["min_length"]
        log["completions/max_length"] = episode_stats["max_length"]

    # ── Policy metrics (only present on metrics_interval steps) ───────
    if "clip_fraction" in train_metrics:
        log["policy/clip_fraction"] = train_metrics["clip_fraction"]
        log["policy/clip_fraction_lower"] = train_metrics["clip_fraction_lower"]
        log["policy/clip_fraction_upper"] = train_metrics["clip_fraction_upper"]
        log["policy/kl_divergence"] = train_metrics["kl_divergence"]
        log["policy/mean_ratio"] = train_metrics["mean_ratio"]
        log["policy/mean_advantage"] = train_metrics["mean_advantage"]
        log["policy/std_advantage"] = train_metrics["std_advantage"]

    # Keep wandb's internal global step aligned with our explicit training step.
    wandb.log(log, step=step)


def log_wandb_completions(
    step: int,
    episodes: List[Any],
    tokenizer: Any,
    use_wandb: bool,
) -> None:
    """Log a wandb.Table of decoded completions for qualitative inspection."""
    if not use_wandb or step % 10 != 0:
        return

    def _as_list(value: Any) -> List[Any]:
        """Normalize list/tuple/array/scalar values into a Python list."""
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        return [value]

    def _flatten_token_ids(value: Any) -> List[int]:
        """Flatten one level of token nesting and coerce to integer token IDs."""
        seq = _as_list(value)
        flat: List[int] = []
        for item in seq:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, tuple):
                item = list(item)
            if isinstance(item, list):
                flat.extend(int(t) for t in item)
            else:
                flat.append(int(item))
        return flat

    rows = []
    for ep in episodes:
        if isinstance(ep, dict):
            obs = ep.get("obs", [])
            act = ep.get("act", [])
            rew = ep.get("rew", [])
        else:
            obs = ep.obs
            act = ep.act
            rew = ep.rew

        obs_list = _as_list(obs)
        act_list = _as_list(act)
        rew_list = _as_list(rew)

        prompt_tokens = _flatten_token_ids(obs_list[0]) if obs_list else []
        completion_tokens = _flatten_token_ids(act_list[0]) if act_list else []

        reward_raw: Any = rew_list[0] if rew_list else 0.0
        if isinstance(reward_raw, (list, tuple)):
            reward_raw = reward_raw[0] if reward_raw else 0.0
        if hasattr(reward_raw, "item"):
            reward_raw = reward_raw.item()
        reward_val = float(reward_raw)

        prompt_text = tokenizer.decode(prompt_tokens) if len(prompt_tokens) > 0 else ""
        completion_text = tokenizer.decode(completion_tokens) if len(completion_tokens) > 0 else ""

        rows.append([
            step,
            prompt_text[:200],
            completion_text[:500],
            reward_val,
            reward_val >= 1.0,
            len(completion_tokens),
        ])

    table = wandb.Table(
        columns=["step", "prompt", "completion", "reward", "correct", "length"],
        data=rows,
    )
    wandb.log({"completions/samples": table}, step=step)


def run_experiment(config: ReasoningConfig) -> None:
    if config.batch_size > config.episodes_per_step:
        raise ValueError(
            f"batch_size ({config.batch_size}) exceeds episodes_per_step "
            f"({config.episodes_per_step}). batch_size should be <= episodes_per_step."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    print(f"Loading model: {config.model_id}")
    base_model, tokenizer = load_model(config.model_id)

    strategic_grams = None
    if config.strategic_grams_path:
        strategic_grams = load_strategic_grams(config.strategic_grams_path)
        print(
            f"Loaded {len(strategic_grams)} strategic grams from "
            f"{config.strategic_grams_path}"
        )

    optimizer = optim.Adam(learning_rate=config.learning_rate)

    print(
        f"GTPO (Eq. 3,5,6): α₁={config.alpha_1}, α₂={config.alpha_2}, "
        f"threshold={config.reward_threshold}, γ_hicra={config.hicra_gamma}"
    )

    # Enable per-step policy metrics only when wandb will actually be active.
    # Otherwise this triggers an extra model forward pass every step with no
    # consumer (wasted compute when --wandb-project is set but wandb missing).
    trainer_kwargs: Dict[str, Any] = {}
    if config.wandb_project and HAS_WANDB:
        trainer_kwargs["metrics_fn"] = grpo.compute_metrics
        trainer_kwargs["metrics_interval"] = 1

    # 1. LoRA setup
    lora_model, memory_stats = create_lora_setup(
        model=base_model,
        lora_config={
            "lora_layers": config.lora_layers,
            "lora_rank": config.lora_rank,
            "lora_scale": config.lora_scale,
            "lora_dropout": config.lora_dropout,
        },
        adapter_save_path=str(output_dir / "lora_adapters.safetensors"),
    )

    # 2. GTPO advantage transform (arXiv 2508.04349 Eq. 3, 5, 6)
    transform = build_gtpo_transform(
        alpha_1=config.alpha_1,
        alpha_2=config.alpha_2,
        reward_threshold=config.reward_threshold,
        tokenizer=tokenizer,
        strategic_grams=strategic_grams,
        hicra_gamma=config.hicra_gamma,
    )

    # 3. Trainer
    trainer = Trainer(
        model=lora_model,
        advantage_fn=compute_advantages,
        loss_fn=policy_loss,
        optimizer=optimizer,
        advantage_transform_fn=transform,
        max_grad_norm=config.max_grad_norm,
        compile_training=config.compile_training,
        gradient_checkpointing=config.gradient_checkpointing,
        micro_batch_size=config.micro_batch_size,
        profile=config.profile_training,
        **trainer_kwargs,
    )
    model = lora_model

    print(
        "Reasoning stack ready"
    )
    print(
        "  Memory savings: "
        f"{memory_stats.get('memory_savings_percent', 0.0):.1f}%"
    )

    print(f"Generating {config.num_problems} countdown problems (seed={config.dataset_seed})...")
    problems = generate_countdown_problems(config.num_problems, seed=config.dataset_seed)
    prompts = [format_countdown_prompt(p["target"], p["numbers"]) for p in problems]

    def create_env():
        return TextGenerationEnv(
            prompts=prompts,
            reward_fn=countdown_reward,
            max_tokens=config.max_completion_tokens,
            tokenizer=tokenizer,
            examples=problems,
            group_size=config.episodes_per_step,
        )

    policy_fn = create_policy(
        model,
        tokenizer,
        generation_params={
            "max_tokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
    )

    rollout = RolloutCoordinator(
        env_fn=create_env,
        policy_fn=lambda: policy_fn,
        algorithm="grpo",
        num_workers=0,
        max_steps=config.episodes_per_step,
        max_episodes=config.episodes_per_step,
        batch_size=config.batch_size,
        model=model,
        tokenizer=tokenizer,
        generation_params={
            "max_tokens": config.max_completion_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        },
        profile=config.profile_training,
    )

    buffer = Buffer(max_episodes=config.episodes_per_step)
    trainer.link_buffer(buffer, data_selector_fn=grpo.select_recent_data)
    emergence = EmergenceLogger(output_dir=output_dir / "emergence")
    use_wandb = init_wandb(config)

    print(f"\nStarting reasoning training for {config.max_steps} steps...")
    global_episode_count = 0
    phase_totals: Dict[str, float] = {
        "rollout_collect_s": 0.0,
        "emergence_log_s": 0.0,
        "train_s": 0.0,
        "checkpoint_s": 0.0,
    }
    trainer_phase_totals: Dict[str, float] = {}
    rollout_phase_totals: Dict[str, float] = {}
    for step in range(config.max_steps):
        step_start = time.perf_counter()

        buffer.clear()
        rollout_start = time.perf_counter()
        rollout_buffer = rollout.collect()
        for ep in rollout_buffer.episodes:
            buffer.add_episode_from_dict(ep.to_dict())
        phase_totals["rollout_collect_s"] += time.perf_counter() - rollout_start

        # Accumulate rollout sub-phase timing when profiling is enabled.
        for phase, secs in rollout.get_rollout_timing().items():
            rollout_phase_totals[phase] = rollout_phase_totals.get(phase, 0.0) + secs

        new_episode = rollout_buffer.episodes[-1]
        example = problems[global_episode_count % len(problems)]
        global_episode_count += 1

        emergence_start = time.perf_counter()
        step_stats = emergence.log_step(
            step=step,
            episodes=[new_episode],
            tokenizer=tokenizer,
            examples=[example],
        )
        phase_totals["emergence_log_s"] += time.perf_counter() - emergence_start

        train_start = time.perf_counter()
        metrics = trainer.train()
        phase_totals["train_s"] += time.perf_counter() - train_start
        for key, value in metrics.items():
            if key.startswith("timing/") and key.endswith("_s"):
                phase = key[len("timing/") : -2]
                trainer_phase_totals[phase] = trainer_phase_totals.get(phase, 0.0) + float(value)

        ep_stats = compute_episode_stats(rollout_buffer.episodes) if use_wandb else {}
        log_wandb_step(step, step_stats, metrics, ep_stats, config, use_wandb)
        log_wandb_completions(step, rollout_buffer.episodes, tokenizer, use_wandb)

        if step % 10 == 0:
            cumulative_total = sum(phase_totals.values()) or 1e-9
            rollout_pct = phase_totals["rollout_collect_s"] / cumulative_total * 100
            train_pct = phase_totals["train_s"] / cumulative_total * 100
            print(
                f"Step {step}: loss={metrics['loss']:.4f} "
                f"reward={step_stats['mean_reward']:.3f} "
                f"correct={step_stats['correct_count']}/{step_stats['total_count']} "
                f"planning_ratio={step_stats['planning_token_ratio']:.4f} "
                f"step_time={time.perf_counter() - step_start:.2f}s "
                f"[rollout {rollout_pct:.0f}% | train {train_pct:.0f}%]"
            )

        if step % 100 == 0 and step > 0:
            checkpoint_start = time.perf_counter()
            save_checkpoint(model, output_dir, step)
            phase_totals["checkpoint_s"] += time.perf_counter() - checkpoint_start

    emergence.finish()
    rollout.close()
    checkpoint_start = time.perf_counter()
    save_checkpoint(model, output_dir, config.max_steps)
    phase_totals["checkpoint_s"] += time.perf_counter() - checkpoint_start
    print_summary(output_dir)
    print_amdahl_summary(phase_totals, trainer_phase_totals, rollout_phase_totals)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run TinyLoRA + GTPO + HICRA on the Countdown task"
    )
    parser.add_argument("--model", default="arcee-ai/Trinity-Nano-Preview", help="Model ID")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--output", default="results/countdown_reasoning_lora", help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num-problems", type=int, default=50, help="Number of countdown problems")
    parser.add_argument("--episodes-per-step", type=int, default=8, help="Episodes per training step")
    parser.add_argument("--batch-size", type=int, default=8, help="Batched generation across episodes")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens per episode")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed")
    parser.add_argument("--lora-rank", type=int, default=2, help="LoRA rank")
    parser.add_argument("--lora-layers", type=int, default=4, help="LoRA layers")
    parser.add_argument("--lora-scale", type=float, default=8.0, help="LoRA scale")
    parser.add_argument(
        "--compile-training",
        choices=["false", "true", "auto"],
        default="false",
        help="Training compilation mode (HICRA defaults to false for compatibility)",
    )
    parser.add_argument(
        "--strategic-grams",
        default=None,
        help="Optional path to strategic grams JSON (defaults to built-ins)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "Enable gradient checkpointing — recomputes activations during "
            "backward pass instead of caching them. This reduces peak "
            "memory at the cost of additional compute. In this trainer, "
            "True uses sqrt(n) layer selection."
        ),
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help=(
            "Process at most N episodes per logprob-extraction forward pass "
            "to reduce peak activation memory. Start with 4, then tune for "
            "your model/hardware. Combine with --gradient-checkpointing "
            "when still memory constrained."
        ),
    )
    parser.add_argument(
        "--profile-training",
        action="store_true",
        help="Enable trainer per-phase timing and Amdahl bottleneck summary",
    )
    parser.add_argument("--alpha-1", type=float, default=1.0, help="GTPO: base reward weight (Eq. 3)")
    parser.add_argument("--alpha-2", type=float, default=0.1, help="GTPO: entropy-shaped weight (Eq. 3)")
    parser.add_argument("--reward-threshold", type=float, default=0.5, help="GTPO: O+/O- partition threshold")
    parser.add_argument("--hicra-gamma", type=float, default=0.3, help="HICRA entropy boost factor for GTPO fusion")
    parser.add_argument("--wandb-project", default=None, help="Wandb project name (enables wandb logging)")
    parser.add_argument("--wandb-run-name", default=None, help="Wandb run name (optional)")
    args = parser.parse_args()

    compile_mode: Union[bool, str]
    if args.compile_training == "true":
        compile_mode = True
    elif args.compile_training == "false":
        compile_mode = False
    else:
        compile_mode = "auto"

    cfg = ReasoningConfig(
        model_id=args.model,
        max_steps=args.steps,
        output_dir=args.output,
        learning_rate=args.lr,
        num_problems=args.num_problems,
        episodes_per_step=args.episodes_per_step,
        batch_size=args.batch_size,
        max_completion_tokens=args.max_tokens,
        temperature=args.temperature,
        dataset_seed=args.seed,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        lora_scale=args.lora_scale,
        strategic_grams_path=args.strategic_grams,
        compile_training=compile_mode,
        gradient_checkpointing=args.gradient_checkpointing,
        micro_batch_size=args.micro_batch_size,
        profile_training=args.profile_training,
        alpha_1=args.alpha_1,
        alpha_2=args.alpha_2,
        reward_threshold=args.reward_threshold,
        hicra_gamma=args.hicra_gamma,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    run_experiment(cfg)
