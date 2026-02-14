#!/usr/bin/env python3
# textpolicy/tinker/train_math.py
"""
Train Qwen3-4B on MATH with textpolicy's advantage pipeline via Tinker.

This script forks Tinker's cookbook rl_loop.py and replaces the advantage
computation with our full pipeline:
    MaxRL → GTPO → HICRA → SEPA → token-level advantages

Everything else (sampling, forward_backward, optim_step) uses Tinker's
existing infrastructure unchanged.

Usage:
    # Set your API key
    export TINKER_API_KEY=<your-key>

    # Run training (smoke test):
    python -m textpolicy.tinker.train_math --max-steps 5 --group-size 4

    # Full run:
    python -m textpolicy.tinker.train_math --max-steps 500

    # Custom hyperparameters:
    python -m textpolicy.tinker.train_math \\
        --gtpo-beta 0.15 --hicra-alpha 0.25 \\
        --sepa-steps 300 --sepa-schedule linear

References:
    Tinker cookbook: tinker_cookbook/recipes/rl_loop.py
    Our algorithms: textpolicy/algorithms/{grpo,hicra}.py, training/sepa.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from textpolicy.tinker.advantages import (
    DEFAULT_STRATEGIC_GRAMS,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    compute_grpo_advantages,
    compute_maxrl_advantages,
    identify_planning_tokens,
)
from textpolicy.tinker.sepa import SEPAController

# Load .env file if it exists (for TINKER_API_KEY)
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    with _env_path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward / grading
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """Extract \\boxed{...} answer from MATH solution text."""
    # Handle nested braces by counting depth
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


def grade_answer(given: str, expected: str) -> bool:
    """Simple string-match grading. Strips whitespace."""
    return given.strip() == expected.strip()


def get_reward(response: str, answer: str) -> float:
    """Binary correctness reward for MATH problems."""
    given = extract_boxed(response)
    return 1.0 if grade_answer(given, answer) else 0.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MATH_SUBJECTS = [
    "intermediate_algebra", "precalculus", "number_theory",
    "counting_and_probability", "geometry",
]


def load_math_dataset(
    split: str = "train",
    max_examples: Optional[int] = None,
    subjects: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Load hendrycks/MATH dataset (via EleutherAI mirror).

    Returns list of dicts with 'problem' and 'answer' keys.
    """
    from datasets import load_dataset

    if subjects is None:
        subjects = MATH_SUBJECTS

    examples = []
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        for item in ds:
            answer = extract_boxed(item["solution"])
            examples.append({
                "problem": item["problem"],
                "answer": answer,
                "level": item.get("level", ""),
                "type": item.get("type", ""),
            })
            if max_examples and len(examples) >= max_examples:
                break
        if max_examples and len(examples) >= max_examples:
            break
    return examples


# ---------------------------------------------------------------------------
# Token-level advantage computation
# ---------------------------------------------------------------------------

def compute_token_advantages(
    rewards_G: List[float],
    sampled_tokens_G: List[List[int]],
    logprobs_G: List[List[float]],
    tokenizer,
    *,
    gtpo_beta: float = 0.1,
    hicra_alpha: float = 0.2,
    sepa_lambda: float = 0.0,
    strategic_grams: Optional[List[str]] = None,
) -> List[List[float]]:
    """
    Compute token-level advantages for a group of completions.

    Pipeline: MaxRL → GTPO → (SEPA →) HICRA → token-level advantages.
    """
    if strategic_grams is None:
        strategic_grams = DEFAULT_STRATEGIC_GRAMS

    # Step 1: MaxRL group-relative advantages (scalar per completion)
    advantages_G = compute_maxrl_advantages(rewards_G)

    # Step 2: For each completion, compute token-level advantages
    all_token_advs = []
    for tokens, logprobs, advantage in zip(
        sampled_tokens_G, logprobs_G, advantages_G
    ):
        # Entropy proxy: -logprob (higher = more uncertain)
        entropies = [-lp for lp in logprobs]

        # HICRA: identify planning tokens (needed for both SEPA and HICRA)
        planning_mask = identify_planning_tokens(
            tokens, tokenizer, strategic_grams
        )

        # SEPA: pool execution-token entropy toward their mean
        if sepa_lambda > 0.0:
            entropies = apply_sepa_pooling(entropies, planning_mask, sepa_lambda)

        # GTPO: entropy-weighted credit assignment
        token_advs = apply_gtpo_weighting(advantage, entropies, beta=gtpo_beta)

        # HICRA: amplify planning tokens
        token_advs = apply_hicra(token_advs, planning_mask, alpha=hicra_alpha)

        all_token_advs.append(token_advs)

    return all_token_advs


def compute_baseline_advantages(
    rewards_G: List[float],
    sampled_tokens_G: List[List[int]],
) -> List[List[float]]:
    """
    Compute baseline GRPO advantages: scalar per completion, repeated for
    all tokens. No GTPO/HICRA/SEPA — just group-relative centering.

    This is the control arm for A/B comparison.
    """
    # Simple reward centering: A_i = r_i - mean(r)
    advantages_G = compute_grpo_advantages(rewards_G)

    # Repeat scalar advantage for every token (uniform weighting)
    all_token_advs = []
    for tokens, advantage in zip(sampled_tokens_G, advantages_G):
        all_token_advs.append([advantage] * len(tokens))

    return all_token_advs


# ---------------------------------------------------------------------------
# Training loop (real Tinker API)
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training loop using the Tinker API."""
    import torch
    import tinker
    from tinker import types
    from tinker.types.tensor_data import TensorData
    from transformers import AutoTokenizer

    # --- Setup ---
    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Tokenizer (local, for prompt building + planning detection)
    logger.info("Loading tokenizer for %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Tinker service client (reads TINKER_API_KEY from env)
    logger.info("Connecting to Tinker...")
    service_client = tinker.ServiceClient(base_url=args.base_url)

    # Create LoRA training client on Tinker's GPUs
    logger.info("Creating LoRA training client (model=%s, rank=%d)...",
                args.model, args.lora_rank)
    training_client = service_client.create_lora_training_client(
        base_model=args.model,
        rank=args.lora_rank,
    )
    logger.info("Training client ready.")

    # Load dataset
    logger.info("Loading MATH dataset...")
    examples = load_math_dataset(max_examples=args.max_examples)
    logger.info("Loaded %d examples", len(examples))

    # Sampling params
    sampling_params = types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.95,
    )

    # Adam params
    adam_params = types.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # SEPA controller
    sepa_controller = SEPAController(
        sepa_steps=args.sepa_steps,
        sepa_schedule=args.sepa_schedule,
        sepa_delay_steps=args.sepa_delay_steps,
        sepa_correct_rate_gate=args.sepa_correct_rate_gate,
    )

    # Emergence-compatible output (for significance analysis framework)
    emergence_dir = log_path / "emergence"
    emergence_dir.mkdir(parents=True, exist_ok=True)
    steps_path = emergence_dir / "steps.jsonl"
    generations_path = emergence_dir / "generations.jsonl"

    metrics_path = log_path / "metrics.jsonl"

    # --- Training loop ---
    example_idx = 0
    total_correct = 0
    total_completions = 0

    for batch_idx in range(args.max_steps):
        step_start = time.time()

        # ---- 1. Get a sampling client from current weights ----
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{batch_idx}",
        )

        # ---- 2. Select prompts and submit sample requests (async) ----
        # Process multiple prompts per batch for efficiency
        batch_prompts = []
        batch_answers = []
        sample_futures = []

        for _ in range(args.batch_size):
            example = examples[example_idx % len(examples)]
            example_idx += 1

            prompt_text = example["problem"]
            answer = example["answer"]

            # Build model input from tokenized prompt
            # Use chat template if available, otherwise raw text
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt_text}]
                prompt_ids = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            else:
                prompt_ids = tokenizer.encode(prompt_text)

            # Ensure prompt_ids is a plain list[int] — some tokenizers
            # return BatchEncoding dicts instead of lists.
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids["input_ids"]
            if not isinstance(prompt_ids, list):
                prompt_ids = list(prompt_ids)

            model_input = types.ModelInput.from_ints(prompt_ids)

            future = sampling_client.sample(
                prompt=model_input,
                num_samples=args.group_size,
                sampling_params=sampling_params,
            )
            sample_futures.append(future)
            batch_prompts.append((prompt_text, prompt_ids))
            batch_answers.append(answer)

        # ---- 3. Collect results, compute advantages, build datums ----
        datums = []
        batch_rewards = []
        batch_correct = 0
        batch_max_token_hits = 0
        batch_total_completions = 0
        # Accumulate across all prompt groups for SEPA auto-schedule
        all_sampled_tokens = []
        all_logprobs = []

        for future, (prompt_text, prompt_ids), answer in zip(
            sample_futures, batch_prompts, batch_answers
        ):
            sample_result = future.result()
            ob_len = len(prompt_ids)

            # Extract per-completion data
            rewards_G = []
            sampled_tokens_G = []
            logprobs_G = []

            for seq in sample_result.sequences:
                # Decode completion text for reward
                completion_tokens = list(seq.tokens)
                completion_text = tokenizer.decode(
                    completion_tokens, skip_special_tokens=True
                )
                reward = get_reward(completion_text, answer)
                rewards_G.append(reward)
                sampled_tokens_G.append(completion_tokens)
                logprobs_G.append(list(seq.logprobs))

            all_sampled_tokens.extend(sampled_tokens_G)
            all_logprobs.extend(logprobs_G)

            batch_rewards.extend(rewards_G)
            batch_correct += sum(1 for r in rewards_G if r > 0.5)

            # Track max-token truncations (Lee et al. 2026: wasted compute signal)
            for seq in sample_result.sequences:
                batch_total_completions += 1
                if len(seq.tokens) >= args.max_tokens:
                    batch_max_token_hits += 1

            group_correct = sum(1 for r in rewards_G if r > 0.5)
            logger.info(
                "  group: %d/%d correct | answer=%s",
                group_correct, len(rewards_G), answer[:40],
            )

            # Skip uninformative groups (all same reward → zero advantage)
            # MaxRL produces zero advantages for these anyway, but skipping
            # saves Tinker API calls.
            if all(r == rewards_G[0] for r in rewards_G):
                logger.info("    → skipped (all %s)",
                            "correct" if rewards_G[0] > 0.5 else "wrong")
                continue

            # Compute advantages using the selected algorithm
            if args.algorithm == "grpo":
                # Baseline: scalar GRPO advantages, uniform across tokens
                token_advs_G = compute_baseline_advantages(
                    rewards_G=rewards_G,
                    sampled_tokens_G=sampled_tokens_G,
                )
                sepa_lambda = 0.0
            else:
                # Full pipeline: MaxRL → GTPO → SEPA → HICRA
                sepa_lambda = sepa_controller.resolve_lambda(step=batch_idx)
                token_advs_G = compute_token_advantages(
                    rewards_G=rewards_G,
                    sampled_tokens_G=sampled_tokens_G,
                    logprobs_G=logprobs_G,
                    tokenizer=tokenizer,
                    gtpo_beta=args.gtpo_beta,
                    hicra_alpha=args.hicra_alpha,
                    sepa_lambda=sepa_lambda,
                    strategic_grams=args.strategic_grams,
                )

            # Build Datum objects for Tinker
            for seq, token_advs in zip(sample_result.sequences, token_advs_G):
                completion_tokens = list(seq.tokens)
                completion_logprobs = list(seq.logprobs)

                # Full sequence: prompt + completion
                full_tokens = list(prompt_ids) + completion_tokens

                # Logprobs: zeros for prompt, real logprobs for completion
                padded_logprobs = [0.0] * ob_len + completion_logprobs

                # Advantages: zeros for prompt, our token-level values
                padded_advantages = [0.0] * ob_len + token_advs

                # model_input must contain the FULL sequence (prompt +
                # completion) so Tinker can compute the forward pass.
                # loss_fn_inputs arrays must match this full length.
                model_input = types.ModelInput.from_ints(full_tokens)

                datum = types.Datum(
                    model_input=model_input,
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(
                            torch.tensor(full_tokens, dtype=torch.long)
                        ),
                        "logprobs": TensorData.from_torch(
                            torch.tensor(padded_logprobs, dtype=torch.float32)
                        ),
                        "advantages": TensorData.from_torch(
                            torch.tensor(padded_advantages, dtype=torch.float32)
                        ),
                    },
                )
                datums.append(datum)

            # Per-generation emergence logging (significance-framework format)
            with open(generations_path, "a") as gf:
                for seq, reward in zip(sample_result.sequences, rewards_G):
                    completion_text = tokenizer.decode(
                        list(seq.tokens), skip_special_tokens=True
                    )
                    gf.write(json.dumps({
                        "step": batch_idx,
                        "prompt": prompt_text[:200],
                        "completion": completion_text[:500],
                        "reward": reward,
                        "num_tokens": len(seq.tokens),
                        "metadata": {"correctness": reward >= 0.99},
                    }) + "\n")

        # ---- 4. SEPA state updates (full pipeline only) ----
        total_completions += len(batch_rewards)
        total_correct += batch_correct
        correct_rate = batch_correct / max(len(batch_rewards), 1)

        if args.algorithm == "full":
            sepa_controller.observe_correct_rate(correct_rate)

            if sepa_controller.enabled and sepa_controller.sepa_schedule == "auto":
                for tokens, logprobs in zip(all_sampled_tokens, all_logprobs):
                    planning_mask = identify_planning_tokens(
                        tokens, tokenizer,
                        args.strategic_grams or DEFAULT_STRATEGIC_GRAMS,
                    )
                    exec_entropies = [
                        -lp for lp, m in zip(logprobs, planning_mask) if not m
                    ]
                    sepa_controller.update_auto_state(exec_entropies)

        # ---- 5. Train: forward_backward + optim_step ----
        if not datums:
            logger.warning("Step %d: no informative datums, skipping.", batch_idx)
            continue

        logger.info("Step %d: submitting %d datums for training...",
                     batch_idx, len(datums))

        fwd_bwd_future = training_client.forward_backward(
            datums, loss_fn="importance_sampling"
        )
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_future.result()  # wait for optimizer step to complete

        step_time = time.time() - step_start

        # ---- 6. Logging ----
        mean_reward = sum(batch_rewards) / max(len(batch_rewards), 1)
        running_correct_rate = total_correct / max(total_completions, 1)

        # Extract loss from forward_backward result.
        # ForwardBackwardOutput.metrics contains {"loss:sum": float, ...}
        # ForwardBackwardOutput.loss_fn_outputs contains per-datum logprobs.
        loss_value = 0.0
        if hasattr(fwd_bwd_result, "metrics") and fwd_bwd_result.metrics:
            # Tinker uses "loss:sum" key, not "loss"
            loss_sum = fwd_bwd_result.metrics.get("loss:sum", 0.0)
            # Normalize by number of datums for per-datum average loss
            loss_value = float(loss_sum) / max(len(datums), 1)

        max_token_hit_rate = (
            batch_max_token_hits / max(batch_total_completions, 1)
        )

        step_metrics = {
            "step": batch_idx,
            "algorithm": args.algorithm,
            "loss": loss_value,
            "mean_reward": mean_reward,
            "correct_rate": correct_rate,
            "running_correct_rate": running_correct_rate,
            "sepa_lambda": sepa_lambda if args.algorithm == "full" else 0.0,
            "sepa_gate_open": sepa_controller.gate_open if args.algorithm == "full" else False,
            "num_datums": len(datums),
            "max_token_hit_rate": max_token_hit_rate,
            "step_time_s": step_time,
        }

        logger.info(
            "Step %d | loss=%.4f | reward=%.2f | correct=%.1f%% | "
            "datums=%d | sepa_λ=%.3f | time=%.1fs",
            batch_idx, loss_value, mean_reward,
            correct_rate * 100, len(datums),
            sepa_controller.resolve_lambda(step=batch_idx),
            step_time,
        )

        with open(metrics_path, "a") as f:
            f.write(json.dumps(step_metrics) + "\n")

        # Emergence-compatible step record (for significance analysis)
        with open(steps_path, "a") as sf:
            sf.write(json.dumps({
                "step": batch_idx,
                "mean_reward": mean_reward,
                "correct_count": batch_correct,
                "total_count": len(batch_rewards),
            }) + "\n")

        # Periodic checkpoint
        if args.save_every and (batch_idx + 1) % args.save_every == 0:
            ckpt_name = f"checkpoint_step_{batch_idx + 1}"
            training_client.save_state(name=ckpt_name)
            logger.info("Saved checkpoint: %s", ckpt_name)

    # Save final checkpoint
    training_client.save_state(name="final")
    logger.info(
        "Training complete. algorithm=%s, %d steps, running correct rate: %.1f%%",
        args.algorithm, args.max_steps,
        100.0 * total_correct / max(total_completions, 1),
    )
    logger.info("Metrics saved to %s", metrics_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on MATH with textpolicy advantages via Tinker"
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm", type=str, default="full",
        choices=["grpo", "full"],
        help="Advantage algorithm: 'grpo' (baseline) or 'full' (MaxRL+GTPO+HICRA+SEPA)",
    )

    # Model & Tinker
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model ID",
    )
    parser.add_argument("--base-url", type=str, default=None,
                        help="Tinker service URL (default: production)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank for training")

    # Training
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Prompts per training step")
    parser.add_argument("--group-size", type=int, default=16,
                        help="Completions per prompt (G)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit dataset size (for debugging)")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Checkpoint every N steps (0 = disabled)")

    # Algorithm hyperparameters
    parser.add_argument("--gtpo-beta", type=float, default=0.1,
                        help="GTPO entropy weighting strength")
    parser.add_argument("--hicra-alpha", type=float, default=0.2,
                        help="HICRA planning amplification factor")

    # SEPA
    parser.add_argument("--sepa-steps", type=int, default=500,
                        help="SEPA linear ramp steps (0 = disabled)")
    parser.add_argument("--sepa-schedule", type=str, default="linear",
                        choices=["linear", "auto"])
    parser.add_argument("--sepa-delay-steps", type=int, default=50)
    parser.add_argument("--sepa-correct-rate-gate", type=float, default=0.1,
                        help="Min correct rate to activate SEPA")

    # Strategic grams
    parser.add_argument("--strategic-grams", type=str, default=None,
                        help="JSON list of strategic gram phrases")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs/tinker_math")

    args = parser.parse_args()

    if args.strategic_grams:
        args.strategic_grams = json.loads(args.strategic_grams)

    return args


if __name__ == "__main__":
    train(parse_args())
