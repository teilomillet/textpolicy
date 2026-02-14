# 12. Running RL Advantages on GPU via Tinker

Train language models on math reasoning with textpolicy's advantage pipeline (MaxRL + GTPO + HICRA + SEPA), running at scale on GPUs through the Tinker API.

## Intent

Run our full token-level advantage pipeline on GPU hardware. Tinker handles sampling, forward/backward passes, and optimizer steps on its GPUs. We handle advantage computation, reward grading, and SEPA scheduling in pure Python on your local machine. The advantages get packed into Tinker's `Datum` objects and sent over the wire.

## What Tinker Provides vs What We Provide

| Tinker (GPU) | textpolicy (local) |
|---|---|
| Model hosting (Qwen3-4B, etc.) | Advantage computation (pure Python) |
| LoRA training client | Reward grading (extract `\boxed{}`, compare) |
| Sampling with temperature/top-p | SEPA scheduling (linear ramp, auto) |
| `forward_backward` + `optim_step` | Planning token detection (strategic grams) |
| Checkpoint saving | Metrics logging (JSONL) |

The boundary is clean: we compute `List[float]` advantages, convert them to `TensorData`, pack them into `Datum` objects, and send them to Tinker. No torch tensors are created on our side except for the final packing step.

## Prerequisites

1. **Tinker account and API key** — set `TINKER_API_KEY` in your environment or a `.env` file at the repo root
2. **Python dependencies**:
   ```bash
   uv add tinker torch transformers datasets
   ```
3. **No GPU required locally** — Tinker runs the model on its infrastructure

## Quick Start

```bash
# 1. Set your API key
export TINKER_API_KEY=<your-key>

# 2. Smoke test (5 steps, small group — takes ~2 minutes)
uv run python -m textpolicy.tinker.train_math \
    --max-steps 5 --group-size 4 --batch-size 2

# 3. Check output
cat logs/tinker_math/metrics.jsonl | python -m json.tool --no-ensure-ascii | head -20

# 4. Run with baseline algorithm for comparison
uv run python -m textpolicy.tinker.train_math \
    --algorithm grpo --max-steps 5 --group-size 4 --batch-size 2 \
    --log-dir logs/tinker_math_baseline

# 5. Full run (500 steps, ~4 hours)
nohup uv run python -m textpolicy.tinker.train_math \
    --max-steps 500 --batch-size 8 --group-size 16 \
    > logs/tinker_math.log 2>&1 &
```

## The Advantage Pipeline

Every training step processes a batch of prompts. For each prompt, Tinker samples `G` completions. Our code computes token-level advantages through a 4-stage pipeline:

```
Rewards (per completion)
    │
    ▼
┌─────────┐   scalar advantages    ┌──────┐   token-level    ┌───────┐   amplified    ┌──────┐
│  MaxRL   │ ──────────────────────▶│ GTPO │ ─────────────▶  │ HICRA │ ────────────▶  │ Done │
│          │    per completion      │      │   advantages     │       │   advantages   │      │
└─────────┘                        └──────┘                  └───────┘               └──────┘
                                      ▲
                                      │ pooled entropies
                                   ┌──────┐
                                   │ SEPA │
                                   └──────┘
```

### Stage Details

| Stage | What it does | Key parameter | Formula | Reference |
|-------|-------------|---------------|---------|-----------|
| **MaxRL** | Inverse success-rate reweighting | `eps` (1e-6) | `A_i = (r_i - mean(r)) / (mean(r) + eps)` | Tajwar et al. 2026, Eq. 10 |
| **GTPO** | Entropy-weighted credit assignment | `--gtpo-beta` (0.1) | `w(t) = max(0, 1 + β(H_norm(t) - 1))` | arXiv 2508.04349 |
| **SEPA** | Execution-token entropy pooling | `--sepa-steps` (500) | `H_pool(t) = λ·mean(H_exec) + (1-λ)·H(t)` | — |
| **HICRA** | Planning token amplification | `--hicra-alpha` (0.2) | `A(t) = A(t) + α·|A(t)|·mask(t)` | — |

**MaxRL** reweights at the prompt level: hard problems (low success rate) get proportionally larger gradient. Within each completion, all tokens start with the same scalar advantage.

**GTPO** breaks this uniformity by weighting each token by its normalized entropy. High-entropy tokens (decision points) get amplified; low-entropy tokens (routine execution) get dampened.

**SEPA** cleans the entropy signal before GTPO uses it. Execution tokens have their entropy interpolated toward the execution-token mean, reducing noise. Planning tokens keep their original entropy. The pooling strength λ ramps from 0 to 1 over training.

**HICRA** applies an additive boost to tokens identified as planning tokens via strategic gram matching (phrases like "wait let me", "try another approach", "notice that").

## Baseline vs Full Pipeline

The `--algorithm` flag selects between two modes:

**`--algorithm grpo`** (baseline):
```
A_i = r_i - mean(r)           # scalar per completion
A(t) = A_i  for all tokens    # uniform across tokens
```

**`--algorithm full`** (default):
```
A_i = (r_i - mean(r)) / (mean(r) + eps)     # MaxRL
A(t) = A_i * max(0, 1 + β(H_norm(t) - 1))  # GTPO
A(t) = A(t) + α |A(t)| mask(t)              # HICRA
```
with SEPA pooling applied to entropies before GTPO.

The `grpo` baseline uses Tinker's cookbook default: simple reward centering with uniform token weighting. The `full` pipeline adds all four advantage stages.

## SEPA Scheduling

SEPA pooling strength λ starts at 0 and increases over training. Two schedules are available:

**Linear** (`--sepa-schedule linear`, default):
```
λ(step) = clamp((step - delay) / sepa_steps, 0, 1)
```
- `--sepa-steps 500` — steps to ramp from 0 to 1
- `--sepa-delay-steps 50` — wait this many steps before ramping

**Auto** (`--sepa-schedule auto`):
Adapts λ based on execution-token entropy variance decay, with the linear ramp as a floor. Uses an EMA of batch variance and compares against the initial variance captured after a warmup period.

**Correctness gate** (`--sepa-correct-rate-gate 0.1`): SEPA stays disabled (λ=0) until the model achieves at least 10% correct rate. Once triggered, the gate stays open permanently (sticky). This prevents SEPA from operating before the model produces any useful signal.

## Planning Token Detection

Planning tokens are identified by matching decoded token text against strategic grams — multi-word phrases that indicate reasoning activity:

- **Hesitation**: "wait let me", "let me think", "on second thought"
- **Verification**: "let me check", "let me verify", "is this right"
- **Backtracking**: "try another approach", "go back and", "that's not right"
- **Alternatives**: "another way to", "or we could", "what if we"
- **Metacognition**: "notice that", "the key is", "the key insight"

Custom grams can be passed as a JSON list:
```bash
uv run python -m textpolicy.tinker.train_math \
    --strategic-grams '["let me reconsider", "alternatively"]'
```

The detection uses a sliding window over decoded tokens with word-boundary regex matching. See `advantages.py:identify_planning_tokens` for the implementation.

## CLI Reference

### Model & Tinker

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | HuggingFace model ID |
| `--base-url` | (production) | Tinker service URL |
| `--lora-rank` | `32` | LoRA rank for training |

### Training

| Flag | Default | Description |
|------|---------|-------------|
| `--algorithm` | `full` | `grpo` (baseline) or `full` (MaxRL+GTPO+HICRA+SEPA) |
| `--max-steps` | `500` | Total training steps |
| `--batch-size` | `8` | Prompts per step |
| `--group-size` | `16` | Completions per prompt (G) |
| `--max-tokens` | `2048` | Max completion length |
| `--temperature` | `0.7` | Sampling temperature |
| `--lr` | `4e-5` | Learning rate |
| `--weight-decay` | `0.0` | Adam weight decay |
| `--max-examples` | (all) | Limit dataset size (for debugging) |
| `--save-every` | `20` | Checkpoint every N steps (0 = disabled) |

### Algorithm Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--gtpo-beta` | `0.1` | Entropy weighting strength (β). 0 disables GTPO. |
| `--hicra-alpha` | `0.2` | Planning amplification factor (α). 0 disables HICRA. |
| `--sepa-steps` | `500` | Linear ramp steps. 0 disables SEPA. |
| `--sepa-schedule` | `linear` | `linear` or `auto` |
| `--sepa-delay-steps` | `50` | Delay before SEPA ramp starts |
| `--sepa-correct-rate-gate` | `0.1` | Min correct rate to activate SEPA |
| `--strategic-grams` | (built-in list) | JSON list of planning phrases |

### Logging

| Flag | Default | Description |
|------|---------|-------------|
| `--log-dir` | `logs/tinker_math` | Output directory for metrics and emergence data |

## Output Files

```
logs/tinker_math/
├── metrics.jsonl           # Per-step: loss, correct_rate, sepa_lambda, ...
└── emergence/
    ├── steps.jsonl          # Per-step: mean_reward, correct_count, total_count
    └── generations.jsonl    # Per-completion: prompt, completion, reward, num_tokens
```

**`metrics.jsonl`** — one JSON object per training step:
```json
{
  "step": 42,
  "algorithm": "full",
  "loss": 0.0312,
  "mean_reward": 0.25,
  "correct_rate": 0.25,
  "running_correct_rate": 0.18,
  "sepa_lambda": 0.084,
  "sepa_gate_open": true,
  "num_datums": 24,
  "max_token_hit_rate": 0.125,
  "step_time_s": 45.2
}
```

**`emergence/`** — formatted for the significance analysis framework (see [13. Rigorous A/B Experiments](13_tinker_campaigns.md)).

## How the Port Works

The advantage code in `textpolicy/tinker/advantages.py` is pure Python — no MLX arrays, no torch tensors, no framework dependencies. The functions take and return plain `List[float]` and `List[int]`.

The math is identical to textpolicy's MLX implementation in `algorithms/grpo.py` and `algorithms/hicra.py`. Each function documents its MLX source reference (e.g., "Source: grpo.py:561-662"). The port is validated by 41 unit tests in `tests/test_advantages.py`, including cross-validation tests that verify numerical equivalence with the MLX originals.

The output lists get packed into Tinker `Datum` objects at the boundary:
```python
# Pure Python advantages → torch tensor → TensorData → Datum
padded_advantages = [0.0] * prompt_len + token_advs
datum = types.Datum(
    model_input=model_input,
    loss_fn_inputs={
        "advantages": TensorData.from_torch(
            torch.tensor(padded_advantages, dtype=torch.float32)
        ),
        ...
    },
)
```

This is a **port**, not a backend integration. There is no seamless switch between MLX and Tinker — the advantage functions were manually rewritten as pure Python to work with Tinker's data format.
