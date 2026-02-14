# 13. Rigorous A/B Experiments on Tinker

Run fair, reproducible A/B comparisons between our full advantage pipeline (MaxRL+GTPO+HICRA+SEPA) and the GRPO baseline, with statistical significance testing and power analysis.

## Intent

Compare two advantage algorithms under controlled conditions on GPU. The methodology follows three phases: **LR sweep** (find optimal learning rate per algorithm), **campaign** (run matched A/B arms at tuned LRs), and **analysis** (significance tests, power estimates). This workflow produces paper-referenceable results.

## Why Three Phases

A fair comparison requires each algorithm to run at its own optimal learning rate. Lee et al. (2026) show that optimal LR varies significantly across RL methods — in our sweeps, the gap between GRPO and the full pipeline is typically 10x (e.g., 5e-4 vs 5e-5). Running both algorithms at the same LR disadvantages one of them, making any A/B comparison meaningless.

## Phase 1: Learning Rate Sweep

### Why It Matters

The LR sweep runs short probes (20 steps each) at multiple learning rates for both algorithms. This is cheap relative to a full campaign and prevents the most common experimental error: comparing algorithms at a learning rate that favors one over the other.

Default sweep grid: `5e-5, 2e-4, 5e-4, 1e-3` for both `grpo` and `full`.

### Running the Sweep

```bash
# Set API key
export TINKER_API_KEY=<your-key>

# Run sweep (~40 min for 4 LRs × 2 algorithms × 20 steps)
nohup uv run python scripts/tinker_lr_sweep.py \
    --sweep-root results/lr_sweep \
    > results/lr_sweep.log 2>&1 &

# Monitor progress
tail -20 results/lr_sweep.log
```

### Customizing the Sweep

```bash
# Custom LR grid
uv run python scripts/tinker_lr_sweep.py \
    --lr-values "1e-5,5e-5,1e-4,5e-4" \
    --sweep-root results/lr_sweep_fine

# More steps per probe (higher confidence, longer runtime)
uv run python scripts/tinker_lr_sweep.py \
    --probe-steps 40 \
    --sweep-root results/lr_sweep_40step

# Sweep only one algorithm
uv run python scripts/tinker_lr_sweep.py \
    --algorithms "full" \
    --sweep-root results/lr_sweep_full_only
```

### Interpreting Results

The sweep prints a results table at the end:

```
================================================================================
  LEARNING RATE SWEEP RESULTS
================================================================================
 Algorithm         LR  Steps  Datums  Correct%  Running%   MeanLoss   Status
────────────────────────────────────────────────────────────────────────────────
      grpo      5e-05     20     114      7.0%      6.8%     0.0234       ok
      grpo      2e-04     20     118     10.0%      9.2%     0.0198       ok
      grpo      5e-04     20     122     14.0%     11.5%     0.0187       ok
      grpo      1e-03     20     108      5.0%      5.1%     0.0312       ok
      full      5e-05     20     120     12.0%     10.8%     0.0201       ok
      full      2e-04     20     116      8.0%      7.9%     0.0245       ok
      full      5e-04     20     110      6.0%      5.5%     0.0289       ok
      full      1e-03     20      98      2.0%      2.1%     0.0401       ok
================================================================================
```

The sweep selects the best LR per algorithm by running correct rate. It then prints the recommended campaign command with per-arm LRs.

**Key finding from our sweeps**: GRPO typically performs best at higher LRs (5e-4), while the full pipeline performs best at lower LRs (5e-5). This 10x gap is expected — the full pipeline produces larger-magnitude advantages due to MaxRL's inverse success-rate reweighting, so it needs a smaller LR to remain stable.

### Output Files

```
results/lr_sweep/
├── sweep_results.json       # Full results with best_lr per algorithm
├── grpo_lr5e-05/
│   └── metrics.jsonl
├── grpo_lr2e-04/
│   └── metrics.jsonl
├── full_lr5e-05/
│   └── metrics.jsonl
└── ...
```

## Phase 2: Campaign

The campaign script runs two arms sequentially (or in parallel via separate invocations) with matched settings. Only the advantage computation differs.

### Dry Run (Plan Only)

Always start with a dry run to review the plan:

```bash
uv run python scripts/tinker_campaign.py \
    --max-steps 100 \
    --batch-size 4 --group-size 8 \
    --baseline-lr 5e-4 --candidate-lr 5e-5
```

This creates:
- `results/tinker_campaign_<timestamp>/manifest.json` — full configuration
- `results/tinker_campaign_<timestamp>/run_commands.sh` — executable shell script

### Execute

```bash
# Sequential execution (both arms, same session)
uv run python scripts/tinker_campaign.py \
    --max-steps 100 \
    --batch-size 4 --group-size 8 \
    --baseline-lr 5e-4 --candidate-lr 5e-5 \
    --campaign-root results/tinker_ab_v1 \
    --execute

# Or run as a background job
nohup uv run python scripts/tinker_campaign.py \
    --max-steps 100 \
    --batch-size 4 --group-size 8 \
    --baseline-lr 5e-4 --candidate-lr 5e-5 \
    --campaign-root results/tinker_ab_v1 \
    --execute \
    > results/tinker_ab_v1.log 2>&1 &
```

### Parallel Arms (Separate Tinker Sessions)

For faster execution, run both arms simultaneously. Each arm gets its own Tinker training client:

```bash
# Terminal 1: baseline arm
nohup uv run python -m textpolicy.tinker.train_math \
    --algorithm grpo --lr 5e-4 \
    --max-steps 100 --batch-size 4 --group-size 8 \
    --lora-rank 64 \
    --log-dir results/tinker_ab_v1/baseline \
    > results/tinker_ab_v1_baseline.log 2>&1 &

# Terminal 2: candidate arm
nohup uv run python -m textpolicy.tinker.train_math \
    --algorithm full --lr 5e-5 \
    --max-steps 100 --batch-size 4 --group-size 8 \
    --lora-rank 64 \
    --log-dir results/tinker_ab_v1/candidate \
    > results/tinker_ab_v1_candidate.log 2>&1 &
```

### Per-Arm Learning Rates

Use `--baseline-lr` and `--candidate-lr` to override the shared `--lr` for each arm:

```bash
uv run python scripts/tinker_campaign.py \
    --lr 4e-5 \
    --baseline-lr 5e-4 \
    --candidate-lr 5e-5 \
    --execute
```

If only `--lr` is given (no per-arm overrides), both arms use the same learning rate.

### Directory Structure

```
results/tinker_ab_v1/
├── manifest.json           # Campaign config, commands, status
├── run_commands.sh          # Re-runnable shell script
├── baseline/
│   ├── metrics.jsonl
│   └── emergence/
│       ├── steps.jsonl
│       └── generations.jsonl
├── candidate/
│   ├── metrics.jsonl
│   └── emergence/
│       ├── steps.jsonl
│       └── generations.jsonl
└── analysis/
    ├── comparison.json      # Raw comparison statistics
    ├── comparison.md        # Human-readable summary
    ├── significance.json    # Statistical tests (if framework available)
    └── significance.md      # Significance report
```

## Phase 3: Monitoring

### W&B Follower

Stream metrics to Weights & Biases in real time while training runs:

```bash
# Follow one arm
uv run python scripts/tinker_wandb_follow.py \
    --metrics-path results/tinker_ab_v1/candidate/metrics.jsonl \
    --project textpolicy-tinker \
    --run-name "candidate-full-pipeline"

# Follow both arms (separate terminals or background jobs)
uv run python scripts/tinker_wandb_follow.py \
    --metrics-path results/tinker_ab_v1/baseline/metrics.jsonl \
    --run-name "baseline-grpo" \
    --group "ab_v1" &

uv run python scripts/tinker_wandb_follow.py \
    --metrics-path results/tinker_ab_v1/candidate/metrics.jsonl \
    --run-name "candidate-full" \
    --group "ab_v1" &
```

The follower polls the JSONL file every 5 seconds (configurable via `--poll-seconds`), backfills any existing data on startup, and exits after 10 minutes of idle (configurable via `--idle-timeout-seconds`). Use `--no-follow` for a one-shot backfill.

### Key Metrics to Watch

| Metric | What to look for |
|--------|-----------------|
| `running_correct_rate` | Primary metric. Should increase over training. |
| `loss` | Should decrease. Spikes may indicate LR too high. |
| `max_token_hit_rate` | Fraction of completions hitting the token limit. High rates (>30%) suggest `--max-tokens` is too low. |
| `sepa_lambda` | Should ramp from 0 toward 1 for the full pipeline. |
| `correct_rate` | Per-step noise; use `running_correct_rate` for trends. |
| `step_time_s` | Wall-clock time per step. Useful for estimating total runtime. |

### W&B Follower CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--metrics-path` | (required) | Path to `metrics.jsonl` |
| `--project` | `textpolicy-tinker` | W&B project name |
| `--entity` | (default) | W&B entity |
| `--run-name` | (auto) | W&B run name |
| `--group` | (none) | W&B run group |
| `--poll-seconds` | `5.0` | Polling interval |
| `--idle-timeout-seconds` | `600.0` | Exit after this many idle seconds |
| `--no-follow` | false | Backfill once and exit |

## Phase 4: Analysis

### Automatic Significance Tests

When both arms complete successfully, the campaign script automatically runs:

1. **Fisher exact test** — tests whether correct counts differ significantly between arms
2. **Bootstrap CI** — confidence interval on the difference in correct rates
3. **Permutation test** — non-parametric test on reward distributions

Results are written to `analysis/significance.json` and `analysis/significance.md`.

### Reading the Report

The significance report includes:

- **Correct rates**: per-arm counts and rates with delta
- **Fisher p-value**: if < 0.05, the difference in correctness is statistically significant
- **Bootstrap 95% CI**: if the interval excludes 0, the effect is significant
- **Recommendation**: `significant_improvement`, `not_significant`, or `significant_regression`
- **Power estimate**: expected statistical power at the observed effect size

### When Results Are Underpowered

If the analysis reports low power (< 0.8) or non-significant results with a promising delta, you need more data. Options:

1. **Increase `--max-steps`** — more training steps per arm
2. **Increase `--batch-size`** — more prompts per step (more generations)
3. **Increase `--group-size`** — more completions per prompt
4. **Run multiple campaigns** — aggregate across runs using the significance framework

Our 16-seed paired campaign (2026-02-13) observed a +0.20pp directional signal on correctness but did not reach significance (p=0.457). Power analysis estimated ~23k generations per arm for 80% power — roughly 9x the completed budget. GPU campaigns on Tinker can reach this scale.

### Running Analysis Separately

If you ran arms in parallel (separate Tinker sessions), run the significance analysis manually:

```python
from textpolicy.analysis import (
    evaluate_sepa_significance,
    build_sepa_significance_markdown,
)

report = evaluate_sepa_significance(
    baseline_run_dirs=["results/tinker_ab_v1/baseline"],
    candidate_run_dirs=["results/tinker_ab_v1/candidate"],
    alpha=0.05,
    num_resamples=20000,
    seed=0,
)
print(build_sepa_significance_markdown(report))
```

## Campaign CLI Reference

### Campaign Script (`scripts/tinker_campaign.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--campaign-root` | `results/tinker_campaign_<timestamp>` | Output directory |
| `--execute` | false | Run the campaign (default is dry-run) |
| `--continue-on-error` | false | Run candidate even if baseline fails |
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | Model ID |
| `--base-url` | (production) | Tinker service URL |
| `--lora-rank` | `64` | LoRA rank |
| `--max-steps` | `100` | Steps per arm |
| `--batch-size` | `4` | Prompts per step |
| `--group-size` | `8` | Completions per prompt |
| `--max-tokens` | `2048` | Max completion length |
| `--temperature` | `0.7` | Sampling temperature |
| `--lr` | `4e-5` | Learning rate (both arms, unless overridden) |
| `--baseline-lr` | (uses `--lr`) | Override LR for baseline arm |
| `--candidate-lr` | (uses `--lr`) | Override LR for candidate arm |
| `--max-examples` | (all) | Limit dataset size |
| `--save-every` | `20` | Checkpoint interval |
| `--gtpo-beta` | `0.1` | GTPO entropy weighting |
| `--hicra-alpha` | `0.2` | HICRA amplification |
| `--sepa-steps` | `100` | SEPA ramp steps |
| `--sepa-schedule` | `linear` | `linear` or `auto` |
| `--sepa-delay-steps` | `10` | SEPA delay |
| `--sepa-correct-rate-gate` | `0.1` | Min correct rate for SEPA |
| `--alpha` | `0.05` | Significance level |
| `--resamples` | `20000` | Permutation/bootstrap resamples |

### LR Sweep Script (`scripts/tinker_lr_sweep.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--sweep-root` | `results/lr_sweep_<timestamp>` | Output directory |
| `--probe-steps` | `20` | Steps per probe |
| `--model` | `Qwen/Qwen3-4B-Instruct-2507` | Model ID |
| `--base-url` | (production) | Tinker service URL |
| `--lora-rank` | `64` | LoRA rank |
| `--batch-size` | `4` | Prompts per step |
| `--group-size` | `16` | Completions per prompt |
| `--max-tokens` | `2048` | Max completion length |
| `--temperature` | `0.7` | Sampling temperature |
| `--lr-values` | `5e-5,2e-4,5e-4,1e-3` | Comma-separated LR values |
| `--algorithms` | `grpo,full` | Comma-separated algorithms to sweep |
| `--sepa-steps` | `100` | SEPA ramp steps (for full pipeline probes) |
| `--sepa-schedule` | `linear` | SEPA schedule |
| `--sepa-delay-steps` | `10` | SEPA delay |
| `--sepa-correct-rate-gate` | `0.1` | Min correct rate for SEPA |

## Reproducibility Checklist

To fully replicate a campaign:

- [ ] Same model ID and LoRA rank
- [ ] Same LR per arm (from sweep)
- [ ] Same batch size and group size
- [ ] Same max tokens and temperature
- [ ] Same SEPA parameters (steps, schedule, delay, gate)
- [ ] Same GTPO beta and HICRA alpha
- [ ] Same dataset (default: hendrycks/MATH, all 5 subjects)
- [ ] Same Tinker API version
- [ ] Record the `manifest.json` — it captures all of the above

The `manifest.json` written by the campaign script captures the full configuration, including the exact commands used for each arm. To replicate, copy the commands from `run_commands.sh` or pass the same flags.
