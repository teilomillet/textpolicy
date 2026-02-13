# 11. SEPA Campaign and Significance

This runbook prepares and executes a paired multi-seed SEPA-vs-HICRA campaign, then emits:

- rule-based litmus verdict (`analysis/litmus.json`, `analysis/litmus.md`)
- statistical report (`analysis/significance.json`, `analysis/significance.md`)

## 1) Prepare (dry-run)

```bash
uv run python scripts/sepa_campaign.py \
  --campaign-root results/sepa_campaign_2026_02_11 \
  --seeds 101,102,103,104,105,106,107,108 \
  --model arcee-ai/Trinity-Nano-Preview \
  --advantage-mode grpo \
  --steps 12 \
  --num-problems 16 \
  --episodes-per-step 8 \
  --batch-size 8 \
  --max-tokens 128 \
  --temperature 0.4 \
  --candidate-sepa-steps 8 \
  --candidate-sepa-schedule linear \
  --wandb-offline
```

This writes:

- `results/sepa_campaign_2026_02_11/run_commands.sh`
- `results/sepa_campaign_2026_02_11/manifest.json` (planned commands)

## 2) Execute

```bash
uv run python scripts/sepa_campaign.py \
  --campaign-root results/sepa_campaign_2026_02_11 \
  --seeds 101,102,103,104,105,106,107,108 \
  --model arcee-ai/Trinity-Nano-Preview \
  --advantage-mode grpo \
  --steps 12 \
  --num-problems 16 \
  --episodes-per-step 8 \
  --batch-size 8 \
  --max-tokens 128 \
  --temperature 0.4 \
  --candidate-sepa-steps 8 \
  --candidate-sepa-schedule linear \
  --wandb-offline \
  --execute
```

## 3) Read Results

- campaign manifest: `<campaign-root>/manifest.json`
- litmus: `<campaign-root>/analysis/litmus.md`
- significance: `<campaign-root>/analysis/significance.md`

If significance is still weak, increase paired seeds and rerun with the same recipe.

## 4) Run the full 2x2 (GRPO vs MaxRL)

Use two campaign roots so hypotheses do not get mixed:

- GRPO pair (A/B): `--advantage-mode grpo`
- MaxRL pair (C/D): `--advantage-mode maxrl`

Example MaxRL campaign:

```bash
uv run python scripts/sepa_campaign.py \
  --campaign-root results/sepa_campaign_2026_02_11_maxrl \
  --seeds 101,102,103,104,105,106,107,108 \
  --model arcee-ai/Trinity-Nano-Preview \
  --advantage-mode maxrl \
  --steps 12 \
  --num-problems 16 \
  --episodes-per-step 8 \
  --batch-size 8 \
  --max-tokens 128 \
  --temperature 0.4 \
  --candidate-sepa-steps 8 \
  --candidate-sepa-schedule linear \
  --wandb-offline \
  --execute
```

## Seed-601 Smoke Test Note (Funding Ask)

On February 13, 2026, we ran a matched one-seed smoke test on `arcee-ai/Trinity-Nano-Preview` with the same budget in both arms (`20` steps, `8` episodes/step, `160` episodes/arm, seed `601`) and with the corrected MaxRL plumbing (prompt-level advantages from explicit binary correctness, with token-level shaping in the GTPO path).

The continuous metrics showed a consistent directional pattern across correlated endpoints, but the primary metric (correctness) requires approximately 15x more compute to resolve at conventional significance thresholds.

The observed correctness counts were identical (`3/160` for MaxRL+HICRA and `3/160` for MaxRL+SEPA), so the primary endpoint is non-significant in this smoke pass (`p = 1.0`, two-proportion test). The directional continuous changes were: mean reward `-0.1469 -> -0.1219`, final reward `-0.125 -> 0.000`, mean planning-token ratio `0.01743 -> 0.01834`, final planning-token ratio `0.01465 -> 0.02344`, and mean gram-entropy delta `-0.00698 -> 0.03278`.

The power analysis below uses a standard two-sample proportion design (`alpha = 0.05`, two-sided; power `= 0.80`; equal arm sizes) with baseline pass rate fixed to the smoke estimate (`p0 = 0.01875`). Under these assumptions, a modest absolute lift around `+0.0126` requires about `2414` episodes per arm, which is `15.1x` the current `160` episodes per arm. The assumed lift range (`+1.0` to `+2.0` percentage points) is not empirically calibrated and is used as a minimum practically meaningful effect size; if the true lift is smaller, required episodes increase materially.

| Assumed absolute lift in correctness | Alternative pass rate | Required episodes per arm | Multiplier vs smoke budget |
| --- | --- | ---: | ---: |
| +0.0100 | 0.02875 | 3639 | 22.7x |
| +0.0126 | 0.03135 | 2414 | 15.1x |
| +0.0150 | 0.03375 | 1782 | 11.1x |
| +0.0200 | 0.03875 | 1095 | 6.8x |

### Caveats

1. This is `N=1` seed; continuous metrics are correlated, so directional consistency is suggestive but not statistically meaningful (order-of-magnitude chance baseline is about `1/8` if reduced to ~3 independent metric families).
2. The effect-size assumptions in the power table are planning assumptions, not calibrated priors.
3. The smoke test used MaxRL prompt-level normalization; the funded campaign must use the same normalization (`--advantage-mode maxrl`) to test the same hypothesis.
4. The Trinity tokenizer regex issue is now handled in `/Users/teilomillet/Code/textpolicy/textpolicy/generation/mlx_generation.py` by applying the Mistral regex fix post-load; campaign reports should still note tokenizer/toolchain versions for reproducibility.
