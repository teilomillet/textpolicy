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
