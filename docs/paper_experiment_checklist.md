# Paper Experiment Checklist

**Status**: Pre-registration (experiments not yet run at sufficient scale)
**Date**: 2026-02-14

---

## Scope of the paper

**We claim**: Token-level credit assignment (SEPA) is an orthogonal and complementary axis to episode-level objective selection (MaxRL). Combining them yields gains that neither achieves alone.

**We do NOT claim**:
- That SEPA beats SOTA on any benchmark
- That MaxRL is our contribution (it's Tajwar et al.'s)
- That GTPO or HICRA are our contribution (existing work)
- That results generalize beyond the model/task we test on
- That larger models would show the same pattern

---

## Hypotheses

Each hypothesis maps to exactly one experiment. Each experiment has a pre-registered comparison, metric, and success criterion. We only report claims supported by the experiment that tests them.

### H1: MaxRL improves over GRPO (replication)

**What we test**: Tajwar et al.'s claim holds in our setup.
**Why we need it**: Establishes the episode-level baseline improvement before we add token-level transforms. If this fails, the paper's framing doesn't work.

| | |
|---|---|
| **Comparison** | GRPO alone vs MaxRL alone (no token transforms) |
| **Primary metric** | Correctness rate (Fisher exact test) |
| **Secondary metrics** | Mean reward, pass@k if available |
| **Success criterion** | MaxRL correctness >= GRPO correctness, directionally consistent across seeds |
| **Failure plan** | If MaxRL does not improve over GRPO at our scale, we report this honestly and reframe: the paper becomes about token-level credit assignment only, comparing GTPO+HICRA vs GTPO+SEPA |

**Method matrix cells**: `grpo_alone` vs `maxrl_alone`

---

### H2: GTPO adds value over flat advantages

**What we test**: Entropy-weighted token-level credit assignment improves over uniform advantage expansion.
**Why we need it**: Establishes that token-level differentiation matters at all. If flat advantages work equally well, neither HICRA nor SEPA has a reason to exist.

| | |
|---|---|
| **Comparison** | GRPO alone (flat) vs GRPO+GTPO (entropy-weighted, no planning mask) |
| **Primary metric** | Correctness rate |
| **Secondary metrics** | Mean reward, mean completion length |
| **Success criterion** | GTPO correctness >= flat correctness, directionally |
| **Failure plan** | If GTPO doesn't help, the paper's token-level story is weakened. We report it and note that the task or scale may be insufficient |

**Method matrix cells**: `grpo_alone` vs `gtpo`

---

### H3: SEPA improves over HICRA boost on top of GTPO

**What we test**: Pooling execution entropy (SEPA) produces better credit assignment than boosting planning entropy (HICRA), when both use the same planning mask on top of GTPO.
**Why we need it**: This is the core novel claim. Same input signal (planning mask), same downstream (GTPO weighting), different mechanism for how the entropy is shaped.

| | |
|---|---|
| **Comparison** | GRPO+GTPO+HICRA vs GRPO+GTPO+SEPA |
| **Primary metric** | Correctness rate |
| **Secondary metrics** | Mean reward, execution entropy variance (should decrease under SEPA), planning token gradient concentration |
| **Success criterion** | SEPA correctness >= HICRA correctness AND execution entropy variance is measurably lower under SEPA |
| **Failure plan** | If correctness is equal but entropy variance is lower, we claim SEPA is a cleaner mechanism without performance gain. If HICRA wins, we report it honestly |

**Method matrix cells**: `grpo_hicra` vs `grpo_sepa`

**Important**: This is the comparison a reviewer will scrutinize most. Needs the most seeds.

---

### H4: MaxRL and SEPA compose (gains are not redundant)

**What we test**: Adding SEPA to MaxRL produces gains beyond what either achieves alone. The two transforms operate on different levels and their benefits add up.
**Why we need it**: This is the paper's structural argument — episode-level and token-level are orthogonal.

| | |
|---|---|
| **Comparison** | 4-way: GRPO alone, MaxRL alone, GRPO+GTPO+SEPA, MaxRL+GTPO+SEPA |
| **Primary metric** | Correctness rate |
| **Test** | Interaction test: is gain(MaxRL+SEPA over GRPO) >= gain(MaxRL alone over GRPO) + gain(SEPA alone over GRPO)? At minimum, MaxRL+SEPA should be the best of the four |
| **Success criterion** | MaxRL+GTPO+SEPA is ranked #1 in correctness across the 4 conditions |
| **Failure plan** | If MaxRL+SEPA is not #1 but both MaxRL and SEPA individually help, we report that they compose but with diminishing returns. If one blocks the other, we investigate and report |

**Method matrix cells**: `grpo_alone`, `maxrl_alone`, `grpo_sepa`, `maxrl_sepa`

**Note**: `maxrl_sepa` is not currently in the method matrix script. Needs to be added.

---

### H5: SEPA measurably changes the entropy distribution (mechanistic)

**What we test**: SEPA reduces execution-token entropy variance while preserving planning-token entropy. This is a mechanistic claim independent of correctness.
**Why we need it**: Even if correctness gains are small, this proves the mechanism works as designed. A reviewer who sees the entropy shift is harder to dismiss than one who only sees a small p-value.

| | |
|---|---|
| **Comparison** | Same runs as H3 (GRPO+GTPO+HICRA vs GRPO+GTPO+SEPA), analyzed for entropy distribution |
| **Metrics** | (a) Execution-token entropy variance (should be lower under SEPA), (b) Planning-token entropy mean (should be unchanged), (c) KL divergence between planning and execution entropy distributions (should be higher under SEPA = cleaner separation) |
| **Success criterion** | Execution entropy variance is significantly lower under SEPA, planning entropy is not significantly different |
| **Failure plan** | If variance doesn't decrease, SEPA's mechanism is not working as designed. Debug before publishing |

**Data source**: Token-level entropy logs from emergence logger. May need to add per-token entropy logging if not already captured.

---

## Experiment design

### Conditions (8 cells)

| ID | Episode-level | Token-level | Matrix method name |
|---|---|---|---|
| C1 | GRPO | None (flat) | `grpo_alone` |
| C2 | GRPO | GTPO only | `gtpo` |
| C3 | GRPO | GTPO + HICRA boost | `grpo_hicra` |
| C4 | GRPO | GTPO + SEPA | `grpo_sepa` |
| C5 | MaxRL | None (flat) | `maxrl_alone` |
| C6 | MaxRL | GTPO only | `maxrl_gtpo` **(new)** |
| C7 | MaxRL | GTPO + HICRA boost | `maxrl_hicra` **(new)** |
| C8 | MaxRL | GTPO + SEPA | `maxrl_sepa` **(new)** |

### Hypothesis-to-cell mapping

| Hypothesis | Cells compared | What it isolates |
|---|---|---|
| H1 | C1 vs C5 | Episode-level objective (MaxRL vs GRPO) |
| H2 | C1 vs C2 | Token-level entropy signal (GTPO vs flat) |
| H3 | C3 vs C4 | SEPA vs HICRA (same GTPO, same planning mask) |
| H4 | C1, C5, C4, C8 | Orthogonal composition |
| H5 | C3 vs C4 (entropy logs) | Mechanistic entropy shift |

### Budget per condition

Based on the 16-seed campaign power analysis (p=0.457 at 2,560 generations/arm):

| Parameter | Minimum viable | Target |
|---|---|---|
| Seeds | 8 | 16 |
| Steps per run | 20 | 40 |
| Episodes per step | 8 | 8 |
| Problems per step | 24 | 24 |
| Max tokens | 4096 | 4096 |
| Generations per cell | ~3,840 | ~7,680 |
| Total generations (8 cells) | ~30,720 | ~61,440 |

The power analysis from the Feb 13 campaign estimated ~23k generations per arm for 80% power at the observed effect size. Target budget approaches this for the critical H3 comparison. If effect sizes are larger than the preliminary signal, fewer generations may suffice.

### What to log per run

Already logged (via emergence logger):
- [x] Mean reward per step
- [x] Correct count / total count
- [x] Planning token ratio
- [x] Strategic gram entropy delta
- [x] SEPA lambda per step
- [x] Mean completion length
- [x] Max tokens hit rate

Needed for H5 (check if available):
- [ ] Per-token entropy histogram (execution vs planning)
- [ ] Execution entropy variance per step
- [ ] Planning entropy mean per step

---

## Infrastructure changes needed

### 1. Add missing method matrix conditions

The current `reasoning_method_matrix.py` has 5 methods: `grpo_alone`, `grpo_hicra`, `grpo_sepa`, `gtpo`, `maxrl_alone`.

Need to add:
- `maxrl_gtpo` — MaxRL + GTPO (no planning mask)
- `maxrl_hicra` — MaxRL + GTPO + HICRA boost
- `maxrl_sepa` — MaxRL + GTPO + SEPA

These require `--maxrl` flag combined with the existing transform modes.

### 2. Entropy distribution logging for H5

Check whether token-level entropy is already written to disk or only used transiently. If transient, add a step-level summary to the emergence logger:
- `exec_entropy_var`: variance of execution-token entropies
- `exec_entropy_mean`: mean of execution-token entropies
- `plan_entropy_mean`: mean of planning-token entropies
- `plan_entropy_var`: variance of planning-token entropies

### 3. Analysis script for the 8-cell matrix

Extend pairwise analysis to support:
- Any pair comparison (not just vs grpo_alone)
- The H4 interaction test
- Entropy distribution comparison for H5

---

## Pre-registered analysis plan

1. Run all 8 conditions with identical seeds and budget
2. For each hypothesis, compute the pre-registered comparison:
   - H1, H2, H3: Fisher exact test on correctness rate, permutation test on mean reward
   - H4: Rank the 4 cells; report whether MaxRL+SEPA is #1
   - H5: Two-sample t-test on execution entropy variance across steps
3. Report all results regardless of direction
4. If a hypothesis fails, do not reframe the failed comparison as "exploratory" post-hoc. Report it as a failed hypothesis
5. Compute confidence intervals for all effect sizes
6. Report power analysis for any non-significant result

---

## Figures for the paper

| Figure | Data source | What it shows |
|---|---|---|
| Fig 1: Schematic | Diagram | The 4-layer stack (MaxRL > GTPO > SEPA > HICRA) and what each addresses |
| Fig 2: 8-cell correctness | H1-H4 experiments | Bar chart or table, 8 conditions, correctness rate with CI |
| Fig 3: Entropy distributions | H5 experiment | Histograms of token entropy under HICRA vs SEPA, split by planning/execution |
| Fig 4: Learning curves | All conditions | Reward over training steps, showing training dynamics differ |
| Fig 5: Weight function | Analytical | MaxRL's 1/p weighting vs GRPO (from Tajwar et al., with attribution) |

---

## What we do NOT test (and should not claim)

- [ ] Generalization to other tasks (only countdown reasoning)
- [ ] Generalization to other models (only Trinity-Nano-Preview on MLX)
- [ ] Scaling behavior (we test one budget, not a compute curve)
- [ ] Comparison with other token-level methods (process reward models, etc.)
- [ ] Whether planning tokens detected by n-gram matching are "real" planning
- [ ] Whether the model is "actually reasoning" (that's reri's question, not this paper's)

If a reviewer asks about any of these, the answer is "out of scope, noted as future work."

---

## Run order

1. **Smoke test** (1 seed, 2 steps, all 8 conditions) — verify all conditions run without error
2. **Pilot** (4 seeds, 10 steps, all 8 conditions) — check that signal is non-zero
3. **Full campaign** (8-16 seeds, 20-40 steps, all 8 conditions) — the real experiment
4. **Analysis** — run pre-registered comparisons, generate figures
5. **Write** — only after results are in
