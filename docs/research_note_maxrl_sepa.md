# Why MaxRL Alone Is Not Enough — and Why SEPA Completes It

**Date**: 2026-02-14

**Context**: Tajwar et al. (2026), "Maximum Likelihood Reinforcement Learning" (arXiv:2602.02710)

---

## The Paper's Contribution

Tajwar et al. make one clean observation: standard RL (GRPO, RLOO, REINFORCE) optimizes only the first-order term of the maximum likelihood objective. The ML gradient decomposes as a harmonic sum over pass@k:

```
nabla J_ML(x) = sum_{k=1}^{inf} (1/k) nabla pass@k(x)
```

RL keeps only the k=1 term. Everything else is thrown away.

MaxRL recovers the full sum by a single-line change to the advantage calculation: normalize by K (number of successes in the batch) instead of N (total rollouts). This gives hard problems — where K is small — gradient proportional to 1/p, which is exactly the inverse-probability reweighting that makes ML the principled objective in supervised learning with binary correctness.

The result: MaxRL Pareto-dominates GRPO on math reasoning, with up to 20x test-time scaling efficiency gains. More compute during training actually improves the objective itself (higher-order ML approximation), not just variance reduction.

This is correct and useful. But it's incomplete.

---

## What the Paper Misses

The entire MaxRL framework operates at the **prompt level**. A problem either gets solved (reward 1) or doesn't (reward 0). The 1/p reweighting decides *which problems* get gradient. But once a problem is selected, every token in the successful rollout receives the same advantage signal.

The paper's formalism makes this explicit. The model generates a "latent variable z" — the full sequence — and success is evaluated on the decoded output y = f(z). The gradient estimator averages score functions uniformly across all tokens in successful trajectories:

```
g_hat = (1/K) sum_{i: r_i=1} nabla log m_theta(z_i | x)
```

There is no mechanism to distinguish which tokens within z_i actually contributed to the success.

This matters. In a reasoning trace, most tokens are execution — routine continuation that any competent model would produce. The actual reasoning happens at specific points: the moment of hesitation, the backtrack, the structural connection. Treating all tokens equally dilutes the learning signal from the tokens that matter into the noise of the tokens that don't.

---

## SEPA Addresses the Missing Level

SEPA (Selective Entropy Pooling with Annealing) operates at the **token level**, which is exactly where MaxRL is silent. The insight is that token entropy is a natural signal for where the model is "thinking" versus "executing":

- **Planning tokens** have high entropy — the model is uncertain, exploring possibilities
- **Execution tokens** have low entropy — routine continuation, not a real decision

But raw entropy is noisy. Execution tokens have low entropy for boring reasons (they're predictable), not because they represent confident decisions. GTPO's multiplicative weighting would amplify all high-entropy tokens equally, including noisy ones.

SEPA fixes this by pooling execution-token entropy toward the group mean:

```
H_pooled = lambda * mean(H_exec) + (1 - lambda) * H_original
```

Planning tokens are left untouched. The effect: execution entropy variance drops, GTPO weighting concentrates more meaningfully on actual planning tokens, and the credit assignment becomes structurally informed rather than uniformly spread.

---

## The Composition

MaxRL and SEPA are orthogonal transforms that compose cleanly:

| Layer | What it decides | Mechanism |
|-------|----------------|-----------|
| **MaxRL** | Which prompts get gradient | Episode-level 1/p reweighting |
| **GTPO** | Which tokens get credit | Entropy-based multiplicative weighting |
| **SEPA** | Clean the entropy signal | Pool execution noise, preserve planning signal |
| **HICRA** | Amplify planning tokens | Strategic phrase detection + amplification |

The pipeline: MaxRL computes episode-level advantages with proper ML reweighting. Those advantages expand to token-level via repeat. SEPA pools execution entropy. GTPO reweights by the cleaned entropy. HICRA further amplifies identified planning tokens.

Each layer addresses a different failure mode:
- Without MaxRL: easy problems steal gradient from hard ones (RL's uniform weighting)
- Without GTPO: all tokens in a success get equal credit (MaxRL's token blindness)
- Without SEPA: execution tokens corrupt the entropy signal (GTPO's noise sensitivity)
- Without HICRA: planning tokens get weighted but not amplified (no structural prior)

---

## Why This Matters More Than the Paper Suggests

Tajwar et al. frame the problem as "RL vs ML objective." That's a clean theoretical contribution. But in practice, the bottleneck is not just *which problems* get gradient — it's *which tokens within a solution* get credit.

Consider: a model solves a hard math problem with pass rate p = 0.01. MaxRL correctly gives this problem 100x the gradient weight of a problem with p = 1.0. Good. But within that rare successful rollout, the model wrote 200 tokens. Maybe 5 of those tokens were the actual insight — the moment it chose the right strategy, or caught an error, or made the structural connection. The other 195 tokens were arithmetic and formatting.

MaxRL gives all 200 tokens the same 100x-weighted advantage. SEPA + GTPO + HICRA concentrates that weight where it belongs.

The paper's Theorem 1 says the ML gradient is "the average score function over successful trajectories only." Average over trajectories, yes. But uniform within each trajectory. That's where the approximation still lives, and that's what token-level credit assignment resolves.

---

## Connection to Reasoning Research

This connects to a deeper question from the reri work: can language models reason, or do they only appear to?

If reasoning happens at specific tokens — the planning moments, the structural transfers — then training signal that treats all tokens equally is training signal that doesn't differentially reinforce reasoning. It reinforces the whole trajectory, reasoning and routine alike.

A training framework that explicitly identifies and amplifies the tokens where reasoning occurs is, in a sense, the training-side analog of what reri tests on the evaluation side: can we isolate and measure the specific capability of structural transfer, separate from the surrounding execution?

MaxRL gets the objective right. SEPA gets the credit assignment right. Together they address both the "what to optimize" and the "where to attribute" questions that any principled approach to training reasoning needs to answer.

---

## Current Evidence

The 16-seed paired campaign (2026-02-13) on MaxRL+SEPA vs MaxRL+HICRA showed a directional positive signal on correctness rate (+0.20 pp) but did not reach statistical significance (p=0.457). Power analysis estimates ~23k generations per arm for 80% power at the observed effect size, roughly 9x the completed budget.

The implementation is stable, instrumented, and reproducible. The theoretical argument is clear. The empirical validation needs more compute.

---

## References

- Tajwar, F. et al. (2026). Maximum Likelihood Reinforcement Learning. arXiv:2602.02710.
- Project: https://zanette-labs.github.io/MaxRL/
- Code: https://github.com/tajwarfahim/maxrl
