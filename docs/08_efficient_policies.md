# 08. Efficient MLX Policies (Anti-patterns and Fixes)

Intent: write policies that run fast on MLX and Apple Silicon.

Core rule: minimize Python work in hot loops; keep data on-device; batch conversions.

1) Per-token Python loops
- Anti-pattern: build responses token by token with Python loops or string concatenation.
- Fix: return complete token ID arrays; let generation happen in MLX (see `create_policy`).
- Example: examples/08_real_rl_training.py returns MLX token arrays from the policy.

2) Excessive MLX↔Python conversions
- Anti-pattern: `tolist()` and `.item()` inside inner loops.
- Fix: keep actions and observations as MLX arrays; convert once at boundaries (runner does this).
- Example: RolloutRunner batches conversions; do not convert inside your policy.

3) Ignoring deterministic parameter
- Anti-pattern: policy signature without `deterministic`, causing strategy mismatches.
- Fix: accept `deterministic=False` and pass it into sampling; strategies call `policy(obs, deterministic=...)`.
- Example: This is shown in docs/00_hello_world.md and strategy.select_action.

4) Non-contiguous arrays before MLX conversion
- Anti-pattern: slicing/transposing numpy arrays and converting to MLX without ensuring contiguity.
- Fix: use `np.ascontiguousarray` before `mx.array`; adapters already do this.
- Example: VectorizedEnvironment ensures contiguity before conversion.

5) Building large Python strings in loops
- Anti-pattern: repeated `+` concatenation to build completions.
- Fix: operate on token IDs; decode to text once per step if needed.
- Example: TextGenerationEnv decodes once per step to score reward.

6) Returning wrong shapes/dtypes
- Anti-pattern: policies returning Python lists when MLX arrays are expected, or wrong dtypes.
- Fix: return MLX arrays (e.g., `mx.int32` for token IDs) as in examples.

7) Device transfers during training
- Anti-pattern: moving data back and forth between host and device within the training step.
- Fix: keep tensors in MLX; store to buffer as Python types only when necessary.

Checklist
- [ ] Policy returns MLX arrays and accepts `deterministic`.
- [ ] No `.tolist()` in hot loops.
- [ ] Batching conversions where possible.
- [ ] Decoding text at most once per step.
- [ ] No Python per-token loops.

