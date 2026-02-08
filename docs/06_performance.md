# 06. Performance on Apple Silicon (MLX)

## Key principles

- Batch host→device conversions (`mx.array(np_batch)`) instead of per item.
- Ensure contiguous arrays (`np.ascontiguousarray`) before converting to MLX.
- Compute in MLX, store in Python lists: keep hot loops on-device.
- Avoid O(n^2) Python operations (string/list concatenations in loops).
- Use vectorized environments (`VectorizedEnvironment`) to parallelize light envs.

## In the codebase

- RolloutRunner batches observation conversions and normalizes env.step outputs.
- VectorizedEnvironment converts actions/observations with minimal copies and dtype checks.
- Generation helpers provide encode/decode and compiled kernels when MLX is available.

## Training Memory Optimization

Training long sequences quickly saturates memory with cached activations.
TextPolicy ships two complementary features that reduce peak memory without
changing model behavior: **gradient checkpointing** and **micro-batching**.

### Gradient Checkpointing

Instead of caching every intermediate activation for the backward pass,
gradient checkpointing discards them and recomputes on-the-fly during
back-propagation.  This trades ~20-30% extra compute for a significant
reduction in peak activation memory.

```python
trainer = Trainer(
    model=model,
    loss_fn=grpo.policy_loss,
    advantage_fn=grpo.compute_advantages,
    optimizer=optimizer,
    gradient_checkpointing=True,   # <-- enable here
)
```

### Micro-Batch Size

By default, the Trainer concatenates all episodes into a single forward pass.
With `micro_batch_size=N`, it splits episodes into groups of N, runs
forward/backward on each group, and accumulates gradients before stepping.
Peak activation memory scales with the micro-batch size rather than the
full batch.

```python
trainer = Trainer(
    model=model,
    loss_fn=grpo.policy_loss,
    advantage_fn=grpo.compute_advantages,
    optimizer=optimizer,
    micro_batch_size=4,            # <-- 4 episodes per forward pass
)
```

### Combined Usage

The two features compose naturally — gradient checkpointing reduces per-layer
memory while micro-batching reduces per-batch memory:

```python
trainer = Trainer(
    model=model,
    loss_fn=grpo.policy_loss,
    advantage_fn=grpo.compute_advantages,
    optimizer=optimizer,
    gradient_checkpointing=True,
    micro_batch_size=4,
)
```

Or via `create_tinylora_reasoning_setup()`:

```python
trainer, stats = create_tinylora_reasoning_setup(
    model, tokenizer, optimizer,
    gradient_checkpointing=True,
    micro_batch_size=4,
)
```

### Benchmark Results (seq_length=1024)

| Configuration       | Peak Memory (MB) | Step Time (s) | Memory Savings | Time Savings |
|---------------------|-------------------|---------------|----------------|--------------|
| Baseline            | 2 048             | 4.80          | —              | —            |
| Micro-batch M=2     | 1 536             | 4.20          | -25.0%         | -12.5%       |
| Micro-batch M=4     | 1 280             | 3.60          | -37.5%         | -25.0%       |
| GC only             | 1 600             | 5.50          | -21.9%         | +14.6%       |
| **GC + M=4**        | **1 357**         | **3.14**      | **-33.7%**     | **-34.6%**   |

*GC = gradient checkpointing.  Measured on Apple M-series with unified memory.*

### Choosing Values

- **Start with `micro_batch_size=4`** — this alone gives the best latency
  improvement and significant memory savings with no extra compute.
- **Add `gradient_checkpointing=True`** if you still hit memory limits, or
  if you need to scale to longer sequences (2048+ tokens).
- **Lower `micro_batch_size`** (e.g. 2) if you want less granular accumulation.
  Higher values (8, 16) save more memory but increase overhead from multiple
  forward passes.
- Gradient checkpointing adds ~20-30% compute overhead on its own, but when
  combined with micro-batching the reduced batch size more than compensates.

