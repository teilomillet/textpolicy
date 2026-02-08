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
With `micro_batch_size=N`, GRPO logprob extraction runs in chunks of
at most N episodes per forward pass. This bounds per-forward logits tensor
size and can reduce peak memory in practice, while preserving the same final
optimizer step semantics. Exact savings depend on MLX's forward/backward
scheduling for the specific model and hardware.

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

Example single-run probe (Apple M4 Pro 24 GB, Trinity-Nano-Preview, G=8):

| Configuration       | Peak Memory (GB) | Train Time (s) | Total Step Time (s) |
|---------------------|------------------|----------------|---------------------|
| Baseline            | 36.67            | 36.92          | 127.78              |
| Micro-batch M=2     | 30.10            | 28.78          | 130.94              |
| Micro-batch M=4     | 29.06            | 18.92          | 92.35               |
| GC only             | 32.14            | 22.40          | 127.94              |
| **GC + M=4**        | **24.31**        | **12.93**      | **83.56**           |

Use `experiments/profile_hardware.py` on your machine before locking defaults;
optimal settings are model- and hardware-dependent.

### Choosing Values

- **Start with `micro_batch_size=4`** — often a good balance for memory
  savings and train-phase throughput.
- **Add `gradient_checkpointing=True`** if you still hit memory limits, or
  if you need to scale to longer sequences (2048+ tokens).
- **Tune with `profile_hardware.py`** since rollout generation can dominate
  end-to-end time even when train time improves.
- Gradient checkpointing usually adds compute overhead in isolation, but can
  still improve end-to-end behavior when it prevents memory pressure.

#### Selective checkpointing (sqrt(n) strategy)

By default, `gradient_checkpointing=True` uses the **sqrt(n) strategy** from
Chen et al. 2016 — only every sqrt(n)-th layer is checkpointed rather than
all layers.  For a 56-layer model (Trinity-Nano) this means checkpointing 8
layers instead of 56 (stride=7).  Gradients are mathematically identical
regardless of which layers are checkpointed.

##### Checkpointing strategy comparison

Benchmarked on Apple M4 Pro 24 GB, Trinity-Nano-Preview (56 layers), G=8,
LoRA rank=2 on 4 layers, `compile_training=False`:

| Strategy | Layers Ckpt | seq=256 Train (s) | seq=512 Train (s) | seq=512 Peak (GB) |
|---|---|---|---|---|
| None (`False`) | 0/56 | 3.74 | 10.37 | 24.51 |
| **sqrt(n) (`True`)** | **8/56** | **3.46 (-7.6%)** | **6.40 (-38.3%)** | **24.51** |
| Every layer (`1`) | 56/56 | 2.28 (-39.2%) | 5.24 (-49.5%) | 24.24 (-1.1%) |

On this model/hardware combination, both strategies improve training time
because reduced activation memory alleviates memory-bandwidth pressure on
Apple Silicon unified memory. The sqrt(n) strategy provides the most
conservative checkpointing (fewer recomputed layers) while still delivering
substantial speedups at longer sequences. Every-layer checkpointing maximizes
both time and memory savings but recomputes more activations.

Reproduce with: `uv run python experiments/benchmark_checkpointing.py`

To force every-layer checkpointing, pass an explicit integer stride of `1`:

```python
trainer = Trainer(
    ...,
    gradient_checkpointing=1,   # every layer (max memory savings)
)
```

Or use a custom stride (e.g. every 4th layer):

```python
trainer = Trainer(
    ...,
    gradient_checkpointing=4,   # every 4th layer
)
```
