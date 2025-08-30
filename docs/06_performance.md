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

