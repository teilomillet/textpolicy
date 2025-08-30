# TextPolicy Tests

Comprehensive test suite for the TextPolicy reinforcement learning framework, optimized for Apple Silicon and MLX.

## Quick Start

```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest -m unit              # Fast unit tests only
uv run pytest -m integration       # Integration tests
uv run pytest -m algorithm         # Algorithm tests (GSPO, GRPO)
uv run pytest -m reward            # Reward system tests
uv run pytest -m "not slow"        # Skip slow tests
```

## Test Categories

### 🚀 Unit Tests (`-m unit`)
Fast, isolated tests of individual components:
- Algorithm functions (GSPO, GRPO)
- Reward calculations
- Data structures
- Utilities

**Expected runtime:** < 30 seconds

### 🔄 Integration Tests (`-m integration`)  
Tests that verify component interactions:
- End-to-end training workflows
- Model + algorithm integration
- Buffer + trainer coordination

**Expected runtime:** 1-5 minutes

### 🧠 Algorithm Tests (`-m algorithm`)
Deep testing of RL algorithms:
- GSPO sequence-level importance sampling
- GRPO token-level processing
- Mathematical correctness
- Training dynamics

### 🎯 Reward Tests (`-m reward`)
Reward system functionality:
- Signature compatibility
- Batch processing
- MLX integration

### 🐌 Slow Tests (`-m slow`)
Comprehensive tests that take time:
- Full training loops
- Large model integration
- Performance benchmarks

## Useful Commands

```bash
# Development workflow - fast feedback
uv run pytest -m "unit and not slow" -x --tb=line

# Before committing - comprehensive check  
uv run pytest -m "not requires_model"

# Full test suite (if you have models)
uv run pytest

# Test specific file
uv run pytest tests/test_gspo_verification.py

# Test specific function
uv run pytest tests/test_gspo_verification.py::TestGSPOBasicFunctionality::test_sequence_importance_weights

# Run with coverage
uv run pytest --cov=textpolicy --cov-report=html

# Debug failing test
uv run pytest tests/test_gspo_verification.py::test_failing -vv --tb=long --pdb

# Performance profiling
uv run pytest --durations=0
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_gspo_verification.py # Comprehensive GSPO algorithm tests
├── test_reward_signatures.py # Reward function compatibility
└── test_rollout_rewards.py   # Rollout reward processing
```

## Writing Tests

### Test Naming Convention
- Files: `test_<module_name>.py`
- Classes: `Test<FeatureName>`  
- Methods: `test_<specific_behavior>`

### Markers Usage
```python
import pytest

@pytest.mark.unit
@pytest.mark.algorithm
def test_gspo_sequence_weights():
    """Test GSPO sequence-level importance weights."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_training_loop():
    """Test complete training workflow."""
    pass

@pytest.mark.requires_model
def test_with_real_model():
    """Test that requires loaded model."""
    pass
```

### Fixtures
Use shared fixtures from `conftest.py`:
```python
def test_with_sample_data(sample_logprobs, sample_sequence_data):
    old_logprobs = sample_logprobs['old_logprobs']
    sequence_lengths = sample_sequence_data['sequence_lengths']
    # Test logic here
```

## Apple Silicon / MLX Optimization

Tests are designed for MLX efficiency:
- Use `mx.array` instead of numpy when possible
- Test MLX-specific functionality
- Verify Apple Silicon performance characteristics

### Integration Guidelines

- Keep integration tests lightweight and deterministic. Avoid loading large models or network calls.
- Prefer `TextGenerationEnv` with a dummy tokenizer to validate rollout + buffer + strategy plumbing.
- Batch conversions to MLX where appropriate; avoid per-step Python loops that grow quadratically.
- For environments returning dict-shaped step results, rely on the runner’s normalization to remain compatible with tuple-based external envs.
- See `docs/INTEGRATION_GUIDE.md` for detailed patterns and examples.

## Continuous Integration

The test markers allow flexible CI strategies:

```bash
# Fast feedback (< 1 minute)
uv run pytest -m "unit and not slow"

# Pre-merge validation (< 5 minutes)  
uv run pytest -m "not requires_model and not slow"

# Full validation (nightly)
uv run pytest
```

## Common Issues

### MLX Array Type Errors
If you see type errors with MLX arrays:
```python
# Use type: ignore for known MLX type annotation issues
batch_size = actions.shape[0]  # type: ignore
```

### Model Loading Errors
Tests requiring models will skip gracefully:
```python
@pytest.mark.requires_model
def test_with_model():
    try:
        model = load_model("path/to/model")
    except Exception:
        pytest.skip("Model not available")
```

### Memory Issues on Large Tests
Use the `slow` marker for memory-intensive tests:
```python
@pytest.mark.slow
def test_large_batch_processing():
    # Large scale test
    pass
```

## Performance Expectations

| Test Category | Target Runtime | Purpose |
|--------------|----------------|---------|
| Unit Tests | < 30s | Fast feedback during development |
| Integration | 1-5min | Pre-commit validation |
| Algorithm | 30s-2min | Algorithm correctness |
| Reward | < 1min | Reward system validation |
| Slow Tests | 5-15min | Comprehensive validation |

## Contributing

When adding tests:
1. Use appropriate markers (`unit`, `integration`, etc.)
2. Add docstrings explaining what is being tested
3. Use fixtures for common test data
4. Follow the naming conventions
5. Keep unit tests fast (< 1s each)
6. Mark slow tests appropriately

Example:
```python
@pytest.mark.unit
@pytest.mark.algorithm  
class TestGSPONewFeature:
    """Test new GSPO feature implementation."""
    
    def test_feature_basic_functionality(self, sample_logprobs):
        """Test that new feature works with basic inputs."""
        # Fast, focused test
        pass
    
    @pytest.mark.slow
    def test_feature_comprehensive(self):
        """Test feature with comprehensive scenarios."""
        # More thorough but slower test
        pass
```
