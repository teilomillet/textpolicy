# Integrated Rollout Reward System

A comprehensive reward and verification system for efficient MLX training, designed to process rewards at the rollout level rather than per-transition for optimal performance.

## Overview

The Integrated Rollout Reward System provides:

1. **Rollout-level reward computation** - Process entire episodes in batches
2. **Quality verification** - Comprehensive text quality checks
3. **MLX optimization** - Vectorized operations and compilation
4. **Seamless integration** - Works with existing rollout and buffer systems
5. **Pure function design** - No side effects, MLX compilation compatible

## Key Components

### 1. RolloutRewardProcessor

Efficiently processes rewards for batches of episodes:

```python
from textpolicy.rewards import RewardConfig, create_rollout_reward_processor

# Configure reward weights
config = RewardConfig(
    length_weight=0.2,
    keyword_weight=0.3,
    perplexity_weight=0.3,
    accuracy_weight=0.2,
    target_length=50,
    keywords=['quantum', 'computer', 'algorithm']
)

# Create processor
processor = create_rollout_reward_processor(config)

# Process episodes
episodes = [
    {'prompt': 'Explain quantum computing', 'response': '...'},
    {'prompt': 'What is ML?', 'response': '...'}
]

rewards = processor.process_episode_rewards(episodes)
```

### 2. Verification System

Quality verification with multiple verifiers:

```python
from textpolicy.rewards import create_default_verifier_pipeline

# Create default verification pipeline
verifier = create_default_verifier_pipeline()

# Verify episodes
reports = verifier.verify_batch(prompts, responses)

# Custom verifier configuration
custom_configs = [
    {'type': 'length', 'min_length': 20, 'max_length': 200},
    {'type': 'toxicity'},
    {'type': 'coherence'},
    {'type': 'factual'}
]

verifier = create_custom_verifier_pipeline(custom_configs)
```

### 3. Integrated System

Combines rewards and verification in one system:

```python
from textpolicy.rewards import IntegratedRewardConfig, create_integrated_reward_system

# Create integrated configuration
config = IntegratedRewardConfig(
    reward_config=RewardConfig(...),
    enable_verification=True,
    verification_threshold=0.7,
    strict_filtering=False,
    min_reward_threshold=0.3
)

# Create integrated system
system = create_integrated_reward_system(config)

# Process episodes with quality control
rewards, accepted, rejected = system.process_episodes(episodes)
quality_metrics = system.get_quality_metrics(accepted + rejected)
```

## Integration with Rollout System

### Basic Integration

```python
from textpolicy.rollout import create_rollout_coordinator
from textpolicy.rewards import create_integrated_reward_system

# Create rollout coordinator
coordinator = create_rollout_coordinator(
    env_fn=lambda: MockEnvironment(),
    policy_fn=lambda: MockPolicy(),
    algorithm='ppo',
    num_workers=4
)

# Create reward system
reward_system = create_integrated_reward_system(config)

# Collect rollouts
buffer = coordinator.collect()

# Process with reward system
rewards, accepted, rejected = reward_system.process_buffer(buffer)
```

### MLX Optimization

The system automatically optimizes for MLX:

```python
# MLX-compiled reward computation
@mx.compile
def compute_reward_vector(
    response_lengths: mx.array,
    keyword_matches: mx.array,
    fluency_scores: mx.array,
    accuracy_scores: mx.array,
    weights: mx.array
) -> mx.array:
    return (
        weights[0] * response_lengths +
        weights[1] * keyword_matches +
        weights[2] * fluency_scores +
        weights[3] * accuracy_scores
    )

# Batch processing with MLX arrays
rewards = compute_reward_vector(
    length_scores, keyword_scores, fluency_scores, accuracy_scores, weights
)
```

## Configuration Options

### RewardConfig

```python
@dataclass
class RewardConfig:
    # Basic reward weights
    length_weight: float = 0.1
    keyword_weight: float = 0.2
    perplexity_weight: float = 0.3
    accuracy_weight: float = 0.4
    
    # Target parameters
    target_length: int = 50
    keywords: List[str] = None
    
    # External reward model
    external_rm_url: Optional[str] = None
    external_rm_timeout: float = 30.0
    
    # Batch processing
    batch_size: int = 32
    max_workers: int = 4
```

### IntegratedRewardConfig

```python
@dataclass
class IntegratedRewardConfig:
    # Reward configuration
    reward_config: RewardConfig
    
    # Verification configuration
    enable_verification: bool = True
    verification_threshold: float = 0.7
    strict_filtering: bool = False
    
    # Quality control
    min_reward_threshold: float = 0.3
    max_reward_threshold: float = 1.0
    
    # Batch processing
    batch_size: int = 32
    enable_mlx_compilation: bool = True
```

## Usage Patterns

### 1. Pure Function Interface

```python
from textpolicy.rewards import process_episodes_with_quality_control

# One-shot processing
rewards, accepted, rejected, metrics = process_episodes_with_quality_control(
    episodes, config
)
```

### 2. System Instance

```python
# Long-running system
system = create_integrated_reward_system(config)

try:
    # Process multiple batches
    for batch in episode_batches:
        rewards, accepted, rejected = system.process_episodes(batch)
        # Use results...
finally:
    system.close()
```

### 3. Buffer Integration

```python
# Process entire buffer
rewards, accepted, rejected = system.process_buffer(buffer)

# Extract quality metrics
metrics = system.get_quality_metrics(accepted + rejected)
```

## Performance Characteristics

### MLX Optimization

- **Vectorized operations** - Process entire batches at once
- **Compilation** - MLX functions are compiled for optimal performance
- **Memory efficiency** - Unified memory usage on Apple Silicon
- **Batch processing** - Configurable batch sizes for optimal throughput

### Throughput

Typical performance on Apple Silicon:
- **Small batches (32 episodes)**: ~100-200 episodes/second
- **Large batches (100+ episodes)**: ~200-500 episodes/second
- **MLX compilation**: 2-5x speedup after warmup

### Memory Usage

- **Efficient MLX arrays** - Minimal memory overhead
- **Batch processing** - Configurable to fit available memory
- **Garbage collection** - Automatic cleanup of temporary arrays

## Examples

See the example scripts for complete usage examples:

- `examples/integrated_reward_example.py` - Basic usage and features
- `examples/rollout_integration_example.py` - Integration with rollout system

## Best Practices

### 1. Batch Sizing

```python
# Optimal batch sizes for different scenarios
config = IntegratedRewardConfig(
    batch_size=32,  # Good for real-time processing
    # batch_size=128,  # Better for throughput
    # batch_size=512,  # Best for large datasets
)
```

### 2. Verification Thresholds

```python
# Strict quality control
config = IntegratedRewardConfig(
    verification_threshold=0.8,  # High quality bar
    strict_filtering=True,  # Both rewards and verification must pass
)

# Lenient quality control
config = IntegratedRewardConfig(
    verification_threshold=0.5,  # Lower quality bar
    strict_filtering=False,  # Either can pass
)
```

### 3. MLX Compilation

```python
# Enable MLX compilation for production
config = IntegratedRewardConfig(
    enable_mlx_compilation=True,  # Compile reward functions
    batch_size=64,  # Larger batches for compilation overhead
)
```

## Extending the System

### Custom Verifiers

```python
from textpolicy.rewards import TextVerifier

class CustomVerifier(TextVerifier):
    def verify(self, prompt: str, response: str) -> VerificationReport:
        # Custom verification logic
        score = self._compute_custom_score(prompt, response)
        
        return VerificationReport(
            result=VerificationResult.PASS if score > 0.7 else VerificationResult.FAIL,
            score=score,
            details={'custom_metric': score},
            message=f"Custom score: {score:.3f}"
        )
```

### Custom Reward Functions

```python
def custom_reward(prompt: str, response: str) -> float:
    """Custom reward function."""
    # Custom reward logic
    return score

# Add to reward processor
processor._custom_rewards = [custom_reward]
```

## Troubleshooting

### Common Issues

1. **MLX compilation errors**: Ensure all inputs are MLX arrays
2. **Memory issues**: Reduce batch size or enable garbage collection
3. **Verification failures**: Adjust thresholds or customize verifiers
4. **Performance issues**: Enable MLX compilation and optimize batch sizes

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check MLX array properties
print(f"Rewards shape: {rewards.shape}")
print(f"Rewards device: {rewards.device}")
print(f"Rewards dtype: {rewards.dtype}")
```

## Future Enhancements

- **Async reward models** - Non-blocking external API calls
- **Distributed processing** - Multi-GPU reward computation
- **Adaptive thresholds** - Dynamic quality control based on data
- **Custom metrics** - User-defined quality measures
- **Real-time monitoring** - Live quality metrics during training

