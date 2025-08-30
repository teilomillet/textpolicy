# textpolicy/training/metrics.py
"""
Training metrics collection and analysis for all RL algorithms.
"""

from typing import Dict, Any, Optional, cast
import mlx.core as mx # type: ignore
from collections import defaultdict, deque


class TrainingMetrics:
    """
    Lightweight metrics collector optimized for MLX training.
    
    Tracks algorithm-agnostic and algorithm-specific metrics
    with minimal overhead during training.
    """
    
    def __init__(self, history_length: int = 100):
        """
        Initialize metrics collector.
        
        Args:
            history_length: Number of recent values to keep for rolling averages
        """
        self.history_length = history_length
        self.metrics = defaultdict(lambda: deque(maxlen=history_length))
        self.total_steps = 0
        
    def update(self, metrics_dict: Dict[str, float]):
        """
        Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metric_name -> value
        """
        for name, value in metrics_dict.items():
            self.metrics[name].append(value)
        
        if 'step' in metrics_dict:
            self.total_steps = metrics_dict['step']
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if metric_name in self.metrics and self.metrics[metric_name]:
            return self.metrics[metric_name][-1]
        return None
    
    def get_mean(self, metric_name: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get mean of recent metric values.
        
        Args:
            metric_name: Name of the metric
            last_n: Number of recent values to average (None for all)
            
        Returns:
            Mean value or None if metric doesn't exist
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = list(self.metrics[metric_name])
        if last_n is not None:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns:
            Dictionary with latest, mean, and other statistics
        """
        summary = {
            'total_steps': self.total_steps,
            'metrics': {}
        }
        
        for metric_name, values in self.metrics.items():
            if not values:
                continue
                
            values_list = list(values)
            summary['metrics'][metric_name] = {
                'latest': values_list[-1],
                'mean': sum(values_list) / len(values_list),
                'min': min(values_list),
                'max': max(values_list),
                'count': len(values_list)
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.total_steps = 0
    
    def __len__(self) -> int:
        """Return number of metrics being tracked."""
        return len(self.metrics)


class RolloutMetrics:
    """
    Metrics specific to rollout collection phase.
    """
    
    def __init__(self):
        self.episodes_collected = 0
        self.total_reward = 0.0
        self.episode_lengths = []
        self.episode_rewards = []
    
    def add_episode(self, reward: float, length: int):
        """Add metrics from a completed episode."""
        self.episodes_collected += 1
        self.total_reward += reward
        self.episode_lengths.append(length)
        self.episode_rewards.append(reward)
    
    def get_summary(self) -> Dict[str, float]:
        """Get rollout metrics summary."""
        if not self.episode_rewards:
            return {
                'episodes_collected': 0,
                'mean_reward': 0.0,
                'mean_length': 0.0,
                'total_reward': 0.0
            }
        
        return {
            'episodes_collected': self.episodes_collected,
            'mean_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'mean_length': sum(self.episode_lengths) / len(self.episode_lengths),
            'total_reward': self.total_reward,
            'min_reward': min(self.episode_rewards),
            'max_reward': max(self.episode_rewards),
            'min_length': min(self.episode_lengths),
            'max_length': max(self.episode_lengths)
        }
    
    def reset(self):
        """Reset rollout metrics."""
        self.episodes_collected = 0
        self.total_reward = 0.0
        self.episode_lengths.clear()
        self.episode_rewards.clear()


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    logger: Optional[Any] = None,
    prefix: str = ""
):
    """
    Log metrics to console and optionally to external logger.
    
    Args:
        metrics: Metrics dictionary
        step: Training step number
        logger: Optional external logger (wandb, tensorboard, etc.)
        prefix: Prefix for metric names
    """
    # Console logging
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Step {step} | {prefix}{metrics_str}")
    
    # External logger
    if logger is not None and hasattr(logger, 'log'):
        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        logger.log(prefixed_metrics, step=step)


def compute_explained_variance(predicted: mx.array, targets: mx.array) -> float:
    """
    Compute explained variance for value function evaluation.
    
    Explained variance = 1 - Var(targets - predicted) / Var(targets)
    
    Args:
        predicted: Predicted values
        targets: Target values
        
    Returns:
        Explained variance (1.0 = perfect, 0.0 = no better than mean)
    """
    target_var = mx.var(targets)
    if target_var == 0:
        return 0.0
    
    residual_var = mx.var(targets - predicted)
    explained_var = 1.0 - (residual_var / target_var)
    
    return cast(float, explained_var.item())  # MLX scalar array .item() returns Python float for float dtypes


def compute_policy_metrics(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    clip_ratio: float = 0.2
) -> Dict[str, float]:
    """
    Compute standard policy optimization metrics.
    
    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy
        clip_ratio: Clipping ratio used in loss
        
    Returns:
        Dictionary of policy metrics
    """
    # Importance ratio
    ratio = mx.exp(new_logprobs - old_logprobs)
    
    # Clipping statistics
    clip_lower = 1 - clip_ratio
    clip_upper = 1 + clip_ratio
    clipped = (ratio < clip_lower) | (ratio > clip_upper)
    clip_fraction = mx.mean(clipped.astype(mx.float32))
    
    # KL divergence approximation
    kl_div = mx.mean(old_logprobs - new_logprobs)
    
    # Policy change magnitude
    ratio_mean = mx.mean(ratio)
    ratio_std = mx.std(ratio)
    
    # Policy entropy (negative log probability mean)
    entropy_mean = mx.mean(new_logprobs)
    
    return {
        'policy/ratio_mean': cast(float, ratio_mean.item()),
        'policy/ratio_std': cast(float, ratio_std.item()),
        'policy/clip_fraction': cast(float, clip_fraction.item()),
        'policy/kl_divergence': cast(float, kl_div.item()),
        'policy/entropy': -cast(float, entropy_mean.item())  # Negative entropy: -E[log(p)]
    }