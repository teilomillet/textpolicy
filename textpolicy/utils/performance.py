"""
Performance monitoring and optimization utilities for MLX-RL.

This module provides real-time performance monitoring, bottleneck detection,
and optimization recommendations for training pipelines.
"""

import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from contextlib import contextmanager
from enum import Enum

from .memory import MemoryMonitor, get_memory_stats
from .debug import debug_print


class PerformanceCategory(Enum):
    """Categories for performance tracking."""
    ENVIRONMENT = "environment"
    POLICY = "policy"
    TRAINING = "training"
    DATA_LOADING = "data_loading"
    LOGGING = "logging"
    MEMORY = "memory"
    OVERALL = "overall"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a specific operation."""
    name: str
    category: PerformanceCategory
    total_time: float = 0.0
    call_count: int = 0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    def update(self, duration: float, memory_stats: Optional[Dict[str, float]] = None):
        """Update metrics with new timing data."""
        self.total_time += duration
        self.call_count += 1
        self.avg_time = self.total_time / self.call_count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.recent_times.append(duration)
        
        if memory_stats:
            self.memory_usage.update(memory_stats)
    
    @property
    def recent_avg_time(self) -> float:
        """Average time for recent calls."""
        if not self.recent_times:
            return 0.0
        return statistics.mean(self.recent_times)
    
    @property
    def calls_per_second(self) -> float:
        """Number of calls per second based on recent data."""
        if not self.recent_times or len(self.recent_times) < 2:
            return 0.0
        total_recent_time = sum(self.recent_times)
        return len(self.recent_times) / total_recent_time if total_recent_time > 0 else 0.0


class PerformanceAlert(Enum):
    """Types of performance alerts."""
    SLOW_OPERATION = "slow_operation"
    MEMORY_HIGH = "memory_high"
    THROUGHPUT_DROP = "throughput_drop"
    BOTTLENECK_DETECTED = "bottleneck_detected"


@dataclass
class Alert:
    """Performance alert information."""
    alert_type: PerformanceAlert
    message: str
    metric_name: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Real-time performance monitoring system for MLX-RL training.
    
    Features:
    - Automatic bottleneck detection
    - Memory usage tracking
    - Performance regression alerts
    - Training efficiency analysis
    - Optimization recommendations
    """
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize performance monitor.
        
        Args:
            alert_thresholds: Custom alert thresholds for different metrics
        """
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.memory_monitor = MemoryMonitor()
        self.alerts: List[Alert] = []
        self.start_time = time.time()
        
        # Default alert thresholds
        self.thresholds = {
            "slow_operation_ms": 100.0,      # Operations slower than 100ms
            "memory_usage_mb": 8000.0,       # Memory usage above 8GB
            "throughput_drop_ratio": 0.7,    # Throughput drops below 70% of recent average
            "memory_growth_rate": 100.0      # Memory growing faster than 100MB/min
        }
        
        if alert_thresholds:
            self.thresholds.update(alert_thresholds)
    
    @contextmanager
    def measure(self, name: str, category: PerformanceCategory = PerformanceCategory.OVERALL,
               track_memory: bool = False):
        """
        Context manager for measuring operation performance.
        
        Args:
            name: Operation name
            category: Performance category
            track_memory: Whether to track memory usage
            
        Example:
            monitor = PerformanceMonitor()
            with monitor.measure("policy_forward", PerformanceCategory.POLICY):
                action = policy(observation)
        """
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetrics(name, category)
        
        memory_before = get_memory_stats() if track_memory else None
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            memory_after = get_memory_stats() if track_memory else None
            
            # Calculate memory delta if tracking
            memory_delta = None
            if memory_before and memory_after:
                memory_delta = {
                    key: memory_after.get(key, 0) - memory_before.get(key, 0)
                    for key in memory_after.keys()
                }
            
            self.metrics[name].update(duration, memory_delta)
            self._check_alerts(name, duration, memory_after)
    
    def record_metric(self, name: str, value: float, category: PerformanceCategory = PerformanceCategory.OVERALL):
        """
        Record a custom metric value.
        
        Args:
            name: Metric name
            value: Metric value
            category: Performance category
        """
        if name not in self.metrics:
            self.metrics[name] = PerformanceMetrics(name, category)
        
        self.metrics[name].update(value)
    
    def _check_alerts(self, metric_name: str, duration: float, memory_stats: Optional[Dict[str, float]]):
        """Check for performance alerts based on current metrics."""
        metric = self.metrics[metric_name]
        
        # Check for slow operations
        if duration * 1000 > self.thresholds["slow_operation_ms"]:
            alert = Alert(
                alert_type=PerformanceAlert.SLOW_OPERATION,
                message=f"Slow operation detected: {metric_name} took {duration*1000:.1f}ms",
                metric_name=metric_name,
                value=duration * 1000,
                threshold=self.thresholds["slow_operation_ms"]
            )
            self.alerts.append(alert)
            debug_print(alert.message, "performance")
        
        # Check for throughput drops
        if len(metric.recent_times) >= 10:
            recent_avg = metric.recent_avg_time
            overall_avg = metric.avg_time
            if recent_avg > overall_avg * (1 / self.thresholds["throughput_drop_ratio"]):
                alert = Alert(
                    alert_type=PerformanceAlert.THROUGHPUT_DROP,
                    message=f"Throughput drop detected in {metric_name}: {recent_avg*1000:.1f}ms vs {overall_avg*1000:.1f}ms avg",
                    metric_name=metric_name,
                    value=recent_avg * 1000,
                    threshold=overall_avg * 1000
                )
                self.alerts.append(alert)
                debug_print(alert.message, "performance")
        
        # Check memory usage
        if memory_stats:
            total_memory = memory_stats.get("mlx_memory_mb", 0) + memory_stats.get("python_memory_mb", 0)
            if total_memory > self.thresholds["memory_usage_mb"]:
                alert = Alert(
                    alert_type=PerformanceAlert.MEMORY_HIGH,
                    message=f"High memory usage: {total_memory:.1f}MB",
                    metric_name="memory_usage",
                    value=total_memory,
                    threshold=self.thresholds["memory_usage_mb"]
                )
                self.alerts.append(alert)
    
    def get_bottlenecks(self, min_time_ms: float = 1.0) -> List[Tuple[str, PerformanceMetrics]]:
        """
        Identify performance bottlenecks.
        
        Args:
            min_time_ms: Minimum average time in ms to consider as bottleneck
            
        Returns:
            List of (metric_name, metrics) tuples sorted by total time
        """
        bottlenecks = []
        for name, metric in self.metrics.items():
            if metric.avg_time * 1000 >= min_time_ms:
                bottlenecks.append((name, metric))
        
        # Sort by total time spent (highest first)
        bottlenecks.sort(key=lambda x: x[1].total_time, reverse=True)
        return bottlenecks
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get performance monitoring summary.
        
        Returns:
            Dictionary with performance summary data
        """
        total_runtime = time.time() - self.start_time
        bottlenecks = self.get_bottlenecks()
        
        summary = {
            "total_runtime_seconds": total_runtime,
            "total_operations": sum(m.call_count for m in self.metrics.values()),
            "total_measured_time": sum(m.total_time for m in self.metrics.values()),
            "overhead_ratio": 1.0 - (sum(m.total_time for m in self.metrics.values()) / total_runtime) if total_runtime > 0 else 0.0,
            "num_alerts": len(self.alerts),
            "num_bottlenecks": len(bottlenecks),
            "top_bottleneck": bottlenecks[0][0] if bottlenecks else None,
            "memory_stats": get_memory_stats(),
            "categories": {}
        }
        
        # Group metrics by category
        for metric in self.metrics.values():
            category = metric.category.value
            if category not in summary["categories"]:
                summary["categories"][category] = {
                    "total_time": 0.0,
                    "call_count": 0,
                    "operations": []
                }
            
            summary["categories"][category]["total_time"] += metric.total_time
            summary["categories"][category]["call_count"] += metric.call_count
            summary["categories"][category]["operations"].append(metric.name)
        
        return summary
    
    def print_summary(self, detailed: bool = True):
        """
        Print performance monitoring summary.
        
        Args:
            detailed: Whether to show detailed metrics
        """
        summary = self.get_summary()
        
        print("\nPerformance Monitoring Summary")
        print("=" * 50)
        print(f"Total runtime: {summary['total_runtime_seconds']:.1f}s")
        print(f"Total operations: {summary['total_operations']}")
        print(f"Monitoring overhead: {summary['overhead_ratio']*100:.1f}%")
        print(f"Alerts: {summary['num_alerts']}")
        print(f"Bottlenecks: {summary['num_bottlenecks']}")
        
        if summary['top_bottleneck']:
            print(f"Top bottleneck: {summary['top_bottleneck']}")
        
        # Memory stats
        memory = summary['memory_stats']
        if memory:
            print("\nMemory Usage:")
            for key, value in memory.items():
                if key.endswith('_mb'):
                    print(f"  {key}: {value:.1f} MB")
        
        if detailed and self.metrics:
            print("\nDetailed Metrics:")
            bottlenecks = self.get_bottlenecks(0.1)  # Show operations > 0.1ms
            
            print(f"{'Operation':<25} {'Calls':<8} {'Total (ms)':<12} {'Avg (ms)':<10} {'Recent (ms)':<12}")
            print("-" * 75)
            
            for name, metric in bottlenecks[:10]:  # Top 10 bottlenecks
                print(f"{name:<25} {metric.call_count:<8} {metric.total_time*1000:<12.1f} "
                      f"{metric.avg_time*1000:<10.2f} {metric.recent_avg_time*1000:<12.2f}")
        
        # Recent alerts
        if self.alerts:
            recent_alerts = [a for a in self.alerts if time.time() - a.timestamp < 300]  # Last 5 minutes
            if recent_alerts:
                print(f"\nRecent Alerts ({len(recent_alerts)}):")
                for alert in recent_alerts[-5:]:  # Last 5 alerts
                    print(f"  {alert.alert_type.value}: {alert.message}")
    
    def print_bottlenecks(self, top_n: int = 10):
        """
        Print identified performance bottlenecks.
        
        Args:
            top_n: Number of top bottlenecks to show
        """
        bottlenecks = self.get_bottlenecks()
        
        if not bottlenecks:
            print("No performance bottlenecks detected.")
            return
        
        print(f"\nTop {min(top_n, len(bottlenecks))} Performance Bottlenecks")
        print("=" * 60)
        
        for i, (name, metric) in enumerate(bottlenecks[:top_n], 1):
            percentage = (metric.total_time / sum(m.total_time for m in self.metrics.values())) * 100
            print(f"{i}. {name}")
            print(f"   Total time: {metric.total_time*1000:.1f}ms ({percentage:.1f}% of total)")
            print(f"   Calls: {metric.call_count} (avg: {metric.avg_time*1000:.2f}ms)")
            print(f"   Range: {metric.min_time*1000:.2f}ms - {metric.max_time*1000:.2f}ms")
            
            # Recommendations
            if metric.avg_time > 0.1:  # > 100ms
                print("    High latency operation - consider optimization")
            elif metric.call_count > 1000 and metric.total_time > 1.0:
                print("    High frequency operation - consider caching/batching")
            
            print()
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on performance data.
        
        Returns:
            List of optimization recommendation strings
        """
        recommendations = []
        bottlenecks = self.get_bottlenecks()
        
        if not bottlenecks:
            recommendations.append("No significant bottlenecks detected")
            return recommendations
        
        # Analyze top bottleneck
        top_name, top_metric = bottlenecks[0]
        
        if "environment" in top_name.lower() or top_metric.category == PerformanceCategory.ENVIRONMENT:
            if top_metric.avg_time > 0.01:  # > 10ms
                recommendations.append(f"Consider vectorizing environment '{top_name}' for better throughput")
            else:
                recommendations.append(f"Environment '{top_name}' is lightweight - vectorization may not help")
        
        if "policy" in top_name.lower() or top_metric.category == PerformanceCategory.POLICY:
            recommendations.append(f"Consider batching policy inference for '{top_name}'")
            recommendations.append("Profile MLX operations in policy forward pass")
        
        # Memory recommendations
        memory_stats = get_memory_stats()
        if memory_stats.get("mlx_memory_mb", 0) > 4000:
            recommendations.append("High MLX memory usage - consider reducing batch size")
        
        # General recommendations
        total_measured = sum(m.total_time for m in self.metrics.values())
        total_runtime = time.time() - self.start_time
        if total_runtime > 0 and (total_measured / total_runtime) < 0.5:
            recommendations.append("Low measurement coverage - add more performance monitoring")
        
        high_variance_ops = [
            name for name, metric in self.metrics.items()
            if len(metric.recent_times) > 10 and 
            (max(metric.recent_times) / min(metric.recent_times)) > 3
        ]
        
        if high_variance_ops:
            recommendations.append(f"High timing variance in: {', '.join(high_variance_ops[:3])}")
        
        return recommendations
    
    def clear_alerts(self, older_than_seconds: Optional[float] = None):
        """
        Clear alerts, optionally only those older than specified time.
        
        Args:
            older_than_seconds: Only clear alerts older than this many seconds
        """
        if older_than_seconds is None:
            self.alerts.clear()
        else:
            cutoff_time = time.time() - older_than_seconds
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
    
    def reset(self):
        """Reset all performance monitoring data."""
        self.metrics.clear()
        self.alerts.clear()
        self.start_time = time.time()


# Global performance monitor instance
global_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(name: str, category: PerformanceCategory = PerformanceCategory.OVERALL,
                       monitor: Optional[PerformanceMonitor] = None, track_memory: bool = False):
    """
    Convenience function for monitoring performance.
    
    Args:
        name: Operation name
        category: Performance category
        monitor: PerformanceMonitor instance (uses global if None)
        track_memory: Whether to track memory usage
        
    Example:
        with monitor_performance("training_step", PerformanceCategory.TRAINING):
            loss = trainer.update(batch)
    """
    m = monitor if monitor is not None else global_monitor
    with m.measure(name, category, track_memory):
        yield


def get_performance_summary() -> Dict[str, Any]:
    """Get performance summary from global monitor."""
    return global_monitor.get_summary()


def print_performance_summary(detailed: bool = True):
    """Print performance summary from global monitor."""
    global_monitor.print_summary(detailed)


def get_optimization_recommendations() -> List[str]:
    """Get optimization recommendations from global monitor."""
    return global_monitor.get_optimization_recommendations()