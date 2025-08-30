# mlx_rl/utils/timing.py
"""
Timing and performance measurement utilities.
"""

import time
from typing import Dict, Optional
from contextlib import contextmanager


class Timer:
    """
    High-precision timer for performance measurement.
    
    Features:
    - Context manager support for easy use
    - Multiple named timers with aggregation
    - Statistics tracking (mean, min, max, count)
    - MLX-optimized for Apple Silicon performance profiling
    """
    
    def __init__(self):
        """Initialize timer with empty statistics."""
        self.times: Dict[str, list] = {}
        self.current_start: Optional[float] = None
        
    def start(self, name: str = "default"):
        """
        Start timing a named operation.
        
        Args:
            name: Timer name for tracking multiple operations
        """
        if name not in self.times:
            self.times[name] = []
        self.current_start = time.perf_counter()
        
    def stop(self, name: str = "default") -> float:
        """
        Stop timing and record the duration.
        
        Args:
            name: Timer name (must match start() call)
            
        Returns:
            Duration in seconds
            
        Raises:
            RuntimeError: If timer wasn't started
        """
        if self.current_start is None:
            raise RuntimeError(f"Timer '{name}' was not started")
            
        duration = time.perf_counter() - self.current_start
        self.times[name].append(duration)
        self.current_start = None
        return duration
        
    @contextmanager
    def time(self, name: str = "default"):
        """
        Context manager for timing code blocks.
        
        Args:
            name: Timer name
            
        Example:
            timer = Timer()
            with timer.time("policy_forward"):
                action = policy(obs)
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
            
    def get_stats(self, name: str = "default") -> Dict[str, float]:
        """
        Get timing statistics for a named timer.
        
        Args:
            name: Timer name
            
        Returns:
            Dictionary with mean, min, max, total, count statistics
        """
        if name not in self.times or not self.times[name]:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "total": 0.0, "count": 0}
            
        times = self.times[name]
        return {
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
            "total": sum(times),
            "count": len(times)
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all named timers."""
        return {name: self.get_stats(name) for name in self.times.keys()}
        
    def reset(self, name: Optional[str] = None):
        """
        Reset timer statistics.
        
        Args:
            name: Specific timer to reset (None = reset all)
        """
        if name is None:
            self.times.clear()
        elif name in self.times:
            self.times[name].clear()
            
    def __str__(self) -> str:
        """String representation of all timer statistics."""
        lines = ["Timer Statistics:"]
        for name, stats in self.get_all_stats().items():
            lines.append(f"  {name}: {stats['mean']:.4f}s avg ({stats['count']} calls)")
        return "\n".join(lines)


# Global timer instance for convenience
global_timer = Timer()


@contextmanager
def time_it(name: str = "default", timer: Optional[Timer] = None):
    """
    Convenience function for timing code blocks.
    
    Args:
        name: Timer name
        timer: Timer instance (uses global_timer if None)
        
    Example:
        with time_it("training_step"):
            loss = trainer.update(batch)
    """
    t = timer if timer is not None else global_timer
    with t.time(name):
        yield


def benchmark_function(func, *args, iterations: int = 100, warmup: int = 10, **kwargs) -> Dict[str, float]:
    """
    Benchmark a function with multiple iterations.
    
    Args:
        func: Function to benchmark
        *args: Function positional arguments
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations (excluded from timing)
        **kwargs: Function keyword arguments
        
    Returns:
        Statistics dictionary with timing results
    """
    timer = Timer()
    
    # Warmup iterations
    for _ in range(warmup):
        func(*args, **kwargs)
        
    # Benchmark iterations
    for i in range(iterations):
        with timer.time("benchmark"):
            func(*args, **kwargs)
            
    return timer.get_stats("benchmark") 