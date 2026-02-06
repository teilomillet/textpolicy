# textpolicy/utils/timing.py
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
    - Multiple named timers with aggregation (overlapping timers supported)
    - Statistics tracking (mean, min, max, count)
    - Breakdown formatting relative to a total timer
    - MLX-optimized for Apple Silicon performance profiling
    """

    def __init__(self):
        """Initialize timer with empty statistics."""
        self.times: Dict[str, list] = {}
        self._starts: Dict[str, float] = {}

    def start(self, name: str = "default"):
        """
        Start timing a named operation.

        Multiple named timers can run concurrently (e.g. a "total" timer
        wrapping several phase timers).

        Args:
            name: Timer name for tracking multiple operations
        """
        if name not in self.times:
            self.times[name] = []
        self._starts[name] = time.perf_counter()

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
        if name not in self._starts:
            raise RuntimeError(f"Timer '{name}' was not started")

        duration = time.perf_counter() - self._starts.pop(name)
        self.times[name].append(duration)
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

    def format_breakdown(self, total_name: str = "total") -> Dict[str, Dict[str, float]]:
        """
        Compute a percentage breakdown of phase timers relative to a total.

        Args:
            total_name: Name of the timer that represents the total wall time.

        Returns:
            ``{phase: {"seconds": mean_s, "percent": pct_of_total}}`` for every
            timer *except* the total itself.  Returns an empty dict if the total
            timer has no recordings.
        """
        total_stats = self.get_stats(total_name)
        total_mean = total_stats["mean"]
        if total_mean == 0.0:
            return {}

        breakdown: Dict[str, Dict[str, float]] = {}
        for name in self.times:
            if name == total_name:
                continue
            stats = self.get_stats(name)
            breakdown[name] = {
                "seconds": stats["mean"],
                "percent": (stats["mean"] / total_mean) * 100.0,
            }
        return breakdown

    def reset(self, name: Optional[str] = None):
        """
        Reset timer statistics.
        
        Args:
            name: Specific timer to reset (None = reset all)
        """
        if name is None:
            self.times.clear()
            self._starts.clear()
        else:
            if name in self.times:
                self.times[name].clear()
            # Always clear a pending start for this name, even if it has no
            # recorded history yet — prevents stale timestamps leaking across
            # a reset boundary.
            self._starts.pop(name, None)
            
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
