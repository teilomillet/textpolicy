# textpolicy/utils/memory.py
"""
Memory monitoring utilities for TextPolicy.
"""

import gc
from typing import Dict, Optional
try:
    import mlx.core as mx # type: ignore
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics in MB
    """
    stats = {}
    
    # MLX memory usage (Apple Silicon GPU/ANE)
    if MLX_AVAILABLE:
        try:
            # MLX memory information
            stats["mlx_memory_mb"] = mx.metal.get_active_memory() / 1024 / 1024
            stats["mlx_peak_mb"] = mx.metal.get_peak_memory() / 1024 / 1024
        except Exception as e:
            print(f"Error getting MLX memory stats: {e}")
            stats["mlx_memory_mb"] = 0.0
            stats["mlx_peak_mb"] = 0.0
    
    # Python memory usage
    try:
        import psutil # type: ignore
        process = psutil.Process()
        stats["python_memory_mb"] = process.memory_info().rss / 1024 / 1024
        stats["python_virtual_mb"] = process.memory_info().vms / 1024 / 1024
    except ImportError:
        stats["python_memory_mb"] = 0.0
        stats["python_virtual_mb"] = 0.0
        
    return stats


def clear_memory():
    """
    Clear memory caches and run garbage collection.
    
    Useful for freeing memory between training runs or evaluations.
    """
    # Python garbage collection
    gc.collect()
    
    # MLX memory cleanup
    if MLX_AVAILABLE:
        try:
            mx.metal.clear_cache()
        except Exception as e:
            print(f"Error clearing MLX memory: {e}")
            pass


class MemoryMonitor:
    """
    Monitor memory usage during training.
    
    Features:
    - Track peak memory usage
    - Automatic memory alerts
    - Integration with logging systems
    """
    
    def __init__(self, alert_threshold_mb: float = 8000):
        """
        Initialize memory monitor.
        
        Args:
            alert_threshold_mb: Memory usage threshold for alerts (default 8GB)
        """
        self.alert_threshold = alert_threshold_mb
        self.peak_stats = {}
        
    def check_memory(self, step: Optional[int] = None) -> Dict[str, float]:
        """
        Check current memory usage and update peaks.
        
        Args:
            step: Optional training step for logging
            
        Returns:
            Current memory statistics
        """
        current = get_memory_stats()
        
        # Update peaks
        for key, value in current.items():
            if key not in self.peak_stats or value > self.peak_stats[key]:
                self.peak_stats[key] = value
                
        # Check for alerts
        total_memory = current.get("mlx_memory_mb", 0) + current.get("python_memory_mb", 0)
        if total_memory > self.alert_threshold:
            print(f"Memory alert: {total_memory:.1f}MB (threshold: {self.alert_threshold:.1f}MB)")
            if step is not None:
                print(f"   At training step: {step}")
                
        return current
        
    def get_peak_stats(self) -> Dict[str, float]:
        """Get peak memory usage statistics."""
        return self.peak_stats.copy()
        
    def reset_peaks(self):
        """Reset peak memory tracking."""
        self.peak_stats.clear() 