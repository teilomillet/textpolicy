# textpolicy/utils/__init__.py
"""
General utilities for TextPolicy.

Organized by functionality:
- logging: Multiple backend logging support
- timing: Performance measurement and benchmarking
- memory: Memory monitoring and cleanup
- data: Data conversion and preprocessing
- math: Mathematical utilities and statistics
"""

# Import logging utilities
from .logging import (
    Logger, WandbLogger, TensorboardLogger, ConsoleLogger, MultiLogger,
    create_logger, create_multi_logger, create_auto_logger
)

# Import other utilities
from .timing import Timer, global_timer, time_it, benchmark_function
from .memory import get_memory_stats, clear_memory, MemoryMonitor
from .data import to_mlx, to_numpy, batch_to_mlx

# Backwards compatibility for existing imports
from .logging import *

__all__ = [
    # Logging utilities
    'Logger', 'WandbLogger', 'TensorboardLogger', 'ConsoleLogger', 'MultiLogger',
    'create_logger', 'create_multi_logger', 'create_auto_logger',
    
    # Timing utilities
    'Timer', 'global_timer', 'time_it', 'benchmark_function',
    
    # Memory utilities
    'get_memory_stats', 'clear_memory', 'MemoryMonitor',
    
    # Data utilities
    'to_mlx', 'to_numpy', 'batch_to_mlx',
] 