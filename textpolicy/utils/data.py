# textpolicy/utils/data.py
"""
Data processing and conversion utilities.
"""

from typing import Any, Dict
import mlx.core as mx # type: ignore
import numpy as np


def to_mlx(data: Any) -> mx.array:
    """
    Convert various data types to MLX arrays.
    
    Args:
        data: Input data (numpy array, list, scalar, etc.)
        
    Returns:
        MLX array
    """
    if isinstance(data, mx.array):
        return data
    elif isinstance(data, np.ndarray):
        # Ensure contiguous array for MLX compatibility (fixes "Invalid type ndarray" error)
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)
        return mx.array(data)
    elif isinstance(data, (list, tuple)):
        return mx.array(data)
    elif isinstance(data, (int, float)):
        return mx.array(data)
    else:
        # Try direct conversion
        return mx.array(data)


def to_numpy(data: mx.array) -> np.ndarray:
    """
    Convert MLX array to numpy array.
    
    Args:
        data: MLX array
        
    Returns:
        Numpy array
    """
    return np.array(data)


def batch_to_mlx(batch: Dict[str, Any]) -> Dict[str, mx.array]:
    """
    Convert batch dictionary to MLX arrays.
    
    Args:
        batch: Dictionary with various data types
        
    Returns:
        Dictionary with MLX arrays
    """
    return {key: to_mlx(value) for key, value in batch.items()}
