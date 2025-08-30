# textpolicy/utils/logging/base.py
"""
Base logger interface and protocols for TextPolicy.
"""

from typing import Dict
from abc import ABC, abstractmethod


class Logger(ABC):
    """
    Abstract base class for logging backends.
    
    Defines the standard interface that all logging implementations must follow.
    Supports both training metrics and evaluation metrics with step tracking.
    """
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step number for time-series tracking
        """
        pass
    
    @abstractmethod
    def log_evaluation(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metric names to values  
            step: Training step number when evaluation was performed
        """
        pass
    
    @abstractmethod
    def finish(self):
        """
        Finish logging session and cleanup resources.
        
        Called at end of training to properly close connections,
        save final data, and cleanup any temporary resources.
        """
        pass 