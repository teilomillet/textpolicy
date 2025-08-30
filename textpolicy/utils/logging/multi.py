# textpolicy/utils/logging/multi.py
"""
Multi-logger for combining multiple logging backends.
"""

from typing import Dict, List
from .base import Logger


class MultiLogger(Logger):
    """
    Combine multiple logging backends into a single interface.
    
    Features:
    - Log to multiple backends simultaneously
    - Graceful error handling (one logger failure doesn't stop others)
    - Unified interface for complex logging setups
    
    Example:
        # Log to both wandb and console
        logger = MultiLogger([
            WandbLogger("my-project"),
            ConsoleLogger(verbose=True)
        ])
    """
    
    def __init__(self, loggers: List[Logger]):
        """
        Initialize multi-logger with list of backends.
        
        Args:
            loggers: List of Logger instances to combine
            
        Raises:
            ValueError: If no loggers provided
        """
        if not loggers:
            raise ValueError("At least one logger must be provided")
        self.loggers = loggers
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics to all backends.
        
        Continues logging to other backends even if one fails.
        
        Args:
            metrics: Training metrics dictionary
            step: Training step number
        """
        for logger in self.loggers:
            try:
                logger.log_metrics(metrics, step)
            except Exception as e:
                print(f"Warning: Logger {type(logger).__name__} failed: {e}")
    
    def log_evaluation(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics to all backends.
        
        Continues logging to other backends even if one fails.
        
        Args:
            metrics: Evaluation metrics dictionary
            step: Training step when evaluation was performed
        """
        for logger in self.loggers:
            try:
                logger.log_evaluation(metrics, step)
            except Exception as e:
                print(f"Warning: Logger {type(logger).__name__} failed: {e}")
    
    def finish(self):
        """
        Finish all loggers.
        
        Attempts to finish all loggers even if some fail.
        """
        for logger in self.loggers:
            try:
                logger.finish()
            except Exception as e:
                print(f"Warning: Logger {type(logger).__name__} finish failed: {e}") 