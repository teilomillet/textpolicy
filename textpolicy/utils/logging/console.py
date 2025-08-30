# textpolicy/utils/logging/console.py
"""
Simple console logging for debugging and minimal setups.
"""

from typing import Dict
from .base import Logger


class ConsoleLogger(Logger):
    """
    Simple console logging for debugging and development.
    
    Features:
    - No external dependencies
    - Immediate output for debugging
    - Configurable verbosity
    - Minimal overhead
    
    Ideal for:
    - Development and debugging
    - CI/CD pipelines
    - Minimal deployment environments
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console logging.
        
        Args:
            verbose: Whether to print metrics to console
        """
        self.verbose = verbose
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics to console.
        
        Args:
            metrics: Training metrics dictionary
            step: Training step number
        """
        if self.verbose:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step} - Training: {metrics_str}")
    
    def log_evaluation(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics to console.
        
        Args:
            metrics: Evaluation metrics dictionary
            step: Training step when evaluation was performed
        """
        if self.verbose:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"Step {step} - Evaluation: {metrics_str}")
    
    def finish(self):
        """No cleanup needed for console logging."""
        pass 