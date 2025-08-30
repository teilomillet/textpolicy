# textpolicy/utils/logging/tensorboard.py
"""
TensorBoard logging integration.
"""

from typing import Dict
from .base import Logger


class TensorboardLogger(Logger):
    """
    TensorBoard integration for local experiment visualization.
    
    Features:
    - Local scalar metric visualization
    - Histogram and distribution tracking
    - Image and model graph visualization
    - No external service dependency
    
    Requires: pip install tensorboard
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logging.
        
        Args:
            log_dir: Directory to store TensorBoard log files
            
        Raises:
            ImportError: If tensorboard is not installed
        """
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError(
                "tensorboard not installed. Install with: pip install tensorboard"
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics to TensorBoard with 'train/' prefix.
        
        Args:
            metrics: Training metrics dictionary
            step: Training step number
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, step)
    
    def log_evaluation(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics to TensorBoard with 'eval/' prefix.
        
        Args:
            metrics: Evaluation metrics dictionary
            step: Training step when evaluation was performed
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, step)
    
    def finish(self):
        """Close TensorBoard writer and flush remaining data."""
        self.writer.close() 