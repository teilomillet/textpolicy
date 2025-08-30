# textpolicy/utils/logging/wandb.py
"""
Weights & Biases (wandb) logging integration.
"""

from typing import Dict, Optional
from .base import Logger


class WandbLogger(Logger):
    """
    Weights & Biases integration for experiment tracking.
    
    Features:
    - Automatic experiment organization with projects
    - Real-time metric visualization
    - Hyperparameter tracking
    - Model artifact management
    
    Requires: pip install wandb
    """
    
    def __init__(self, project_name: str, run_name: Optional[str] = None, **kwargs):
        """
        Initialize wandb logging.
        
        Args:
            project_name: Wandb project name for organization
            run_name: Optional run name (auto-generated if None)
            **kwargs: Additional wandb.init() parameters (tags, config, etc.)
            
        Raises:
            ImportError: If wandb is not installed
        """
        try:
            import wandb # type: ignore
            self.wandb = wandb
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with: pip install wandb"
            )
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log training metrics to wandb with 'train/' prefix.
        
        Args:
            metrics: Training metrics dictionary
            step: Training step number
        """
        prefixed_metrics = {"train/" + k: v for k, v in metrics.items()}
        self.wandb.log(prefixed_metrics, step=step)
    
    def log_evaluation(self, metrics: Dict[str, float], step: int):
        """
        Log evaluation metrics to wandb with 'eval/' prefix.
        
        Args:
            metrics: Evaluation metrics dictionary
            step: Training step when evaluation was performed
        """
        prefixed_metrics = {"eval/" + k: v for k, v in metrics.items()}
        self.wandb.log(prefixed_metrics, step=step)
    
    def finish(self):
        """Finish wandb run and upload final data."""
        self.wandb.finish() 