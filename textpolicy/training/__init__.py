# textpolicy/training/__init__.py
"""
Unified training infrastructure for all RL algorithms.
"""

from .trainer import Trainer
from .rollout_manager import RolloutManager
from .metrics import TrainingMetrics
from .reasoning_stack import (
    build_gtpo_hicra_transform,
    create_tinylora_reasoning_setup,
)

__all__ = [
    "Trainer",
    "RolloutManager", 
    "TrainingMetrics",
    "build_gtpo_hicra_transform",
    "create_tinylora_reasoning_setup",
]
