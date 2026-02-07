# textpolicy/training/__init__.py
"""
Unified training infrastructure for all RL algorithms.
"""

from .trainer import Trainer
from .rollout_manager import RolloutManager
from .metrics import TrainingMetrics

__all__ = [
    "Trainer",
    "RolloutManager", 
    "TrainingMetrics",
]

# Optional reasoning-stack helpers (module may be absent in partial installs).
try:
    from .reasoning_stack import (
        build_gtpo_hicra_transform,
        create_tinylora_reasoning_setup,
    )

    __all__.extend(
        [
            "build_gtpo_hicra_transform",
            "create_tinylora_reasoning_setup",
        ]
    )
except Exception:
    pass
