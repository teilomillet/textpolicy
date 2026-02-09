# textpolicy/training/__init__.py
"""
Unified training infrastructure for all RL algorithms.
"""

from .trainer import Trainer
from .rollout_manager import RolloutManager
from .metrics import (
    TrainingMetrics,
    RolloutMetrics,
    log_metrics,
    compute_explained_variance,
    compute_policy_metrics,
)

__all__ = [
    "Trainer",
    "RolloutManager",
    "TrainingMetrics",
    "RolloutMetrics",
    "log_metrics",
    "compute_explained_variance",
    "compute_policy_metrics",
]

# Gradient checkpointing utilities.
from .gradient_checkpointing import (
    apply_gradient_checkpointing,
    is_gradient_checkpointing_active,
    remove_gradient_checkpointing,
)
from .sepa import SEPAController, normalize_sepa_schedule
from .semantic_entropy import (
    SemanticEntropyTracker,
    build_prompt_group_keys,
    pool_planning_hidden_states,
)

__all__.extend(
    [
        "apply_gradient_checkpointing",
        "remove_gradient_checkpointing",
        "is_gradient_checkpointing_active",
        "SEPAController",
        "normalize_sepa_schedule",
        "SemanticEntropyTracker",
        "build_prompt_group_keys",
        "pool_planning_hidden_states",
    ]
)

# Optional reasoning-stack helpers (module may be absent in partial installs).
try:
    from .reasoning_stack import (
        build_gtpo_hicra_transform,
        build_gtpo_transform,
        create_tinylora_reasoning_setup,
    )

    __all__.extend(
        [
            "build_gtpo_hicra_transform",
            "build_gtpo_transform",
            "create_tinylora_reasoning_setup",
        ]
    )
except Exception:
    pass
