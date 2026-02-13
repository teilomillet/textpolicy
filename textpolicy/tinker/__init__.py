# textpolicy/tinker/__init__.py
"""
Tinker integration: run textpolicy algorithms on GPU via Tinker.

This package provides pure-Python ports of our advantage computation
pipeline (MaxRL + GTPO + HICRA + SEPA) that slot into Tinker's
cookbook training loop. No MLX or torch tensors — just Python lists
that get packed into Tinker's Datum objects.
"""

from .advantages import (
    compute_grpo_advantages,
    compute_maxrl_advantages,
    apply_gtpo_weighting,
    apply_hicra,
    apply_sepa_pooling,
    identify_planning_tokens,
)
from .sepa import SEPAController
