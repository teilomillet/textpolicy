# textpolicy/tinker/__init__.py
"""
Tinker integration: run textpolicy algorithms on GPU via Tinker.

This package provides pure-Python ports of our advantage computation
pipeline (MaxRL + GTPO + HICRA + SEPA) that slot into Tinker's
cookbook training loop. No MLX or torch tensors — just Python lists
that get packed into Tinker's Datum objects.
"""

from .advantages import (
    apply_gtpo_weighting as apply_gtpo_weighting,
    apply_hicra as apply_hicra,
    apply_sepa_pooling as apply_sepa_pooling,
    compute_grpo_advantages as compute_grpo_advantages,
    compute_maxrl_advantages as compute_maxrl_advantages,
    identify_planning_tokens as identify_planning_tokens,
)
from .sepa import SEPAController as SEPAController
