# textpolicy/algorithms/__init__.py
"""
Reinforcement learning algorithms for MLX and Apple Silicon.

GRPO: group-relative advantages with PPO-style clipping.
GSPO: sequence-level importance sampling (sequence, token, and hybrid variants).
HICRA: planning token amplification via strategic gram matching.
"""

from .grpo import (
    # Core GRPO functions
    compute_advantages,
    compute_advantages_dr_grpo,
    policy_loss,
    grpo_loss,
    compute_metrics,
    entropy_bonus,
    select_all_data,
    select_recent_data,
    # GTPO: Entropy-weighted credit assignment
    compute_token_entropy,
    apply_entropy_weighting,
    compute_advantages_gtpo,
    # GTPO: Paper-exact implementation (arXiv 2508.04349)
    compute_gtpo_shaped_rewards,
    normalize_gtpo_advantages,
    gtpo_loss,
    # Compiled versions
    compute_advantages_compiled,
    policy_loss_compiled,
    policy_loss_compiled_constant_norm,
    # Length shaping (DAPO-style soft overlong penalties)
    compute_length_penalty,
    apply_length_shaping,
    compute_length_shaping_stats,
    # Dynamic batch filtering
    filter_informative_prompts,
    compute_prompt_group_stats,
    select_informative_data,
)

from .gspo import (
    create_gspo_policy_loss,
    create_gspo_metrics,
    policy_loss_sequence,
    policy_loss_hybrid,
    create_policy_loss_hybrid,
    policy_loss_token,
    compute_metrics_sequence,
    compute_metrics_hybrid,
    compute_metrics_token,
    select_gspo_data
)

from .hicra import (
    identify_planning_tokens,
    apply_hicra_amplification,
    boost_entropy_with_planning,
    compute_advantages_hicra,
)

__all__ = [
    # GRPO core functions
    "compute_advantages",
    "compute_advantages_dr_grpo",
    "policy_loss",
    "grpo_loss",
    "compute_metrics",
    "entropy_bonus",
    "select_all_data",
    "select_recent_data",
    # GTPO: Entropy-weighted credit assignment
    "compute_token_entropy",
    "apply_entropy_weighting",
    "compute_advantages_gtpo",
    # GTPO: Paper-exact implementation (arXiv 2508.04349)
    "compute_gtpo_shaped_rewards",
    "normalize_gtpo_advantages",
    "gtpo_loss",
    # GRPO compiled versions
    "compute_advantages_compiled",
    "policy_loss_compiled",
    "policy_loss_compiled_constant_norm",
    # GRPO length shaping (DAPO-style)
    "compute_length_penalty",
    "apply_length_shaping",
    "compute_length_shaping_stats",
    # GRPO dynamic batch filtering
    "filter_informative_prompts",
    "compute_prompt_group_stats",
    "select_informative_data",
    # GSPO functions
    "create_gspo_policy_loss",
    "create_gspo_metrics",
    "policy_loss_sequence",
    "policy_loss_hybrid",
    "create_policy_loss_hybrid",
    "policy_loss_token",
    "compute_metrics_sequence",
    "compute_metrics_hybrid",
    "compute_metrics_token",
    "select_gspo_data",
    # HICRA: Planning token amplification (Issue #11)
    "identify_planning_tokens",
    "apply_hicra_amplification",
    "boost_entropy_with_planning",
    "compute_advantages_hicra",
]
