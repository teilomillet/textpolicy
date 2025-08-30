# textpolicy/rewards/__init__.py
"""
Unified reward and verification system for MLX-optimized text generation.

Key principles:
- Decorator-based registration (@reward, @verifier)
- Pure function composition with zero abstraction cost
- MLX compilation for Apple Silicon optimization
- Modular system allowing custom rewards and verifiers
- Pre-filtering verification approach
- Signature consistency: (prompt, completion, example, **kwargs)

This system provides modular reward computation with MLX optimization.
"""

# Registry system
from .registry import (
    reward, verifier,  # Decorators for registration
    RewardFunction, VerifierFunction, RewardConfig,
    get_reward_function, get_verifier_function,
    apply_verifiers_to_reward,
    create_configured_reward_function,
    list_registered_functions, clear_registries,
    REWARD_REGISTRY, VERIFIER_REGISTRY
)

# Core reward functions (auto-registered via decorators)
from .basic import (
    length_reward,
    keyword_reward, 
    perplexity_reward,
    accuracy_reward
)

# Core verifier functions (auto-registered via decorators)
from .verifiers import (
    length_verifier,
    toxicity_verifier,
    coherence_verifier,
    factual_verifier,
    has_greeting,
    no_empty_response,
    contains_keywords,
    # Legacy compatibility (deprecated)
    create_default_verifier_pipeline,
    create_custom_verifier_pipeline
)

# MLX-optimized batch processing (following DESIGN_GUIDELINES.md)
from .mlx_batch_processor import (
    create_batch_reward_processor,
    create_mlx_optimized_batch_processor,
    create_async_batch_processor,
    create_processor_from_config,
    list_available_processors,
    compute_length_rewards_vectorized,
    compute_keyword_rewards_vectorized
)

# Core adapters and utilities for MLX reward system
from .adapters import (
    # Configuration models
    MLXRewardConfig, MLXRewardSystemConfig,
    # Sample types
    MLXSample, MLXExternalRewardModel,
    # Math utilities
    extract_boxed_answer, normalize_math_answer, compute_f1_score,
    # Bridge functions
    create_mlx_batch_adapter, create_mlx_system_from_config,
    samples_to_mlx_format, mlx_format_to_samples,
    get_available_adapters,
    # Reward functions
    math_accuracy_reward, f1_score_reward
)

# Legacy systems (maintained for backward compatibility)
from .rollout_rewards import (
    RolloutRewardProcessor,
    create_rollout_reward_processor,
    process_episode_batch_rewards,
    compute_reward_vector
)

from .integrated_system import (
    IntegratedRewardConfig,
    IntegratedRolloutRewardSystem,
    create_integrated_reward_system,
    process_episodes_with_quality_control,
    compute_integrated_rewards
)

__all__ = [
    # Registry system (primary interface)
    "reward", "verifier",  # Decorators
    "RewardFunction", "VerifierFunction", "RewardConfig",
    "get_reward_function", "get_verifier_function",
    "apply_verifiers_to_reward", "create_configured_reward_function",
    "list_registered_functions", "clear_registries",
    "REWARD_REGISTRY", "VERIFIER_REGISTRY",
    
    # Core reward functions (auto-registered)
    "length_reward", "keyword_reward", "perplexity_reward", "accuracy_reward",
    
    # Core verifier functions (auto-registered)
    "length_verifier", "toxicity_verifier", "coherence_verifier", "factual_verifier",
    "has_greeting", "no_empty_response", "contains_keywords",
    
    # MLX-optimized batch processing (primary interface for training)
    "create_batch_reward_processor",
    "create_mlx_optimized_batch_processor", 
    "create_async_batch_processor",
    "create_processor_from_config",
    "list_available_processors",
    "compute_length_rewards_vectorized",
    "compute_keyword_rewards_vectorized",
    
    # Core adapters and utilities
    "MLXRewardConfig", "MLXRewardSystemConfig",
    "MLXSample", "MLXExternalRewardModel", 
    "extract_boxed_answer", "normalize_math_answer", "compute_f1_score",
    "create_mlx_batch_adapter", "create_mlx_system_from_config",
    "samples_to_mlx_format", "mlx_format_to_samples", "get_available_adapters",
    "math_accuracy_reward", "f1_score_reward",
    
    # Legacy systems (backward compatibility)
    "RolloutRewardProcessor", "create_rollout_reward_processor",
    "process_episode_batch_rewards", "compute_reward_vector",
    "IntegratedRewardConfig", "IntegratedRolloutRewardSystem",
    "create_integrated_reward_system", "process_episodes_with_quality_control",
    "compute_integrated_rewards",
    "create_default_verifier_pipeline", "create_custom_verifier_pipeline",
]

# Import basic rewards and verifiers to trigger registration
import textpolicy.rewards.basic
import textpolicy.rewards.verifiers
import textpolicy.rewards.adapters  # Register adapted rewards