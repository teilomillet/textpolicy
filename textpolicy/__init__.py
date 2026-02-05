"""
TextPolicy: RL library for text generation with MLX.

This module exposes the public API entry points for algorithms,
training, generation, environment, and rewards.
"""

# Submodule imports for building the public API
from . import algorithms, generation, training

# Import tasks to trigger auto-registration of task reward functions
from . import tasks  # noqa: F401

# Export RL algorithms as defined in textpolicy.algorithms.__all__
from .algorithms import *  # noqa: F403,F401

# Export text generation utilities (load_model, generate_tokens, etc.)
from .generation import *  # noqa: F403,F401

# Export training components (Trainer, RolloutManager, TrainingMetrics)
from .training import *  # noqa: F403,F401

# Export environment components and factory functions
from .environment import (
    TextGenerationEnvironment,
    TextGenerationEnv,
    create_text_generation_test_env,
    validate_learning_progress,
)

# Export installation validation utilities
from .validate import validate_installation

# Export core reward functions and the reward decorator
from .rewards.basic import length_reward, keyword_reward, perplexity_reward, accuracy_reward
from .rewards.registry import reward, verifier

# Build __all__ combining submodule __all__ lists and additional symbols
__all__ = (
    algorithms.__all__
    + generation.__all__
    + training.__all__
    + [
        "TextGenerationEnvironment",
        "TextGenerationEnv",
        "create_text_generation_test_env",
        "validate_learning_progress",
        "validate_installation",
        "length_reward",
        "keyword_reward",
        "perplexity_reward",
        "accuracy_reward",
        "reward",
        "verifier",
    ]
)
