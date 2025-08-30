# mlx_rl/environment/__init__.py
"""
Modular environment implementation for MLX-RL.
"""

from .environment import (
    # Base classes
    Environment,
    EnvironmentAdapter,
    
    # Adapters  
    GymAdapter,
    
    # Factory functions
    create_environment,
    register_environment,
    list_registered_environments,
    is_gymnasium_available,
    
    # Registry
    ENVIRONMENT_REGISTRY
)

from .vectorized import (
    # Vectorized environment classes
    VectorizedEnvironment,
    VectorizedCollector,
    
    # Factory function
    make_vectorized_env,
)

# Task suite registry for text generation environments
from .task_suites import (
    register_task_suite,
    list_task_suites,
    get_task_suite,
)
# Re-export text generation environments and helpers for public API access
from .text_generation import (
    TextGenerationEnvironment,
    TextGenerationEnv,
    create_text_generation_test_env,
    validate_learning_progress,
)

__all__ = [
    # Base classes
    "Environment", 
    "EnvironmentAdapter",
    
    # Adapters
    "GymAdapter",
    
    # Vectorized environment classes
    "VectorizedEnvironment",
    "VectorizedCollector",
    
    # Factory functions
    "create_environment",
    "register_environment",
    "list_registered_environments", 
    "is_gymnasium_available",
    "make_vectorized_env",
    
    # Task suite registry
    "register_task_suite",
    "list_task_suites",
    "get_task_suite",
    
    # Registry
    "ENVIRONMENT_REGISTRY",
    
    # Text generation environments and helpers
    "TextGenerationEnvironment",
    "TextGenerationEnv",
    "create_text_generation_test_env",
    "validate_learning_progress",
] 
