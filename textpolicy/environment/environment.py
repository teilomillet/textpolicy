# mlx_rl/environment/environment.py
"""
Unified environment module that coordinates all environment components.

Environment system for MLX-RL.

This module provides a unified interface for different environment types
(gymnasium, dm_env, custom environments) with adapter patterns for compatibility.

Design principles:
1. Unified Environment interface for all environment types
2. Adapter pattern for external environment libraries
3. Factory system for easy environment creation and registration
4. Support for multiprocessing through cloning
5. Extensible design for adding new environment types
"""

# Import all components for unified API
from .base import Environment, EnvironmentAdapter
from .gym import GymAdapter
from .factory import (
    create_environment,
    register_environment,
    list_registered_environments,
    is_gymnasium_available,
    ENVIRONMENT_REGISTRY
)

# Re-export everything for compatibility
__all__ = [
    # Base classes
    "Environment",
    "EnvironmentAdapter",
    
    # Adapters
    "GymAdapter",
    
    # Factory functions
    "create_environment",
    "register_environment", 
    "list_registered_environments",
    "is_gymnasium_available",
    
    # Registry
    "ENVIRONMENT_REGISTRY"
] 
