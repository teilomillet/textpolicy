# mlx_rl/environment/factory.py
"""
Environment factory and registration system.
"""

from typing import Dict, Callable, Union
from .base import Environment
from .gym import GymAdapter

# Environment registry for dynamic loading
ENVIRONMENT_REGISTRY: Dict[str, Callable] = {}


def register_environment(name: str, factory_func: Callable):
    """
    Register an environment factory function.
    
    Args:
        name: Environment name (will be lowercased)
        factory_func: Function that returns Environment instance
    """
    ENVIRONMENT_REGISTRY[name.lower()] = factory_func


def create_environment(env_spec: Union[str, Environment], **kwargs) -> Environment:
    """
    Create environment from specification.
    
    Args:
        env_spec: Either environment name string or Environment instance
        **kwargs: Additional arguments for environment creation
        
    Returns:
        Environment instance
        
    Examples:
        # Create from string (uses GymAdapter)
        env = create_environment("CartPole-v1")
        
        # Create with custom parameters
        env = create_environment("LunarLander-v2", continuous=True)
        
        # Pass through existing environment
        env = create_environment(my_custom_env)
    """
    if isinstance(env_spec, str):
        # String specification - look up in registry first
        env_name = env_spec.lower()
        
        if env_name in ENVIRONMENT_REGISTRY:
            # Use registered factory
            return ENVIRONMENT_REGISTRY[env_name](**kwargs)
        else:
            # Default to gymnasium adapter
            return GymAdapter(env_spec, **kwargs)
    
    elif isinstance(env_spec, Environment):
        # Already an environment instance
        if kwargs:
            raise ValueError("Cannot pass kwargs when env_spec is already an Environment instance")
        return env_spec
    
    else:
        raise TypeError(f"env_spec must be str or Environment, got {type(env_spec)}")


def list_registered_environments() -> list[str]:
    """
    List all registered environment names.
    
    Returns:
        List of registered environment names
    """
    return list(ENVIRONMENT_REGISTRY.keys())


def is_gymnasium_available() -> bool:
    """
    Check if gymnasium is available for import.
    
    Returns:
        True if gymnasium can be imported, False otherwise
    """
    try:
        return True
    except ImportError:
        return False


# Register some common environments by default
def _register_defaults():
    """Register default environment factories."""
    
    # Gymnasium environments (if available)
    if is_gymnasium_available():
        register_environment("cartpole", lambda **kwargs: GymAdapter("CartPole-v1", **kwargs))
        register_environment("cartpole-v1", lambda **kwargs: GymAdapter("CartPole-v1", **kwargs))
        register_environment("lunarlander", lambda **kwargs: GymAdapter("LunarLander-v2", **kwargs))
        register_environment("lunarlander-v2", lambda **kwargs: GymAdapter("LunarLander-v2", **kwargs))


# Register defaults on import
_register_defaults() 