# textpolicy/utils/debug.py
"""
Debug utilities and configuration for TextPolicy.
"""

import os

class DebugConfig:
    """Global debug configuration for TextPolicy."""
    
    def __init__(self):
        # Read from environment variables with defaults
        self.policy_init = os.getenv('MLX_RL_DEBUG_POLICY_INIT', 'false').lower() == 'true'
        self.value_init = os.getenv('MLX_RL_DEBUG_VALUE_INIT', 'false').lower() == 'true'
        self.training = os.getenv('MLX_RL_DEBUG_TRAINING', 'false').lower() == 'true'
        self.gradients = os.getenv('MLX_RL_DEBUG_GRADIENTS', 'false').lower() == 'true'
        self.baseline_estimation = os.getenv('MLX_RL_DEBUG_BASELINE', 'false').lower() == 'true'
        
        # New performance and vectorization debug categories
        self.vectorization = os.getenv('MLX_RL_DEBUG_VECTORIZATION', 'false').lower() == 'true'
        self.environment = os.getenv('MLX_RL_DEBUG_ENVIRONMENT', 'false').lower() == 'true'
        self.performance = os.getenv('MLX_RL_DEBUG_PERFORMANCE', 'false').lower() == 'true'
        self.benchmarking = os.getenv('MLX_RL_DEBUG_BENCHMARKING', 'false').lower() == 'true'
        self.memory = os.getenv('MLX_RL_DEBUG_MEMORY', 'false').lower() == 'true'
        self.timing = os.getenv('MLX_RL_DEBUG_TIMING', 'false').lower() == 'true'
        
        # Overall debug level
        debug_level = os.getenv('MLX_RL_DEBUG', 'info').lower()
        self.enabled = debug_level in ['debug', 'verbose']
        self.verbose = debug_level == 'verbose'
    
    def should_debug(self, category: str) -> bool:
        """Check if debugging is enabled for a specific category."""
        if not self.enabled:
            return False
        
        category_map = {
            'policy_init': self.policy_init,
            'value_init': self.value_init,
            'training': self.training,
            'gradients': self.gradients,
            'baseline': self.baseline_estimation,
            'vectorization': self.vectorization,
            'environment': self.environment,
            'performance': self.performance,
            'benchmarking': self.benchmarking,
            'memory': self.memory,
            'timing': self.timing
        }
        
        return category_map.get(category, self.verbose)

# Global debug configuration instance
debug_config = DebugConfig()

def debug_print(message: str, category: str = 'general', force: bool = False):
    """Print debug message if debugging is enabled for the category."""
    if force or debug_config.should_debug(category):
        print(f"[DEBUG] {message}")

def error_print(message: str, category: str = 'general'):
    """Print error messages only if any debug mode is enabled."""
    if debug_config.enabled:
        print(f"[ERROR] {message}")

def info_print(message: str, category: str = 'general'):
    """Print info messages only if explicitly enabled for the category."""
    if debug_config.should_debug(category):
        print(f"[INFO] {message}")


def performance_debug(message: str, force: bool = False):
    """Debug print for performance-related messages."""
    debug_print(message, 'performance', force)


def vectorization_debug(message: str, force: bool = False):
    """Debug print for vectorization-related messages."""
    debug_print(message, 'vectorization', force)


def environment_debug(message: str, force: bool = False):
    """Debug print for environment-related messages."""
    debug_print(message, 'environment', force)


def benchmarking_debug(message: str, force: bool = False):
    """Debug print for benchmarking-related messages."""
    debug_print(message, 'benchmarking', force)


def memory_debug(message: str, force: bool = False):
    """Debug print for memory-related messages."""
    debug_print(message, 'memory', force)


def timing_debug(message: str, force: bool = False):
    """Debug print for timing-related messages."""
    debug_print(message, 'timing', force)


def get_debug_categories() -> list:
    """Get list of available debug categories."""
    return [
        'policy_init', 'value_init', 'training', 'gradients', 'baseline',
        'vectorization', 'environment', 'performance', 'benchmarking', 
        'memory', 'timing', 'general'
    ]


def is_debug_enabled(category: str = 'general') -> bool:
    """Check if debug is enabled for a specific category."""
    return debug_config.should_debug(category)


def set_debug_level(level: str):
    """
    Set debug level programmatically.
    
    Args:
        level: Debug level ('info', 'debug', 'verbose')
    """
    os.environ['MLX_RL_DEBUG'] = level.lower()
    # Reinitialize global config
    global debug_config
    debug_config = DebugConfig()


def enable_category_debug(category: str, enabled: bool = True):
    """
    Enable/disable debug for a specific category.
    
    Args:
        category: Debug category name
        enabled: Whether to enable or disable
    """
    env_var = f'MLX_RL_DEBUG_{category.upper()}'
    os.environ[env_var] = 'true' if enabled else 'false'
    
    # Reinitialize global config
    global debug_config
    debug_config = DebugConfig()


def print_debug_status():
    """Print current debug configuration status."""
    print("MLX-RL Debug Configuration:")
    print("=" * 40)
    print(f"Overall enabled: {debug_config.enabled}")
    print(f"Verbose mode: {debug_config.verbose}")
    print()
    print("Category-specific settings:")
    
    categories = {
        'policy_init': debug_config.policy_init,
        'value_init': debug_config.value_init,
        'training': debug_config.training,
        'gradients': debug_config.gradients,
        'baseline': debug_config.baseline_estimation,
        'vectorization': debug_config.vectorization,
        'environment': debug_config.environment,
        'performance': debug_config.performance,
        'benchmarking': debug_config.benchmarking,
        'memory': debug_config.memory,
        'timing': debug_config.timing
    }
    
    for category, enabled in categories.items():
        status = "Enabled" if enabled else "Disabled"
        print(f"  {category:<15} {status}")