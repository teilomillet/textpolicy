# textpolicy/utils/logging/factory.py
"""
Factory functions for creating logger instances.
"""

from typing import Optional, List
from .base import Logger
from .wandb import WandbLogger
from .tensorboard import TensorboardLogger
from .console import ConsoleLogger
from .multi import MultiLogger


def create_logger(
    logger_type: str = "console", 
    **kwargs
) -> Logger:
    """
    Factory function to create logger instances.
    
    Args:
        logger_type: Type of logger ("wandb", "tensorboard", "console")
        **kwargs: Logger-specific parameters
        
    Returns:
        Logger instance
        
    Raises:
        ValueError: If logger_type is unknown
        
    Examples:
        # Console logger
        logger = create_logger("console", verbose=True)
        
        # Wandb logger
        logger = create_logger("wandb", project_name="my-project", run_name="test")
        
        # TensorBoard logger
        logger = create_logger("tensorboard", log_dir="./logs")
    """
    if logger_type == "wandb":
        if "project_name" not in kwargs:
            raise ValueError("project_name is required for wandb logger")
        project_name = kwargs.pop("project_name")
        return WandbLogger(project_name, **kwargs)
    elif logger_type == "tensorboard":
        if "log_dir" not in kwargs:
            raise ValueError("log_dir is required for tensorboard logger")
        log_dir = kwargs.pop("log_dir")
        return TensorboardLogger(log_dir)
    elif logger_type == "console":
        return ConsoleLogger(**kwargs)
    else:
        available_types = ["wandb", "tensorboard", "console"]
        raise ValueError(f"Unknown logger type: {logger_type}. Available: {available_types}")


def create_multi_logger(
    configs: List[dict]
) -> MultiLogger:
    """
    Create a MultiLogger from a list of logger configurations.
    
    Args:
        configs: List of dictionaries with "type" and other parameters
        
    Returns:
        MultiLogger instance
        
    Example:
        logger = create_multi_logger([
            {"type": "console", "verbose": True},
            {"type": "wandb", "project_name": "my-project"}
        ])
    """
    loggers = []
    for config in configs:
        logger_type = config.pop("type")
        logger = create_logger(logger_type, **config)
        loggers.append(logger)
    
    return MultiLogger(loggers)


def create_auto_logger(
    project_name: Optional[str] = None,
    log_dir: Optional[str] = None,
    console: bool = True
) -> Logger:
    """
    Automatically create appropriate logger based on available dependencies.
    
    Priority order:
    1. Wandb (if project_name provided and wandb available)
    2. TensorBoard (if log_dir provided and tensorboard available)  
    3. Console (always available)
    
    Args:
        project_name: Wandb project name (enables wandb if available)
        log_dir: TensorBoard log directory (enables tensorboard if available)
        console: Whether to include console logging
        
    Returns:
        Logger instance (MultiLogger if multiple backends, single Logger otherwise)
    """
    loggers = []
    
    # Try wandb first
    if project_name:
        try:
            loggers.append(WandbLogger(project_name))
        except ImportError:
            print("Warning: wandb not available, skipping")
    
    # Try tensorboard
    if log_dir:
        try:
            loggers.append(TensorboardLogger(log_dir))
        except ImportError:
            print("Warning: tensorboard not available, skipping")
    
    # Always add console if requested
    if console:
        loggers.append(ConsoleLogger(verbose=True))
    
    # Return appropriate logger type
    if len(loggers) == 0:
        # Fallback to console
        return ConsoleLogger(verbose=True)
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return MultiLogger(loggers) 