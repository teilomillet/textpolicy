"""
Task suite registry for TextGenerationEnvironment.

This module provides a minimal registry that maps suite names to loader
functions returning lists of TextGenerationTask instances. It enables
customizable evaluation suites without hardcoding them in the environment.

Notes:
- Keep this registry lightweight and dependency-free.
- Default suites are registered from text_generation.py to avoid import cycles.
- A file-backed loader (JSON/YAML) can be added later if needed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Any, Optional


# Registry type: name -> callable that returns a List[TextGenerationTask]
_TASK_SUITE_REGISTRY: Dict[str, Callable[[], List[Any]]] = {}


def register_task_suite(name: str, loader: Callable[[], List[Any]]) -> None:
    """
    Register a task suite by name.

    Args:
        name: Suite identifier (e.g., "basic", "challenging").
        loader: Callable that returns a list of TextGenerationTask instances.
    """
    if not callable(loader):
        raise TypeError("loader must be callable and return a list of tasks")
    _TASK_SUITE_REGISTRY[name] = loader


def get_task_suite(name: str) -> Optional[List[Any]]:
    """
    Load a registered task suite by name.

    Returns None if the suite is not registered.
    """
    loader = _TASK_SUITE_REGISTRY.get(name)
    if loader is None:
        return None
    return loader()


def list_task_suites() -> List[str]:
    """List available task suite names."""
    return sorted(_TASK_SUITE_REGISTRY.keys())

