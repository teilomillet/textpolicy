# textpolicy/rollout/__init__.py
"""
Modular rollout system for TextPolicy.

Main components:
- RolloutCoordinator: Interface for rollout collection
- RolloutStrategy: Algorithm-specific rollout behavior
- RolloutWorker: Multi-process worker management
- BufferAggregator: Multi-worker data coordination
"""

from .rollout import RolloutCoordinator, create_rollout_coordinator
from .base import RolloutStrategy
from .worker import RolloutWorker
from .runner import RolloutRunner
from .aggregator import BufferAggregator
from .strategy import PPOStrategy, GRPOStrategy, create_strategy

# Backwards compatibility exports
from .runner import RolloutRunner as RolloutRunner_Legacy
from .aggregator import BufferAggregator as BufferAggregator_Legacy
from .worker import RolloutWorker as RolloutWorker_Legacy

__all__ = [
    # Main public interface
    'RolloutCoordinator',
    'create_rollout_coordinator',
    
    # Core components
    'RolloutStrategy',
    'RolloutWorker', 
    'RolloutRunner',
    'BufferAggregator',
    
    # Strategies
    'PPOStrategy',
    'GRPOStrategy', 
    'create_strategy',
    
    # Legacy compatibility
    'RolloutRunner_Legacy',
    'BufferAggregator_Legacy',
    'RolloutWorker_Legacy',
] 
