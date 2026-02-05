# textpolicy/analysis/__init__.py
"""
Post-hoc analysis tooling for TextPolicy training runs.

Main components:
- EmergenceLogger: Captures all generations during GRPO training
- PlanningPatternDetector: Configurable planning-phrase detection
- PlanningPatternConfig: Pattern configuration dataclass
- StreamingJSONLWriter: Append-only JSONL writer
- to_json_safe: MLX/numpy → JSON-native conversion
"""

from .emergence_logger import EmergenceLogger
from .planning_patterns import PlanningPatternConfig, PlanningPatternDetector
from .serialization import StreamingJSONLWriter, to_json_safe

__all__ = [
    "EmergenceLogger",
    "PlanningPatternDetector",
    "PlanningPatternConfig",
    "StreamingJSONLWriter",
    "to_json_safe",
]
