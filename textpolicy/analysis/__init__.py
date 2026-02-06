# textpolicy/analysis/__init__.py
"""
Post-hoc analysis tooling for TextPolicy training runs.

Main components:
- EmergenceLogger: Captures all generations during GRPO training
- PlanningPatternDetector: Configurable planning-phrase detection
- PlanningPatternConfig: Pattern configuration dataclass
- StreamingJSONLWriter: Append-only JSONL writer
- to_json_safe: MLX/numpy → JSON-native conversion
- Strategic grams: Mining & persistence for HICRA planning tokens
"""

from .emergence_logger import EmergenceLogger
from .planning_patterns import PlanningPatternConfig, PlanningPatternDetector
from .serialization import StreamingJSONLWriter, to_json_safe
from .strategic_grams import (
    DEFAULT_STRATEGIC_GRAMS,
    get_default_strategic_grams,
    extract_ngrams,
    compute_document_frequency,
    load_generations,
    mine_strategic_grams,
    save_strategic_grams,
    load_strategic_grams,
)

__all__ = [
    "EmergenceLogger",
    "PlanningPatternDetector",
    "PlanningPatternConfig",
    "StreamingJSONLWriter",
    "to_json_safe",
    # Strategic grams (Issue #15)
    "DEFAULT_STRATEGIC_GRAMS",
    "get_default_strategic_grams",
    "extract_ngrams",
    "compute_document_frequency",
    "load_generations",
    "mine_strategic_grams",
    "save_strategic_grams",
    "load_strategic_grams",
]
