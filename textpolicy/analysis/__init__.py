# textpolicy/analysis/__init__.py
"""
Post-hoc analysis tooling for TextPolicy training runs.

Main components:
- EmergenceLogger: Captures all generations during GRPO training
- PlanningPatternDetector: Configurable planning-phrase detection
- PlanningPatternConfig: Pattern configuration dataclass
- StreamingJSONLWriter: Append-only JSONL writer
- to_json_safe: MLX/numpy → JSON-native conversion
- SEPA litmus: Thresholded SEPA-vs-baseline decision helper
- SEPA significance: Statistical tests + sample-size estimates
- Strategic grams: Mining & persistence for HICRA planning tokens
"""

from .emergence_logger import EmergenceLogger
from .planning_patterns import PlanningPatternConfig, PlanningPatternDetector
from .serialization import StreamingJSONLWriter, to_json_safe
from .sepa_litmus import (
    EmergenceRunStats,
    MetricCheck,
    OFFICIAL_SEPA_LITMUS_PROFILE,
    SEPALitmusEvidence,
    SEPALitmusGroupStats,
    SEPALitmusProfile,
    SEPALitmusResult,
    SEPALitmusThresholds,
    aggregate_group_stats,
    build_litmus_markdown,
    evaluate_sepa_litmus,
    get_sepa_litmus_profile,
    load_emergence_run_stats,
)
from .sepa_significance import (
    MeanDiffTestResult,
    RateDiffTestResult,
    SampleSizeEstimate,
    SEPASignificanceReport,
    build_sepa_significance_markdown,
    evaluate_sepa_significance,
)
from .strategic_grams import (
    COUNTDOWN_STRATEGIC_GRAMS,
    DEFAULT_STRATEGIC_GRAMS,
    get_countdown_strategic_grams,
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
    # SEPA litmus
    "SEPALitmusThresholds",
    "SEPALitmusEvidence",
    "SEPALitmusProfile",
    "OFFICIAL_SEPA_LITMUS_PROFILE",
    "get_sepa_litmus_profile",
    "EmergenceRunStats",
    "SEPALitmusGroupStats",
    "MetricCheck",
    "SEPALitmusResult",
    "load_emergence_run_stats",
    "aggregate_group_stats",
    "evaluate_sepa_litmus",
    "build_litmus_markdown",
    # SEPA significance
    "MeanDiffTestResult",
    "RateDiffTestResult",
    "SampleSizeEstimate",
    "SEPASignificanceReport",
    "evaluate_sepa_significance",
    "build_sepa_significance_markdown",
    # Strategic grams (Issue #15)
    "COUNTDOWN_STRATEGIC_GRAMS",
    "DEFAULT_STRATEGIC_GRAMS",
    "get_countdown_strategic_grams",
    "get_default_strategic_grams",
    "extract_ngrams",
    "compute_document_frequency",
    "load_generations",
    "mine_strategic_grams",
    "save_strategic_grams",
    "load_strategic_grams",
]
