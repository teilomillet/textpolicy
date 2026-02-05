# textpolicy/analysis/planning_patterns.py
"""
Planning pattern detection for emergence analysis.

Provides configurable pattern matching to identify reasoning behaviors
(hesitation, verification, backtracking, etc.) in model generations.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlanningPatternConfig:
    """Configuration for planning pattern detection.

    Each category maps to a list of literal phrases. The detector builds
    a single compiled regex from all phrases for efficient matching.
    """

    hesitation: List[str] = field(
        default_factory=lambda: [
            "wait",
            "hmm",
            "actually",
            "let me think",
            "on second thought",
        ]
    )
    verification: List[str] = field(
        default_factory=lambda: [
            "let me check",
            "verify",
            "double check",
            "is this right",
        ]
    )
    backtracking: List[str] = field(
        default_factory=lambda: [
            "try another",
            "different approach",
            "go back",
            "start over",
        ]
    )
    alternatives: List[str] = field(
        default_factory=lambda: [
            "alternatively",
            "or we could",
            "another way",
        ]
    )
    metacognition: List[str] = field(
        default_factory=lambda: [
            "notice that",
            "the key is",
            "importantly",
        ]
    )
    case_sensitive: bool = False

    @property
    def all_patterns(self) -> List[str]:
        """Return a flat list of all patterns across every category."""
        patterns: List[str] = []
        for cat in ("hesitation", "verification", "backtracking",
                     "alternatives", "metacognition"):
            patterns.extend(getattr(self, cat))
        return patterns


class PlanningPatternDetector:
    """Efficient planning-phrase detector using a single compiled regex.

    Args:
        config: Optional pattern configuration. Uses defaults if *None*.
    """

    def __init__(self, config: Optional[PlanningPatternConfig] = None) -> None:
        self.config = config or PlanningPatternConfig()
        flags = 0 if self.config.case_sensitive else re.IGNORECASE
        # Sort longest-first so greedy alternation prefers longer matches
        patterns = sorted(self.config.all_patterns, key=len, reverse=True)
        escaped = [re.escape(p) for p in patterns]
        # Guard against empty pattern list — an empty regex matches every position
        self._regex = re.compile("|".join(escaped), flags) if escaped else None

    def detect(self, text: str) -> List[str]:
        """Return all matched planning phrases found in *text*."""
        if not text or self._regex is None:
            return []
        return [m.group() for m in self._regex.finditer(text)]

    def planning_token_ratio(self, text: str, total_tokens: int) -> float:
        """Ratio of planning-phrase words to *total_tokens*.

        Uses whitespace word count of matched phrases as numerator.
        Returns 0.0 when *total_tokens* is zero.
        """
        if total_tokens == 0:
            return 0.0
        matches = self.detect(text)
        planning_words = sum(len(m.split()) for m in matches)
        return planning_words / total_tokens
