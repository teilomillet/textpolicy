"""
Countdown reward function for GRPO training.
"""

import logging
from typing import Any, Dict

from textpolicy.rewards.registry import reward
from .evaluator import ExpressionError, evaluate_expression
from .prompt import extract_expression_from_completion

logger = logging.getLogger(__name__)


@reward(name="countdown")
def countdown_reward(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    **kwargs,
) -> float:
    """
    Reward function for the Countdown Numbers Game.

    Scoring:
        1.0  — expression equals target with valid numbers
        0.0  — evaluates but wrong answer or invalid numbers
       -0.5  — syntax error, empty, or unparseable

    The example dict must contain 'target' (int) and 'numbers' (list of int).
    """
    # Extract task parameters
    target = example.get("target")
    numbers = example.get("numbers")

    if target is None or numbers is None:
        logger.warning("Malformed example: missing 'target' or 'numbers'")
        return 0.0

    # Extract expression from completion
    expression = extract_expression_from_completion(completion)
    if not expression:
        return -0.5

    # Evaluate
    try:
        result = evaluate_expression(expression, available_numbers=numbers)
    except ExpressionError as e:
        logger.debug(f"Expression error: {e}")
        return -0.5

    # Check if result matches target (use tolerance for float comparison)
    if abs(result.value - target) < 1e-9:
        return 1.0

    return 0.0
