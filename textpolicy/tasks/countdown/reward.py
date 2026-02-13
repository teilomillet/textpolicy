"""Countdown reward and verifier helpers."""

import logging
from typing import Any, Dict, Tuple

from textpolicy.rewards.registry import reward
from .evaluator import ExpressionError, evaluate_expression
from .prompt import extract_expression_from_completion

logger = logging.getLogger(__name__)


def _countdown_score_and_correct(
    completion: str,
    example: Dict[str, Any],
) -> Tuple[float, bool]:
    """Return (reward_score, is_correct) for a countdown completion."""
    target = example.get("target")
    numbers = example.get("numbers")

    if target is None or numbers is None:
        logger.warning("Malformed example: missing 'target' or 'numbers'")
        return 0.0, False

    expression = extract_expression_from_completion(completion)
    if not expression:
        return -0.5, False

    try:
        result = evaluate_expression(expression, available_numbers=numbers)
    except ExpressionError as e:
        logger.debug(f"Expression error: {e}")
        return -0.5, False

    if abs(result.value - target) < 1e-9:
        return 1.0, True

    return 0.0, False


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
        0.0  — evaluates but wrong answer, or malformed example
       -0.5  — syntax error, empty, unparseable, number reuse, or invalid numbers

    The example dict must contain 'target' (int) and 'numbers' (list of int).
    """
    score, _ = _countdown_score_and_correct(
        completion=completion,
        example=example,
    )
    return score


def countdown_reward_with_info(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    """
    Countdown reward variant that emits explicit verifier correctness metadata.

    Returns:
        {"reward": float, "is_correct": bool}
    """
    score, is_correct = _countdown_score_and_correct(
        completion=completion,
        example=example,
    )
    return {"reward": score, "is_correct": is_correct}
