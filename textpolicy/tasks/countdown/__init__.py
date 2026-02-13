"""
Countdown Numbers Game task for TextPolicy.

Importing this module registers the 'countdown' reward function.
"""

from .evaluator import ExpressionError, EvalResult, evaluate_expression
from .prompt import format_countdown_prompt, extract_expression_from_completion
from .reward import countdown_reward, countdown_reward_with_info
from .dataset import generate_countdown_problems, load_countdown_dataset

__all__ = [
    "ExpressionError",
    "EvalResult",
    "evaluate_expression",
    "format_countdown_prompt",
    "extract_expression_from_completion",
    "countdown_reward",
    "countdown_reward_with_info",
    "generate_countdown_problems",
    "load_countdown_dataset",
]
