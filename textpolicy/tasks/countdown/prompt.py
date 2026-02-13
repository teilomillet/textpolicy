"""
Prompt formatting and expression extraction for the Countdown task.
"""

import re
from typing import List


def format_countdown_prompt(target: int, numbers: List[int]) -> str:
    """
    Format a Countdown task prompt.

    Args:
        target: The target number to reach.
        numbers: The available numbers to use.

    Returns:
        A formatted prompt string.
    """
    return (
        f"Using the numbers {numbers}, create an arithmetic expression "
        f"that equals {target}. You may use each number at most once. "
        f"Use only +, -, *, / and parentheses. "
        "Use at most two lines. "
        "First line: very brief reasoning (max 18 words, no label). "
        "Final line: arithmetic expression only (no words or labels)."
    )


# Pattern matching pure arithmetic expressions (digits, operators, parens, spaces)
_EXPR_PATTERN = re.compile(r'^[\d\s+\-*/()]+$')

# Pattern for lines with delimiters like "= ...", ": ...", "answer ..."
_DELIMITER_PATTERN = re.compile(
    r'(?:=|:|answer\s*(?:is|:)?)\s*([\d\s+\-*/()]+)',
    re.IGNORECASE,
)

# Find longest arithmetic-like substring
_ARITH_SUBSTRING = re.compile(r'[\d\s+\-*/()]+')

_UNICODE_OPERATOR_MAP = str.maketrans({
    "×": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "﹣": "-",
})


def extract_expression_from_completion(completion: str) -> str:
    """
    Extract an arithmetic expression from model output.

    Uses fallback strategies:
    1. Lines that are pure arithmetic expressions
    2. Text after =, :, or 'answer' delimiters
    3. Longest arithmetic-like substring

    Args:
        completion: The raw model output.

    Returns:
        The extracted expression string (may still be invalid).
    """
    if not completion or not completion.strip():
        return ""

    text = completion.strip().translate(_UNICODE_OPERATOR_MAP)

    # Strategy 1: Find lines that are pure arithmetic expressions
    for line in text.splitlines():
        line = line.strip()
        if line and _EXPR_PATTERN.match(line) and _has_digit_and_operator(line):
            return line

    # Strategy 2: Look for delimiters
    match = _DELIMITER_PATTERN.search(text)
    if match:
        candidate = match.group(1).strip()
        if candidate and _has_digit_and_operator(candidate):
            return candidate

    # Strategy 3: Longest arithmetic-like substring containing at least
    # one digit and one operator
    candidates = _ARITH_SUBSTRING.findall(text)
    valid = [c.strip() for c in candidates if _has_digit_and_operator(c.strip())]
    if valid:
        return max(valid, key=len)

    # Last resort: return the whole text stripped
    return text


def _has_digit_and_operator(s: str) -> bool:
    """Check if string has at least one digit and one operator."""
    has_digit = any(c.isdigit() for c in s)
    has_op = any(c in '+-*/' for c in s)
    return has_digit and has_op
