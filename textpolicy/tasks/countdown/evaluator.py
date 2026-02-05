"""
Safe arithmetic expression evaluator using recursive descent parsing.

No eval(), no ast module. Handles +, -, *, /, parentheses, integers only.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


class ExpressionError(Exception):
    """Raised when an expression is invalid or cannot be evaluated."""
    pass


@dataclass
class EvalResult:
    """Result of evaluating an arithmetic expression."""
    value: float
    numbers_used: List[int] = field(default_factory=list)


# Allowed characters in expressions
_ALLOWED_CHARS = re.compile(r'^[0-9\s+\-*/()]+$')


def evaluate_expression(
    expression: str, available_numbers: Optional[List[int]] = None
) -> EvalResult:
    """
    Safely evaluate an arithmetic expression using recursive descent parsing.

    Args:
        expression: Arithmetic expression string (e.g. "(2+3)*4")
        available_numbers: If provided, validates that only these numbers are used
                          (each at most once).

    Returns:
        EvalResult with the computed value and list of numbers used.

    Raises:
        ExpressionError: On syntax errors, division by zero, disallowed chars,
                        or number reuse/unavailability.
    """
    if not expression or not expression.strip():
        raise ExpressionError("Empty expression")

    expr = expression.strip()

    if not _ALLOWED_CHARS.match(expr):
        raise ExpressionError(
            f"Expression contains disallowed characters: {expr!r}"
        )

    tokens = _tokenize(expr)
    if not tokens:
        raise ExpressionError("Empty expression after tokenization")

    parser = _Parser(tokens)
    value = parser.parse_expression()

    if parser.pos < len(parser.tokens):
        raise ExpressionError(
            f"Unexpected token after end of expression: "
            f"{parser.tokens[parser.pos]!r}"
        )

    numbers_used = parser.numbers_used

    if available_numbers is not None:
        _validate_numbers(numbers_used, available_numbers)

    return EvalResult(value=value, numbers_used=numbers_used)


def _tokenize(expr: str) -> List[str]:
    """Tokenize an expression into numbers, operators, and parentheses."""
    tokens = []
    i = 0
    while i < len(expr):
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < len(expr) and expr[j].isdigit():
                j += 1
            tokens.append(expr[i:j])
            i = j
        elif ch in '+-*/()':
            tokens.append(ch)
            i += 1
        else:
            raise ExpressionError(f"Unexpected character: {ch!r}")
    return tokens


class _Parser:
    """Recursive descent parser for arithmetic expressions.

    Grammar:
        expression := term (('+' | '-') term)*
        term       := factor (('*' | '/') factor)*
        factor     := NUMBER | '(' expression ')' | ('+' | '-') factor
    """

    def __init__(self, tokens: List[str]):
        self.tokens = tokens
        self.pos = 0
        self.numbers_used: List[int] = []

    def _peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse_expression(self) -> float:
        """Parse an expression: term (('+' | '-') term)*"""
        result = self._parse_term()
        while self._peek() in ('+', '-'):
            op = self._consume()
            right = self._parse_term()
            if op == '+':
                result = result + right
            else:
                result = result - right
        return result

    def _parse_term(self) -> float:
        """Parse a term: factor (('*' | '/') factor)*"""
        result = self._parse_factor()
        while self._peek() in ('*', '/'):
            op = self._consume()
            right = self._parse_factor()
            if op == '*':
                result = result * right
            else:
                if right == 0:
                    raise ExpressionError("Division by zero")
                result = result / right
        return result

    def _parse_factor(self) -> float:
        """Parse a factor: NUMBER | '(' expression ')' | unary +/-"""
        token = self._peek()
        if token is None:
            raise ExpressionError("Unexpected end of expression")

        # Unary plus/minus
        if token in ('+', '-'):
            op = self._consume()
            value = self._parse_factor()
            return value if op == '+' else -value

        # Parenthesized expression
        if token == '(':
            self._consume()  # eat '('
            value = self.parse_expression()
            if self._peek() != ')':
                raise ExpressionError("Unmatched opening parenthesis")
            self._consume()  # eat ')'
            return value

        if token == ')':
            raise ExpressionError("Unmatched closing parenthesis")

        # Number — _tokenize only produces all-digit tokens, so isdigit()
        # is sufficient. Avoid masking malformed tokens with a looser check.
        if token.isdigit():
            self._consume()
            num = int(token)
            self.numbers_used.append(num)
            return float(num)

        raise ExpressionError(f"Unexpected token: {token!r}")


def _validate_numbers(
    numbers_used: List[int], available_numbers: List[int]
) -> None:
    """Validate that numbers_used is a valid subset of available_numbers."""
    available_copy = list(available_numbers)
    for num in numbers_used:
        if num in available_copy:
            available_copy.remove(num)
        else:
            raise ExpressionError(
                f"Number {num} is not available or has been used too many times. "
                f"Available: {available_numbers}"
            )
