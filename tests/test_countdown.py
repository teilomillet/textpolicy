"""
Tests for the Countdown Numbers Game task.

Covers: evaluator, reward, dataset, prompt, and registration.
"""

import pytest

from textpolicy.tasks.countdown.evaluator import (
    ExpressionError,
    EvalResult,
    evaluate_expression,
)
from textpolicy.tasks.countdown.prompt import (
    format_countdown_prompt,
    extract_expression_from_completion,
)
from textpolicy.tasks.countdown.reward import countdown_reward
from textpolicy.tasks.countdown.dataset import generate_countdown_problems


# =========================================================================
# Expression Evaluator
# =========================================================================


class TestEvaluatorSimple:
    """Basic arithmetic operations."""

    def test_addition(self):
        r = evaluate_expression("1+2")
        assert r.value == pytest.approx(3.0)

    def test_subtraction(self):
        r = evaluate_expression("10-3")
        assert r.value == pytest.approx(7.0)

    def test_multiplication(self):
        r = evaluate_expression("3*4")
        assert r.value == pytest.approx(12.0)

    def test_division(self):
        r = evaluate_expression("12/4")
        assert r.value == pytest.approx(3.0)

    def test_spaces_ignored(self):
        r = evaluate_expression("  1 + 2  ")
        assert r.value == pytest.approx(3.0)


class TestEvaluatorPrecedence:
    """Operator precedence and parentheses."""

    def test_mul_before_add(self):
        r = evaluate_expression("2+3*4")
        assert r.value == pytest.approx(14.0)

    def test_parentheses_override(self):
        r = evaluate_expression("(2+3)*4")
        assert r.value == pytest.approx(20.0)

    def test_nested_parentheses(self):
        r = evaluate_expression("((1+2)*(3+4))")
        assert r.value == pytest.approx(21.0)

    def test_classic_24_game(self):
        r = evaluate_expression("(1+2+3)*4")
        assert r.value == pytest.approx(24.0)

    def test_complex_expression(self):
        r = evaluate_expression("(10-2)*(3+1)/4")
        assert r.value == pytest.approx(8.0)


class TestEvaluatorErrors:
    """Error conditions."""

    def test_division_by_zero(self):
        with pytest.raises(ExpressionError, match="Division by zero"):
            evaluate_expression("1/0")

    def test_empty_expression(self):
        with pytest.raises(ExpressionError, match="Empty"):
            evaluate_expression("")

    def test_whitespace_only(self):
        with pytest.raises(ExpressionError, match="Empty"):
            evaluate_expression("   ")

    def test_disallowed_chars_power(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("2**3")

    def test_disallowed_chars_import(self):
        with pytest.raises(ExpressionError, match="disallowed"):
            evaluate_expression("import os")

    def test_disallowed_chars_eval(self):
        with pytest.raises(ExpressionError, match="disallowed"):
            evaluate_expression("eval('1+2')")

    def test_unmatched_open_paren(self):
        with pytest.raises(ExpressionError, match="parenthesis"):
            evaluate_expression("(1+2")

    def test_unmatched_close_paren(self):
        with pytest.raises(ExpressionError):
            evaluate_expression("1+2)")

    def test_code_injection_semicolon(self):
        with pytest.raises(ExpressionError, match="disallowed"):
            evaluate_expression("1+2; import os")

    def test_code_injection_underscore(self):
        with pytest.raises(ExpressionError, match="disallowed"):
            evaluate_expression("__import__('os')")


class TestEvaluatorNumberValidation:
    """Number availability and reuse validation."""

    def test_valid_numbers(self):
        r = evaluate_expression("1+2", available_numbers=[1, 2, 3])
        assert r.value == pytest.approx(3.0)
        assert sorted(r.numbers_used) == [1, 2]

    def test_number_not_available(self):
        with pytest.raises(ExpressionError, match="not available"):
            evaluate_expression("5+1", available_numbers=[1, 2, 3])

    def test_number_reuse(self):
        with pytest.raises(ExpressionError, match="not available"):
            evaluate_expression("2+2", available_numbers=[2, 3])

    def test_duplicate_available_numbers(self):
        # Two 2s available, so using two 2s is fine
        r = evaluate_expression("2+2", available_numbers=[2, 2, 3])
        assert r.value == pytest.approx(4.0)

    def test_numbers_used_tracking(self):
        r = evaluate_expression("(3+5)*2")
        assert sorted(r.numbers_used) == [2, 3, 5]


# =========================================================================
# Reward Function
# =========================================================================


class TestCountdownReward:
    """Reward scoring logic."""

    def test_correct_answer(self):
        score = countdown_reward(
            prompt="",
            completion="(1+2+3)*4",
            example={"target": 24, "numbers": [1, 2, 3, 4]},
        )
        assert score == 1.0

    def test_wrong_answer(self):
        score = countdown_reward(
            prompt="",
            completion="1+2+3+4",
            example={"target": 24, "numbers": [1, 2, 3, 4]},
        )
        assert score == 0.0

    def test_syntax_error(self):
        score = countdown_reward(
            prompt="",
            completion="abc def",
            example={"target": 3, "numbers": [1, 2]},
        )
        assert score == -0.5

    def test_division_by_zero(self):
        score = countdown_reward(
            prompt="",
            completion="1/0",
            example={"target": 0, "numbers": [1, 0]},
        )
        assert score == -0.5

    def test_empty_completion(self):
        score = countdown_reward(
            prompt="",
            completion="",
            example={"target": 10, "numbers": [5, 2]},
        )
        assert score == -0.5

    def test_number_reuse(self):
        score = countdown_reward(
            prompt="",
            completion="2+2",
            example={"target": 4, "numbers": [2, 3]},
        )
        assert score == -0.5

    def test_malformed_example(self):
        score = countdown_reward(
            prompt="",
            completion="1+2",
            example={},
        )
        assert score == 0.0

    def test_expression_with_surrounding_text(self):
        completion = "Let me think about this...\nThe answer is: (1+2+3)*4\nSo the result is 24."
        score = countdown_reward(
            prompt="",
            completion=completion,
            example={"target": 24, "numbers": [1, 2, 3, 4]},
        )
        assert score == 1.0

    def test_kwargs_compatibility(self):
        # Should accept arbitrary kwargs without error
        score = countdown_reward(
            prompt="",
            completion="1+2",
            example={"target": 3, "numbers": [1, 2]},
            extra_param="ignored",
        )
        assert score == 1.0


# =========================================================================
# Dataset Generation
# =========================================================================


class TestDatasetGeneration:
    """Problem generation."""

    def test_generate_basic(self):
        problems = generate_countdown_problems(10, seed=42)
        assert len(problems) == 10
        for p in problems:
            assert "target" in p
            assert "numbers" in p
            assert isinstance(p["target"], int)
            assert isinstance(p["numbers"], list)
            assert len(p["numbers"]) == 4

    def test_generate_litmus_1000(self):
        """Litmus test: generate 1000 problems without error."""
        problems = generate_countdown_problems(1000, seed=123)
        assert len(problems) == 1000

    def test_solvable_problems_are_solvable(self):
        problems = generate_countdown_problems(
            5, ensure_solvable=True, seed=42
        )
        from textpolicy.tasks.countdown.dataset import _is_solvable

        for p in problems:
            assert _is_solvable(p["numbers"], p["target"]), (
                f"Problem claimed solvable but isn't: {p}"
            )

    def test_num_numbers_3(self):
        problems = generate_countdown_problems(5, num_numbers=3, seed=42)
        for p in problems:
            assert len(p["numbers"]) == 3

    def test_num_numbers_4(self):
        problems = generate_countdown_problems(5, num_numbers=4, seed=42)
        for p in problems:
            assert len(p["numbers"]) == 4

    def test_reproducible_with_seed(self):
        p1 = generate_countdown_problems(5, seed=99)
        p2 = generate_countdown_problems(5, seed=99)
        assert p1 == p2


# =========================================================================
# Prompt Formatting & Extraction
# =========================================================================


class TestPromptFormat:
    """Prompt formatting."""

    def test_format_includes_target(self):
        prompt = format_countdown_prompt(24, [1, 2, 3, 4])
        assert "24" in prompt

    def test_format_includes_all_numbers(self):
        prompt = format_countdown_prompt(24, [1, 2, 3, 4])
        for n in [1, 2, 3, 4]:
            assert str(n) in prompt


class TestExpressionExtraction:
    """Expression extraction from model completions."""

    def test_extract_bare_expression(self):
        expr = extract_expression_from_completion("(1+2)*3")
        assert expr == "(1+2)*3"

    def test_extract_after_equals(self):
        expr = extract_expression_from_completion("The answer = (1+2)*3")
        assert "(1+2)*3" in expr

    def test_extract_after_colon(self):
        expr = extract_expression_from_completion("Answer: (1+2)*3")
        assert "(1+2)*3" in expr

    def test_extract_from_multiline(self):
        text = "Let me think about this.\nI need to use 1, 2, 3.\n(1+2)*3\nThat gives 9."
        expr = extract_expression_from_completion(text)
        assert expr == "(1+2)*3"

    def test_extract_empty(self):
        assert extract_expression_from_completion("") == ""

    def test_extract_with_answer_keyword(self):
        expr = extract_expression_from_completion("The answer is 2+3*4")
        assert "2+3*4" in expr


# =========================================================================
# Registration (integration)
# =========================================================================


class TestRegistration:
    """Verify the countdown reward is registered."""

    def test_countdown_in_registry(self):
        from textpolicy.rewards.registry import REWARD_REGISTRY

        assert "countdown" in REWARD_REGISTRY

    def test_get_reward_function(self):
        from textpolicy.rewards.registry import get_reward_function

        fn = get_reward_function("countdown")
        assert fn is not None
        assert callable(fn)

    def test_registered_function_works(self):
        from textpolicy.rewards.registry import get_reward_function

        fn = get_reward_function("countdown")
        score = fn(
            prompt="",
            completion="1+2",
            example={"target": 3, "numbers": [1, 2]},
        )
        assert score == 1.0
