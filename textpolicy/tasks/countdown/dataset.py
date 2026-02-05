"""
Problem generation and HuggingFace dataset loading for the Countdown task.
"""

import ast
import itertools
import random
from typing import Dict, List, Optional, Tuple


def generate_countdown_problems(
    num_problems: int,
    num_numbers: int = 4,
    number_range: Tuple[int, int] = (1, 25),
    target_range: Tuple[int, int] = (10, 100),
    ensure_solvable: bool = True,
    seed: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> List[Dict]:
    """
    Generate Countdown Numbers Game problems.

    Args:
        num_problems: Number of problems to generate.
        num_numbers: How many numbers per problem (3 or 4 recommended).
        number_range: (min, max) inclusive range for available numbers.
        target_range: (min, max) inclusive range for target.
        ensure_solvable: If True, only return problems with at least one solution.
        seed: Random seed for reproducibility.
        max_attempts: Maximum number of candidate problems to try before stopping.
                      Defaults to num_problems * 100 when ensure_solvable is True.

    Returns:
        List of dicts with keys 'target' and 'numbers'.

    Raises:
        RuntimeError: If max_attempts is exhausted before generating enough problems.
    """
    rng = random.Random(seed)
    problems = []

    if max_attempts is None:
        max_attempts = num_problems * 100 if ensure_solvable else num_problems

    attempts = 0
    while len(problems) < num_problems:
        if attempts >= max_attempts:
            raise RuntimeError(
                f"Could not generate {num_problems} problems within "
                f"{max_attempts} attempts (got {len(problems)}). "
                f"Try wider number_range/target_range or increase max_attempts."
            )

        numbers = [rng.randint(*number_range) for _ in range(num_numbers)]
        target = rng.randint(*target_range)
        attempts += 1

        if ensure_solvable and not _is_solvable(numbers, target):
            continue

        problems.append({"target": target, "numbers": numbers})

    return problems


def load_countdown_dataset(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict]:
    """
    Load the Countdown task dataset from HuggingFace.

    Requires the `datasets` library (optional dependency).

    Args:
        split: Dataset split to load ('train', 'test', etc.).
        max_examples: Maximum number of examples to return.

    Returns:
        List of dicts with keys 'target' and 'numbers'.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load HuggingFace datasets. "
            "Install it with: pip install datasets"
        )

    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)

    examples = []
    for item in ds:
        target = item.get("target")
        numbers = item.get("nums") or item.get("numbers")
        if target is not None and numbers is not None:
            if isinstance(numbers, str):
                numbers = ast.literal_eval(numbers)
            examples.append({"target": int(target), "numbers": list(numbers)})
            if max_examples is not None and len(examples) >= max_examples:
                break

    return examples


# ---------------------------------------------------------------------------
# Brute-force solvability check
# ---------------------------------------------------------------------------

# Commutative ops: only need (a, b), not (b, a)
_COMMUTATIVE_OPS = [
    lambda a, b: a + b,
    lambda a, b: a * b,
]

# Non-commutative ops: must try both orderings
_NON_COMMUTATIVE_OPS = [
    lambda a, b: a - b,
    lambda a, b: a / b if b != 0 else None,
]


def _is_solvable(numbers: List[int], target: int) -> bool:
    """Check if target is reachable using any subset and arrangement of numbers."""
    return _solve(list(map(float, numbers)), float(target))


def _solve(nums: List[float], target: float) -> bool:
    """Recursively try all pairs of numbers with all operations.

    Allows using a subset of numbers — if any single number in the
    current list already equals the target, that counts as solved.

    Uses combinations (not permutations) for pair selection and only
    tries both orderings for non-commutative operations (-, /).
    """
    # Any number in the current set already equals the target → solvable
    for n in nums:
        if abs(n - target) < 1e-9:
            return True

    if len(nums) < 2:
        return False

    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            a, b = nums[i], nums[j]
            remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]

            # Commutative: a+b == b+a, a*b == b*a — one ordering suffices
            for op in _COMMUTATIVE_OPS:
                result = op(a, b)
                if _solve(remaining + [result], target):
                    return True

            # Non-commutative: try both (a,b) and (b,a)
            for op in _NON_COMMUTATIVE_OPS:
                for x, y in ((a, b), (b, a)):
                    result = op(x, y)
                    if result is not None and _solve(remaining + [result], target):
                        return True

    return False
