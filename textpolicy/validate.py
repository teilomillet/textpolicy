"""
Lightweight programmatic installation validation for TextPolicy.

This module provides a fast health check without network or model downloads.
It verifies critical imports, environment step contracts, and basic rollout
plumbing using a minimal text generation environment.

Usage:
    from textpolicy.validate import validate_installation
    report = validate_installation()
    assert report["status"] == "ok"
"""

from typing import Any, Dict, List


def validate_installation(verbose: bool = True) -> Dict[str, Any]:
    """
    Run a series of quick validation checks.

    - Import checks: mlx, gymnasium, mlx_lm (optional)
    - Environment contract: TextGenerationEnv reset/step returns
    - Rollout shape sanity: basic reward extraction path

    Returns:
        A dictionary with keys:
            status: "ok" or "fail"
            checks: mapping of check name to details
            errors: list of error messages
    """
    checks: Dict[str, Any] = {}
    errors: List[str] = []

    # 1) Import checks: MLX (required for policies), gymnasium (adapters), mlx_lm (optional)
    try:
        import mlx.core as mx  # type: ignore
        checks["mlx"] = {"available": True, "version": getattr(mx, "__version__", "unknown")}
    except Exception as e:  # pragma: no cover - environment dependent
        checks["mlx"] = {"available": False, "error": str(e)}
        errors.append("MLX not available: install 'mlx' for full functionality")

    try:
        import gymnasium as gym  # type: ignore
        checks["gymnasium"] = {"available": True, "version": getattr(gym, "__version__", "unknown")}
    except Exception as e:  # pragma: no cover
        checks["gymnasium"] = {"available": False, "error": str(e)}

    try:
        import mlx_lm  # type: ignore
        checks["mlx_lm"] = {"available": True, "version": getattr(mlx_lm, "__version__", "unknown")}
    except Exception:  # pragma: no cover
        checks["mlx_lm"] = {"available": False}

    # 2) Environment contract + reward path using a dummy tokenizer
    try:
        from textpolicy.environment.text_generation import TextGenerationEnv

        class _DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 256 for c in text]

            def decode(self, ids):
                return "".join(chr(int(i) % 256) for i in ids)

        def _reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
            return float(len(completion.split()))

        env = TextGenerationEnv(["Hello"], _reward, tokenizer=_DummyTokenizer())
        obs, info = env.reset()
        step_result = env.step("a b c")
        ok = (
            isinstance(step_result, dict)
            and {"observation", "reward", "terminated", "truncated", "info"}.issubset(step_result.keys())
            and step_result["reward"] > 0
        )
        checks["environment_contract"] = {"ok": ok}
        if not ok:
            errors.append("Environment.step did not return the required dict shape or reward")
    except Exception as e:
        checks["environment_contract"] = {"ok": False, "error": str(e)}
        errors.append(f"Environment contract failed: {e}")

    status = "ok" if not errors else "fail"

    report = {"status": status, "checks": checks, "errors": errors}
    if verbose:
        _print_report(report)
    return report


def _print_report(report: Dict[str, Any]) -> None:
    """Pretty-print validation results with high signal-to-noise."""
    status = report.get("status", "fail")
    print(f"TextPolicy validation: {status}")
    for name, detail in report.get("checks", {}).items():
        print(f"- {name}: {detail}")
    if report.get("errors"):
        print("Errors:")
        for msg in report["errors"]:
            print(f"  - {msg}")

