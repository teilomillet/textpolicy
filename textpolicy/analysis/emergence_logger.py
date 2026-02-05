# textpolicy/analysis/emergence_logger.py
"""
Generation logging for emergence analysis during GRPO training.

Captures every generation produced during training and writes two JSONL
streams: per-generation records (``generations.jsonl``) and per-step
aggregate statistics (``steps.jsonl``).
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .planning_patterns import PlanningPatternConfig, PlanningPatternDetector
from .serialization import StreamingJSONLWriter, to_json_safe


def _flatten(value: Any) -> list:
    """Flatten an MLX array, Python list, or scalar to a plain Python list."""
    if hasattr(value, "tolist"):
        result = value.tolist()
        if isinstance(result, list):
            return result
        return [result]
    if isinstance(value, list):
        return value
    return [value]


def _default_metadata_extractor(
    example: Optional[dict],
    reward: float,
) -> dict:
    """Extract countdown-task metadata from an example dict.

    Returns ``target``, ``numbers``, and ``correctness`` (reward >= 0.99).
    """
    if example is None:
        return {}
    meta: Dict[str, Any] = {}
    if "target" in example:
        meta["target"] = to_json_safe(example["target"])
    if "numbers" in example:
        meta["numbers"] = to_json_safe(example["numbers"])
    meta["correctness"] = reward >= 0.99
    return meta


class EmergenceLogger:
    """Logs every generation during training for post-hoc emergence analysis.

    Writes two JSONL files under *output_dir*:

    * ``generations.jsonl`` — one record per generation
    * ``steps.jsonl`` — one record per training step (aggregated stats)

    Args:
        output_dir: Directory for JSONL output files (created if needed).
        planning_config: Optional :class:`PlanningPatternConfig`.
        metadata_extractor: Optional callable ``(example, reward) -> dict``.
            Defaults to countdown-task extractor.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        planning_config: Optional[PlanningPatternConfig] = None,
        metadata_extractor: Optional[Callable] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._gen_writer = StreamingJSONLWriter(self._output_dir / "generations.jsonl")
        self._step_writer = StreamingJSONLWriter(self._output_dir / "steps.jsonl")

        self._detector = PlanningPatternDetector(planning_config)
        self._extract_metadata = metadata_extractor or _default_metadata_extractor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        episodes: list,
        tokenizer: Any,
        examples: Optional[list] = None,
    ) -> dict:
        """Log all generations for a single training step.

        Args:
            step: Current training step index.
            episodes: List of :class:`Episode` objects (or dicts with the
                same fields: ``obs``, ``act``, ``rew``, ``logprob``).
            tokenizer: Tokenizer with a ``decode`` method.
            examples: Optional parallel list of example dicts (same length
                as *episodes*).  Used by the metadata extractor.

        Returns:
            Aggregated step statistics dict (also written to ``steps.jsonl``).
        """
        t0 = time.perf_counter()

        rewards: List[float] = []
        completion_lengths: List[int] = []
        planning_ratios: List[float] = []
        entropy_values: List[float] = []
        correct_count = 0

        for idx, ep in enumerate(episodes):
            record = self._process_episode(
                step=step,
                episode=ep,
                tokenizer=tokenizer,
                example=examples[idx] if examples and idx < len(examples) else None,
            )
            self._gen_writer.write(record)

            # Accumulate for step aggregate
            rewards.append(record["reward"])
            completion_lengths.append(len(record["tokens"]))
            planning_ratios.append(record["planning_token_ratio"])
            if record["entropy_per_token"]:
                entropy_values.extend(record["entropy_per_token"])
            if record.get("metadata", {}).get("correctness", False):
                correct_count += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total = len(episodes)

        step_record = self._build_step_record(
            step=step,
            rewards=rewards,
            completion_lengths=completion_lengths,
            planning_ratios=planning_ratios,
            entropy_values=entropy_values,
            correct_count=correct_count,
            total_count=total,
            elapsed_ms=elapsed_ms,
        )
        self._step_writer.write(step_record)
        return step_record

    def finish(self) -> None:
        """Close underlying file handles."""
        self._gen_writer.close()
        self._step_writer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_episode(
        self,
        step: int,
        episode: Any,
        tokenizer: Any,
        example: Optional[dict],
    ) -> dict:
        """Build a per-generation record from a single episode."""
        # Support both Episode objects and plain dicts
        obs = getattr(episode, "obs", None) or episode.get("obs", [])
        act = getattr(episode, "act", None) or episode.get("act", [])
        rew = getattr(episode, "rew", None) or episode.get("rew", [])
        logprob_raw = getattr(episode, "logprob", None)
        if logprob_raw is None and isinstance(episode, dict):
            logprob_raw = episode.get("logprob")

        # Flatten to plain Python lists
        prompt_tokens = _flatten(obs[0]) if obs else []
        completion_tokens = _flatten(act[0]) if act else []
        reward_val = float(_flatten(rew)[0]) if rew else 0.0

        # Logprobs (may be None)
        logprobs: List[float] = []
        if logprob_raw is not None and len(logprob_raw) > 0:
            logprobs = [float(v) for v in _flatten(logprob_raw[0])]

        # Entropy proxy: -logprob per token
        entropy_per_token = [-lp for lp in logprobs] if logprobs else []

        # Decode text for pattern detection
        prompt_text = tokenizer.decode(prompt_tokens) if prompt_tokens else ""
        completion_text = tokenizer.decode(completion_tokens) if completion_tokens else ""

        # Planning pattern detection
        planning_phrases = self._detector.detect(completion_text)
        planning_ratio = self._detector.planning_token_ratio(
            completion_text, len(completion_tokens)
        )

        # Metadata
        metadata = self._extract_metadata(example, reward_val)

        return {
            "step": step,
            "prompt": prompt_text,
            "completion": completion_text,
            "reward": reward_val,
            "tokens": completion_tokens,
            "logprobs": logprobs,
            "entropy_per_token": entropy_per_token,
            "planning_phrases_found": planning_phrases,
            "planning_token_ratio": planning_ratio,
            "metadata": metadata,
        }

    @staticmethod
    def _build_step_record(
        step: int,
        rewards: List[float],
        completion_lengths: List[int],
        planning_ratios: List[float],
        entropy_values: List[float],
        correct_count: int,
        total_count: int,
        elapsed_ms: float,
    ) -> dict:
        """Compute aggregate statistics for a training step."""
        import math

        def _mean(xs: list) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        def _std(xs: list) -> float:
            if len(xs) < 2:
                return 0.0
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        return {
            "step": step,
            "mean_reward": _mean(rewards),
            "std_reward": _std(rewards),
            "mean_completion_length": _mean([float(l) for l in completion_lengths]),
            "planning_token_ratio": _mean(planning_ratios),
            "entropy_mean": _mean(entropy_values),
            "entropy_std": _std(entropy_values),
            "correct_count": correct_count,
            "total_count": total_count,
            "logging_overhead_ms": round(elapsed_ms, 2),
        }
