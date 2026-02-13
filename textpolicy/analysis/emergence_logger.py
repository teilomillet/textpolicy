# textpolicy/analysis/emergence_logger.py
"""
Generation logging for emergence analysis during GRPO training.

Captures every generation produced during training and writes two JSONL
streams: per-generation records (``generations.jsonl``) and per-step
aggregate statistics (``steps.jsonl``).
"""

import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .planning_patterns import PlanningPatternConfig, PlanningPatternDetector
from .serialization import StreamingJSONLWriter, to_json_safe

_CHAT_MARKER_RE = re.compile(r"<\|[^<>|]+\|>")


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


def _decode_clean(tokenizer: Any, token_ids: List[int]) -> str:
    """Decode token IDs while stripping chat-control markers."""
    if not token_ids:
        return ""
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(token_ids)
    except Exception:
        text = tokenizer.decode(token_ids)
    cleaned = _CHAT_MARKER_RE.sub("", str(text))
    cleaned = cleaned.replace("\u200b", "")
    return cleaned.strip()


def _split_words_with_whitespace(text: str) -> List[str]:
    return re.findall(r"\S+|\s+", text)


def _map_word_entropies(
    parts: List[str],
    entropies: List[float],
) -> List[Optional[float]]:
    mapped: List[Optional[float]] = [None] * len(parts)
    word_positions = [idx for idx, part in enumerate(parts) if not part.isspace()]
    if not word_positions or not entropies:
        return mapped

    n_words = len(word_positions)
    n_tokens = len(entropies)
    for rank, part_idx in enumerate(word_positions):
        if n_words == 1:
            tok_idx = 0
        else:
            tok_idx = int(round(rank * (n_tokens - 1) / (n_words - 1)))
        tok_idx = min(max(tok_idx, 0), n_tokens - 1)
        mapped[part_idx] = float(entropies[tok_idx])
    return mapped


def _find_gram_spans(text: str, grams: List[str]) -> List[tuple]:
    text_l = text.lower()
    spans: List[tuple] = []
    for gram in grams:
        gram_l = gram.strip().lower()
        if not gram_l:
            continue
        start = 0
        while True:
            idx = text_l.find(gram_l, start)
            if idx < 0:
                break
            end = idx + len(gram_l)
            left_ok = idx == 0 or not text_l[idx - 1].isalnum()
            right_ok = end >= len(text_l) or not text_l[end].isalnum()
            if left_ok and right_ok:
                spans.append((idx, end))
            start = idx + 1
    return spans


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _compute_gram_entropy_metrics(
    completion_text: str,
    entropies: List[float],
    strategic_grams: List[str],
) -> Dict[str, Any]:
    if not strategic_grams:
        return {
            "contains_strategic_gram": False,
            "strategic_gram_word_count": 0,
            "non_strategic_gram_word_count": 0,
            "strategic_gram_word_ratio": 0.0,
            "gram_entropy_mean": None,
            "non_gram_entropy_mean": None,
            "gram_entropy_delta": None,
        }

    parts = _split_words_with_whitespace(completion_text)
    mapped = _map_word_entropies(parts, entropies)
    spans = _find_gram_spans(completion_text, strategic_grams)

    gram_vals: List[float] = []
    non_gram_vals: List[float] = []
    gram_words = 0
    non_gram_words = 0

    cursor = 0
    for part, ent in zip(parts, mapped):
        part_start = cursor
        part_end = cursor + len(part)
        cursor = part_end
        if part.isspace():
            continue

        overlaps_gram = any(part_start < ge and part_end > gs for gs, ge in spans)
        if overlaps_gram:
            gram_words += 1
            if ent is not None:
                gram_vals.append(float(ent))
        else:
            non_gram_words += 1
            if ent is not None:
                non_gram_vals.append(float(ent))

    total_words = gram_words + non_gram_words
    gram_mean = _mean(gram_vals)
    non_gram_mean = _mean(non_gram_vals)
    delta = (
        float(gram_mean - non_gram_mean)
        if gram_mean is not None and non_gram_mean is not None
        else None
    )

    return {
        "contains_strategic_gram": bool(spans),
        "strategic_gram_word_count": gram_words,
        "non_strategic_gram_word_count": non_gram_words,
        "strategic_gram_word_ratio": (
            (gram_words / total_words) if total_words > 0 else 0.0
        ),
        "gram_entropy_mean": gram_mean,
        "non_gram_entropy_mean": non_gram_mean,
        "gram_entropy_delta": delta,
    }


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
        strategic_grams: Optional strategic grams used to compute
            entropy-on-grams diagnostics.
        metadata_extractor: Optional callable ``(example, reward) -> dict``.
            Defaults to countdown-task extractor.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        planning_config: Optional[PlanningPatternConfig] = None,
        strategic_grams: Optional[List[str]] = None,
        metadata_extractor: Optional[Callable] = None,
        max_completion_tokens: Optional[int] = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._gen_writer = StreamingJSONLWriter(self._output_dir / "generations.jsonl")
        self._step_writer = StreamingJSONLWriter(self._output_dir / "steps.jsonl")

        self._detector = PlanningPatternDetector(planning_config)
        self._strategic_grams = [
            gram.strip()
            for gram in (strategic_grams or [])
            if isinstance(gram, str) and gram.strip()
        ]
        self._extract_metadata = metadata_extractor or _default_metadata_extractor
        self._max_completion_tokens = (
            int(max_completion_tokens)
            if isinstance(max_completion_tokens, (int, float)) and float(max_completion_tokens) > 0
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(
        self,
        step: int,
        episodes: list,
        tokenizer: Any,
        examples: Optional[list] = None,
        extra_step_metrics: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Log all generations for a single training step.

        Args:
            step: Current training step index.
            episodes: List of :class:`Episode` objects (or dicts with the
                same fields: ``obs``, ``act``, ``rew``, ``logprob``).
            tokenizer: Tokenizer with a ``decode`` method.
            examples: Optional parallel list of example dicts (same length
                as *episodes*).  Used by the metadata extractor.
            extra_step_metrics: Optional scalar metrics merged into the
                per-step aggregate record (e.g. ``{"sepa_lambda": 0.4}``).

        Returns:
            Aggregated step statistics dict (also written to ``steps.jsonl``).
        """
        t0 = time.perf_counter()

        rewards: List[float] = []
        completion_lengths: List[int] = []
        planning_ratios: List[float] = []
        entropy_values: List[float] = []
        strategic_gram_word_ratios: List[float] = []
        gram_entropy_on: List[float] = []
        gram_entropy_off: List[float] = []
        gram_entropy_delta: List[float] = []
        strategic_gram_match_count = 0
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
            strategic_gram_word_ratios.append(record["strategic_gram_word_ratio"])
            if record["contains_strategic_gram"]:
                strategic_gram_match_count += 1
            if record["gram_entropy_mean"] is not None:
                gram_entropy_on.append(float(record["gram_entropy_mean"]))
            if record["non_gram_entropy_mean"] is not None:
                gram_entropy_off.append(float(record["non_gram_entropy_mean"]))
            if record["gram_entropy_delta"] is not None:
                gram_entropy_delta.append(float(record["gram_entropy_delta"]))
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
            strategic_gram_word_ratios=strategic_gram_word_ratios,
            strategic_gram_match_count=strategic_gram_match_count,
            gram_entropy_on=gram_entropy_on,
            gram_entropy_off=gram_entropy_off,
            gram_entropy_delta=gram_entropy_delta,
            correct_count=correct_count,
            total_count=total,
            elapsed_ms=elapsed_ms,
            max_tokens_limit=self._max_completion_tokens,
        )

        if isinstance(extra_step_metrics, dict):
            for key, value in extra_step_metrics.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    step_record[key] = float(value)

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
        # Support both Episode objects and plain dicts.
        # Use isinstance check instead of `or` to avoid falsy empty-list fallthrough.
        if isinstance(episode, dict):
            obs = episode.get("obs", [])
            act = episode.get("act", [])
            rew = episode.get("rew", [])
            logprob_raw = episode.get("logprob")
        else:
            obs = episode.obs
            act = episode.act
            rew = episode.rew
            logprob_raw = episode.logprob

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
        prompt_text = _decode_clean(tokenizer, prompt_tokens)
        completion_text = _decode_clean(tokenizer, completion_tokens)

        # Planning pattern detection
        planning_phrases = self._detector.detect(completion_text)
        planning_ratio = self._detector.planning_token_ratio(
            completion_text, len(completion_tokens)
        )
        gram_metrics = _compute_gram_entropy_metrics(
            completion_text=completion_text,
            entropies=entropy_per_token,
            strategic_grams=self._strategic_grams,
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
            **gram_metrics,
            "metadata": metadata,
        }

    @staticmethod
    def _build_step_record(
        step: int,
        rewards: List[float],
        completion_lengths: List[int],
        planning_ratios: List[float],
        entropy_values: List[float],
        strategic_gram_word_ratios: List[float],
        strategic_gram_match_count: int,
        gram_entropy_on: List[float],
        gram_entropy_off: List[float],
        gram_entropy_delta: List[float],
        correct_count: int,
        total_count: int,
        elapsed_ms: float,
        max_tokens_limit: Optional[int],
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

        max_tokens_hit_count = 0
        max_tokens_hit_rate = 0.0
        if isinstance(max_tokens_limit, int) and max_tokens_limit > 0:
            max_tokens_hit_count = sum(
                1 for length in completion_lengths if int(length) >= max_tokens_limit
            )
            max_tokens_hit_rate = (
                (max_tokens_hit_count / total_count) if total_count > 0 else 0.0
            )

        return {
            "step": step,
            "mean_reward": _mean(rewards),
            "std_reward": _std(rewards),
            "mean_completion_length": _mean([float(l) for l in completion_lengths]),
            "max_tokens_limit": max_tokens_limit,
            "max_tokens_hit_count": max_tokens_hit_count,
            "max_tokens_hit_rate": max_tokens_hit_rate,
            "planning_token_ratio": _mean(planning_ratios),
            "entropy_mean": _mean(entropy_values),
            "entropy_std": _std(entropy_values),
            "strategic_gram_match_rate": (
                (strategic_gram_match_count / total_count) if total_count > 0 else 0.0
            ),
            "strategic_gram_word_ratio": _mean(strategic_gram_word_ratios),
            "gram_entropy_on_mean": (
                _mean(gram_entropy_on) if gram_entropy_on else None
            ),
            "gram_entropy_off_mean": (
                _mean(gram_entropy_off) if gram_entropy_off else None
            ),
            "gram_entropy_delta": (
                _mean(gram_entropy_delta) if gram_entropy_delta else None
            ),
            "gram_entropy_pair_count": len(gram_entropy_delta),
            "correct_count": correct_count,
            "total_count": total_count,
            "logging_overhead_ms": round(elapsed_ms, 2),
        }
