# tests/test_emergence_logger.py
"""Comprehensive tests for the textpolicy.analysis emergence logging system."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from textpolicy.analysis import (
    EmergenceLogger,
    PlanningPatternConfig,
    PlanningPatternDetector,
    StreamingJSONLWriter,
    to_json_safe,
)
from textpolicy.buffer.episode import Episode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def _make_tokenizer(vocab=None):
    """Return a mock tokenizer with a simple decode method."""
    tok = MagicMock()
    tok.decode = lambda ids: " ".join(str(i) for i in ids)
    return tok


def _make_episode(prompt_tokens, completion_tokens, reward, logprobs=None):
    """Create an Episode with single-step text-gen data."""
    ep = Episode()
    ep.append(
        obs=prompt_tokens,
        act=completion_tokens,
        rew=reward,
        next_obs=completion_tokens,
        done=True,
        logprob=logprobs,
    )
    return ep


def _make_episode_dict(prompt_tokens, completion_tokens, reward, logprobs=None):
    """Create a plain dict mirroring Episode fields."""
    d = {
        "obs": [prompt_tokens],
        "act": [completion_tokens],
        "rew": [reward],
    }
    if logprobs is not None:
        d["logprob"] = [logprobs]
    return d


# ===========================================================================
# Serialization tests
# ===========================================================================

class TestToJsonSafe:
    def test_mlx_array_to_list(self):
        arr = mx.array([1, 2, 3])
        assert to_json_safe(arr) == [1, 2, 3]

    def test_nested_dict_with_mlx(self):
        data = {"a": mx.array([1.0, 2.0]), "b": {"c": mx.array([3])}}
        result = to_json_safe(data)
        assert result == {"a": [1.0, 2.0], "b": {"c": [3]}}

    def test_passthrough_native_types(self):
        data = {"x": 1, "y": "hello", "z": None, "w": True}
        assert to_json_safe(data) == data

    def test_list_with_mlx(self):
        data = [mx.array([1]), mx.array([2])]
        assert to_json_safe(data) == [[1], [2]]


class TestStreamingJSONLWriter:
    def test_creates_file_and_writes_lines(self, tmp_dir):
        path = tmp_dir / "test.jsonl"
        writer = StreamingJSONLWriter(path)
        writer.write({"a": 1})
        writer.write({"b": 2})
        writer.close()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_compact_json(self, tmp_dir):
        path = tmp_dir / "compact.jsonl"
        writer = StreamingJSONLWriter(path)
        writer.write({"key": "value"})
        writer.close()

        raw = path.read_text().strip()
        # compact separators: no spaces after , or :
        assert raw == '{"key":"value"}'

    def test_flush_readable_before_close(self, tmp_dir):
        path = tmp_dir / "flush.jsonl"
        writer = StreamingJSONLWriter(path)
        writer.write({"flushed": True})
        # Read without closing writer
        content = path.read_text().strip()
        assert json.loads(content) == {"flushed": True}
        writer.close()

    def test_lazy_open_no_file_before_write(self, tmp_dir):
        path = tmp_dir / "lazy.jsonl"
        writer = StreamingJSONLWriter(path)
        assert not path.exists()
        writer.write({"x": 1})
        assert path.exists()
        writer.close()

    def test_creates_parent_dirs(self, tmp_dir):
        path = tmp_dir / "nested" / "deep" / "file.jsonl"
        writer = StreamingJSONLWriter(path)
        writer.write({"nested": True})
        writer.close()
        assert path.exists()

    def test_close_idempotent(self, tmp_dir):
        path = tmp_dir / "idem.jsonl"
        writer = StreamingJSONLWriter(path)
        writer.write({"x": 1})
        writer.close()
        writer.close()  # Should not raise


# ===========================================================================
# Planning pattern tests
# ===========================================================================

class TestPlanningPatternConfig:
    def test_default_all_patterns(self):
        cfg = PlanningPatternConfig()
        patterns = cfg.all_patterns
        assert "wait" in patterns
        assert "let me check" in patterns
        assert "try another" in patterns
        assert "alternatively" in patterns
        assert "notice that" in patterns
        assert len(patterns) == 19  # 5+4+4+3+3 defaults from spec

    def test_custom_patterns(self):
        cfg = PlanningPatternConfig(hesitation=["ugh"], verification=[])
        assert "ugh" in cfg.all_patterns
        assert "wait" not in cfg.all_patterns


class TestPlanningPatternDetector:
    def test_detects_all_categories(self):
        detector = PlanningPatternDetector()
        text = (
            "Wait, let me think. Let me check this. "
            "Try another approach. Alternatively, notice that..."
        )
        found = detector.detect(text)
        # Should match at least one from each category
        lower_found = [f.lower() for f in found]
        assert "wait" in lower_found
        assert "let me think" in lower_found
        assert "let me check" in lower_found
        assert "try another" in lower_found
        assert "alternatively" in lower_found
        assert "notice that" in lower_found

    def test_case_insensitive_default(self):
        detector = PlanningPatternDetector()
        assert len(detector.detect("Wait")) > 0
        assert len(detector.detect("WAIT")) > 0
        assert len(detector.detect("wait")) > 0

    def test_case_sensitive_mode(self):
        cfg = PlanningPatternConfig(case_sensitive=True)
        detector = PlanningPatternDetector(cfg)
        assert len(detector.detect("wait")) > 0
        assert len(detector.detect("WAIT")) == 0

    def test_custom_config(self):
        cfg = PlanningPatternConfig(
            hesitation=["yo hold on"],
            verification=[],
            backtracking=[],
            alternatives=[],
            metacognition=[],
        )
        detector = PlanningPatternDetector(cfg)
        assert detector.detect("yo hold on please") == ["yo hold on"]
        assert detector.detect("wait a moment") == []

    def test_empty_text(self):
        detector = PlanningPatternDetector()
        assert detector.detect("") == []

    def test_no_patterns_found(self):
        detector = PlanningPatternDetector()
        assert detector.detect("the cat sat on the mat") == []

    def test_planning_token_ratio(self):
        detector = PlanningPatternDetector()
        # "let me think" = 3 words, total_tokens = 10
        ratio = detector.planning_token_ratio("let me think about it", 10)
        assert ratio == pytest.approx(0.3)

    def test_planning_token_ratio_zero_tokens(self):
        detector = PlanningPatternDetector()
        assert detector.planning_token_ratio("wait", 0) == 0.0


# ===========================================================================
# EmergenceLogger tests
# ===========================================================================

class TestEmergenceLogger:
    def test_creates_output_dir(self, tmp_dir):
        out = tmp_dir / "logs"
        logger = EmergenceLogger(output_dir=out)
        assert out.is_dir()
        logger.finish()

    def test_produces_both_jsonl_files(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1, 2], [3, 4], 1.0, logprobs=[-1.0, -0.5])
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        assert (tmp_dir / "generations.jsonl").exists()
        assert (tmp_dir / "steps.jsonl").exists()

    def test_generation_record_fields(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1, 2], [3, 4], 0.8, logprobs=[-1.0, -0.5])
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        line = (tmp_dir / "generations.jsonl").read_text().strip()
        rec = json.loads(line)

        required_fields = [
            "step", "prompt", "completion", "reward", "tokens",
            "logprobs", "entropy_per_token", "planning_phrases_found",
            "planning_token_ratio", "metadata",
        ]
        for field in required_fields:
            assert field in rec, f"Missing field: {field}"

    def test_step_record_fields(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1, 2], [3, 4], 1.0, logprobs=[-1.0, -0.5])
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        line = (tmp_dir / "steps.jsonl").read_text().strip()
        rec = json.loads(line)

        required_fields = [
            "step", "mean_reward", "std_reward", "mean_completion_length",
            "planning_token_ratio", "entropy_mean", "entropy_std",
            "correct_count", "total_count", "logging_overhead_ms",
        ]
        for field in required_fields:
            assert field in rec, f"Missing field: {field}"

    def test_entropy_proxy_is_neg_logprob(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2, 3], 1.0, logprobs=[-1.5, -0.3])
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["entropy_per_token"] == [1.5, 0.3]

    def test_planning_phrases_appear(self, tmp_dir):
        # Use a tokenizer that returns text containing planning phrases
        tok = MagicMock()
        tok.decode = lambda ids: "wait let me think about this"

        logger = EmergenceLogger(output_dir=tmp_dir)
        ep = _make_episode([1], [2, 3, 4, 5, 6], 0.5)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        lower_phrases = [p.lower() for p in rec["planning_phrases_found"]]
        assert "wait" in lower_phrases
        assert "let me think" in lower_phrases

    def test_countdown_metadata_extraction(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2], 1.0)
        examples = [{"target": 24, "numbers": [1, 2, 3, 4]}]
        logger.log_step(step=0, episodes=[ep], tokenizer=tok, examples=examples)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["metadata"]["target"] == 24
        assert rec["metadata"]["numbers"] == [1, 2, 3, 4]
        assert rec["metadata"]["correctness"] is True

    def test_no_example_empty_metadata(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2], 0.5)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["metadata"] == {}

    def test_handles_plain_dicts(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep_dict = _make_episode_dict([1, 2], [3, 4], 0.7, logprobs=[-0.5, -1.0])
        logger.log_step(step=0, episodes=[ep_dict], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["reward"] == pytest.approx(0.7)
        assert rec["logprobs"] == [-0.5, -1.0]

    def test_handles_empty_episodes(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        result = logger.log_step(step=0, episodes=[], tokenizer=tok)
        logger.finish()

        assert result["total_count"] == 0
        assert result["mean_reward"] == 0.0

    def test_handles_none_logprobs(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2, 3], 0.5, logprobs=None)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["logprobs"] == []
        assert rec["entropy_per_token"] == []

    def test_finish_closes_handles(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2], 1.0)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        # Writers should be closed (internal _file set to None)
        assert logger._gen_writer._file is None
        assert logger._step_writer._file is None

    def test_log_step_returns_aggregate(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2], 1.0, logprobs=[-0.5])
        result = logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        assert isinstance(result, dict)
        assert result["step"] == 0
        assert result["mean_reward"] == 1.0
        assert result["total_count"] == 1

    def test_multiple_steps_accumulate(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()

        for step in range(3):
            ep = _make_episode([1], [2, 3], float(step) * 0.5)
            logger.log_step(step=step, episodes=[ep], tokenizer=tok)
        logger.finish()

        gen_lines = (tmp_dir / "generations.jsonl").read_text().strip().split("\n")
        step_lines = (tmp_dir / "steps.jsonl").read_text().strip().split("\n")
        assert len(gen_lines) == 3
        assert len(step_lines) == 3

    def test_custom_metadata_extractor(self, tmp_dir):
        def custom_extractor(example, reward):
            return {"custom": True, "score": reward}

        logger = EmergenceLogger(output_dir=tmp_dir, metadata_extractor=custom_extractor)
        tok = _make_tokenizer()
        ep = _make_episode([1], [2], 0.9)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok, examples=[{"x": 1}])
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["metadata"]["custom"] is True
        assert rec["metadata"]["score"] == pytest.approx(0.9)

    def test_custom_planning_config(self, tmp_dir):
        cfg = PlanningPatternConfig(
            hesitation=["hmm ok"],
            verification=[],
            backtracking=[],
            alternatives=[],
            metacognition=[],
        )
        tok = MagicMock()
        tok.decode = lambda ids: "hmm ok this is fine"

        logger = EmergenceLogger(output_dir=tmp_dir, planning_config=cfg)
        ep = _make_episode([1], [2, 3, 4, 5], 0.5)
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert "hmm ok" in [p.lower() for p in rec["planning_phrases_found"]]

    def test_handles_mlx_arrays_in_episode(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()

        ep = Episode()
        ep.append(
            obs=mx.array([10, 20]),
            act=mx.array([30, 40]),
            rew=mx.array([0.9]),
            next_obs=mx.array([30, 40]),
            done=True,
            logprob=mx.array([-0.5, -1.0]),
        )
        logger.log_step(step=0, episodes=[ep], tokenizer=tok)
        logger.finish()

        rec = json.loads((tmp_dir / "generations.jsonl").read_text().strip())
        assert rec["tokens"] == [30, 40]
        assert rec["logprobs"] == [-0.5, -1.0]


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:
    def test_all_jsonl_parseable(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()

        for step in range(5):
            episodes = [
                _make_episode([1, 2], [3, 4, 5], step * 0.2, logprobs=[-0.5, -1.0, -0.3])
                for _ in range(4)
            ]
            logger.log_step(step=step, episodes=episodes, tokenizer=tok)
        logger.finish()

        # Every line in both files must be valid JSON
        for fname in ("generations.jsonl", "steps.jsonl"):
            path = tmp_dir / fname
            for i, line in enumerate(path.read_text().strip().split("\n")):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"{fname} line {i} is not valid JSON: {line[:80]}")

    def test_100_episodes_under_50ms(self, tmp_dir):
        logger = EmergenceLogger(output_dir=tmp_dir)
        tok = _make_tokenizer()

        episodes = [
            _make_episode(
                list(range(10)),
                list(range(20)),
                0.5,
                logprobs=[-0.5] * 20,
            )
            for _ in range(100)
        ]

        t0 = time.perf_counter()
        logger.log_step(step=0, episodes=episodes, tokenizer=tok)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.finish()

        assert elapsed_ms < 50.0, f"Logging 100 episodes took {elapsed_ms:.1f}ms (>50ms)"
