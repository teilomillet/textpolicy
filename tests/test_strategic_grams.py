"""
Tests for Strategic Gram Mining Pipeline (Issue #15).

Validates the offline analysis module that extracts strategic grams
from successful reasoning traces for downstream HICRA amplification.

Hypotheses tested:
  TestExtractNgrams:
    H1: Empty text → empty list
    H2: Text shorter than min_n → empty list
    H3: Exact extraction of known n-grams
    H4: Single-n extraction (min_n == max_n)

  TestDocumentFrequency:
    H1: All documents contain gram → df = 1.0
    H2: No documents contain gram → df = 0.0
    H3: Half of documents contain gram → df = 0.5

  TestLoadGenerations:
    H1: Empty file → empty list
    H2: All below threshold → empty list
    H3: Correct filtering by reward

  TestMineStrategicGrams:
    H1: Fallback on few completions
    H2: Mocked mining produces expected structure
    H3: Missing deps → ImportError
    H4: Output dict has required keys

  TestDefaultStrategicGrams:
    H1: Non-empty list
    H2: All entries are multi-word strings

  TestSaveLoadRoundtrip:
    H1: Roundtrip identity (list format)
    H2: Simple list format loads correctly
    H3: Full metadata format loads correctly

References:
    Issue #15 — Strategic Gram Mining Pipeline
"""

import json

import pytest

from textpolicy.analysis.strategic_grams import (
    DEFAULT_STRATEGIC_GRAMS,
    compute_document_frequency,
    extract_ngrams,
    get_default_strategic_grams,
    load_generations,
    load_strategic_grams,
    mine_strategic_grams,
    save_strategic_grams,
)


# ---------------------------------------------------------------------------
# TestExtractNgrams
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExtractNgrams:
    """Validate word-level n-gram extraction."""

    def test_empty_text_returns_empty(self):
        """H1: Empty string produces no n-grams."""
        assert extract_ngrams("") == []
        assert extract_ngrams("   ") == []

    def test_short_text_returns_empty(self):
        """H2: Text with fewer words than min_n produces no n-grams."""
        assert extract_ngrams("hello world", min_n=3) == []

    def test_exact_extraction(self):
        """H3: Known text produces expected n-grams."""
        text = "a b c d"
        # min_n=3, max_n=3 → trigrams only
        result = extract_ngrams(text, min_n=3, max_n=3)
        assert result == ["a b c", "b c d"]

    def test_single_n(self):
        """H4: When min_n == max_n, only that order is produced."""
        text = "one two three four"
        result = extract_ngrams(text, min_n=2, max_n=2)
        assert result == ["one two", "two three", "three four"]

    def test_lowercases_input(self):
        """N-grams are lowercased regardless of input case."""
        result = extract_ngrams("Hello World Foo", min_n=2, max_n=2)
        assert result == ["hello world", "world foo"]
        assert all(g == g.lower() for g in result)

    def test_invalid_min_n_raises(self):
        """min_n < 1 is rejected."""
        with pytest.raises(ValueError, match="min_n"):
            extract_ngrams("a b c", min_n=0)

    def test_invalid_max_n_raises(self):
        """max_n < min_n is rejected."""
        with pytest.raises(ValueError, match="max_n"):
            extract_ngrams("a b c", min_n=3, max_n=2)


# ---------------------------------------------------------------------------
# TestDocumentFrequency
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDocumentFrequency:
    """Validate document frequency computation."""

    def test_all_contain_gram(self):
        """H1: When every document contains the gram, df = 1.0."""
        docs = ["let me think about it", "let me think twice", "let me think again"]
        df = compute_document_frequency(docs, ["let me think"])
        assert df["let me think"] == 1.0

    def test_none_contain_gram(self):
        """H2: When no document contains the gram, df = 0.0."""
        docs = ["hello world", "foo bar baz"]
        df = compute_document_frequency(docs, ["let me think"])
        assert df["let me think"] == 0.0

    def test_half_contain_gram(self):
        """H3: When half the documents contain the gram, df = 0.5."""
        docs = ["let me think about it", "totally unrelated text"]
        df = compute_document_frequency(docs, ["let me think"])
        assert df["let me think"] == 0.5

    def test_empty_completions(self):
        """Empty completions list gives df = 0.0 for all grams."""
        df = compute_document_frequency([], ["let me think"])
        assert df["let me think"] == 0.0

    def test_case_insensitive(self):
        """Document frequency matching is case-insensitive."""
        docs = ["LET ME THINK about it"]
        df = compute_document_frequency(docs, ["let me think"])
        assert df["let me think"] == 1.0


# ---------------------------------------------------------------------------
# TestLoadGenerations
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadGenerations:
    """Validate JSONL loading with reward filtering."""

    def test_empty_file(self, tmp_path):
        """H1: Empty file produces empty list."""
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_generations(f) == []

    def test_all_below_threshold(self, tmp_path):
        """H2: All records below min_reward are filtered out."""
        f = tmp_path / "low.jsonl"
        lines = [
            json.dumps({"completion": "foo", "reward": 0.1}),
            json.dumps({"completion": "bar", "reward": 0.2}),
        ]
        f.write_text("\n".join(lines))
        assert load_generations(f, min_reward=0.5) == []

    def test_correct_filtering(self, tmp_path):
        """H3: Only records meeting the threshold are returned."""
        f = tmp_path / "mixed.jsonl"
        lines = [
            json.dumps({"completion": "good", "reward": 0.8}),
            json.dumps({"completion": "bad", "reward": 0.2}),
            json.dumps({"completion": "great", "reward": 1.0}),
        ]
        f.write_text("\n".join(lines))
        result = load_generations(f, min_reward=0.5)
        assert len(result) == 2
        completions = [r["completion"] for r in result]
        assert "good" in completions
        assert "great" in completions
        assert "bad" not in completions

    def test_missing_file(self, tmp_path):
        """Non-existent path returns empty list (no crash)."""
        assert load_generations(tmp_path / "nope.jsonl") == []

    def test_missing_fields_skipped(self, tmp_path):
        """Lines missing 'completion' or 'reward' are skipped."""
        f = tmp_path / "partial.jsonl"
        lines = [
            json.dumps({"completion": "only comp"}),  # no reward
            json.dumps({"reward": 0.9}),                # no completion
            json.dumps({"completion": "ok", "reward": 0.9}),
        ]
        f.write_text("\n".join(lines))
        result = load_generations(f, min_reward=0.5)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestMineStrategicGrams
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMineStrategicGrams:
    """Validate the full mining pipeline."""

    def test_fallback_on_few_completions(self, tmp_path):
        """H1: When too few qualifying generations, returns defaults."""
        f = tmp_path / "few.jsonl"
        lines = [json.dumps({"completion": "short", "reward": 0.9})]
        f.write_text("\n".join(lines))

        result = mine_strategic_grams(f, min_completions=10)
        assert result["source"] == "default"
        assert result["grams"] == get_default_strategic_grams()

    def test_output_has_required_keys(self, tmp_path):
        """H4: Output dict always contains the expected keys."""
        f = tmp_path / "empty.jsonl"
        f.write_text("")

        result = mine_strategic_grams(f, min_completions=10)
        assert "grams" in result
        assert "source" in result
        assert "num_completions" in result
        assert "cluster_metadata" in result

    def test_mined_output_structure(self, tmp_path):
        """H2: With enough data and deps, output has 'mined' source."""
        st = pytest.importorskip("sentence_transformers")
        sklearn = pytest.importorskip("sklearn")

        f = tmp_path / "data.jsonl"
        # Generate enough completions with varied content that includes
        # n-grams we can mine
        lines = []
        for i in range(20):
            comp = f"let me think about step {i} and try another approach"
            lines.append(json.dumps({"completion": comp, "reward": 0.9}))
        f.write_text("\n".join(lines))

        result = mine_strategic_grams(
            f,
            min_completions=5,
            min_n=3,
            max_n=4,
            min_df=0.05,
            max_df=0.99,
            n_clusters=3,
            top_per_cluster=2,
        )
        assert result["source"] == "mined"
        assert isinstance(result["grams"], list)
        assert len(result["grams"]) > 0
        assert result["cluster_metadata"] is not None

    def test_missing_deps_raises(self, tmp_path, monkeypatch):
        """H3: If sentence-transformers is not installed, ImportError is raised."""
        # Create enough data to trigger the mining path
        f = tmp_path / "data.jsonl"
        lines = []
        for i in range(20):
            lines.append(json.dumps({"completion": f"text {i}", "reward": 0.9}))
        f.write_text("\n".join(lines))

        # Simulate missing sentence_transformers
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="sentence-transformers"):
            mine_strategic_grams(f, min_completions=5)


# ---------------------------------------------------------------------------
# TestDefaultStrategicGrams
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDefaultStrategicGrams:
    """Validate the curated default gram list."""

    def test_non_empty(self):
        """H1: Default list is not empty."""
        assert len(DEFAULT_STRATEGIC_GRAMS) > 0

    def test_all_multi_word(self):
        """H2: Every default gram has at least 2 words."""
        for gram in DEFAULT_STRATEGIC_GRAMS:
            words = gram.strip().split()
            assert len(words) >= 2, f"Gram '{gram}' is not multi-word"

    def test_get_default_returns_copy(self):
        """get_default_strategic_grams returns an independent copy."""
        a = get_default_strategic_grams()
        b = get_default_strategic_grams()
        assert a == b
        assert a is not b
        a.append("mutated")
        assert "mutated" not in get_default_strategic_grams()


# ---------------------------------------------------------------------------
# TestSaveLoadRoundtrip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSaveLoadRoundtrip:
    """Validate persistence roundtrip for strategic grams."""

    def test_roundtrip_list(self, tmp_path):
        """H1: Saving and loading a plain list preserves identity."""
        grams = ["let me think", "try another approach", "the key is"]
        path = tmp_path / "grams.json"
        save_strategic_grams(grams, path)
        loaded = load_strategic_grams(path)
        assert loaded == grams

    def test_load_simple_list(self, tmp_path):
        """H2: A plain JSON list file loads correctly."""
        path = tmp_path / "simple.json"
        path.write_text(json.dumps(["a b c", "d e f"]))
        loaded = load_strategic_grams(path)
        assert loaded == ["a b c", "d e f"]

    def test_load_full_metadata(self, tmp_path):
        """H3: A full metadata dict (from mine_strategic_grams) loads correctly."""
        data = {
            "grams": ["let me think", "try another"],
            "source": "mined",
            "num_completions": 50,
            "cluster_metadata": [{"cluster_id": 0, "size": 5, "top_grams": ["let me think"]}],
        }
        path = tmp_path / "meta.json"
        save_strategic_grams(data, path)
        loaded = load_strategic_grams(path)
        assert loaded == ["let me think", "try another"]

    def test_load_unrecognized_format_raises(self, tmp_path):
        """Unrecognized JSON format raises ValueError."""
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"no_grams_key": 42}))
        with pytest.raises(ValueError, match="Unrecognized"):
            load_strategic_grams(path)

    def test_load_missing_file_raises(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_strategic_grams(tmp_path / "missing.json")
