# textpolicy/analysis/strategic_grams.py
"""
Strategic gram mining pipeline for HICRA planning token identification.

Provides:
1. Curated default grams — multi-word n-grams derived from reasoning behaviors
   (hesitation, verification, backtracking, alternatives, metacognition).
2. N-gram extraction and document frequency — pure functions for text analysis.
3. Full mining pipeline — load generations, embed, cluster, filter (requires
   optional ``mining`` dependencies: sentence-transformers, scikit-learn).
4. Persistence — save/load strategic grams to/from JSON.

The default grams work out of the box with zero extra dependencies. The mining
pipeline is intended for offline analysis (not training-time).

References:
    Issue #15 — Strategic Gram Mining Pipeline
    Issue #11 — HICRA Planning Token Amplification (consumer of this module)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# 1. Curated default strategic grams
# ---------------------------------------------------------------------------

DEFAULT_STRATEGIC_GRAMS: List[str] = [
    # Hesitation — pausing to reconsider
    "wait let me",
    "let me think",
    "on second thought",
    # Verification — checking correctness
    "let me check",
    "let me verify",
    "is this right",
    "double check",
    # Backtracking — abandoning a failing path
    "try another approach",
    "go back and",
    "start over",
    "that's not right",
    "that doesn't work",
    # Alternatives — exploring other strategies
    "another way to",
    "or we could",
    "what if we",
    # Metacognition — reasoning about reasoning
    "notice that",
    "the key is",
    "the key insight",
]


COUNTDOWN_STRATEGIC_GRAMS: List[str] = [
    # Countdown-specific planning / narration patterns.
    "let me try",
    "let me check",
    "i need to",
    "we need to",
    "using the numbers",
    "use each number",
    "each number once",
    "closest to",
    "too high",
    "too low",
    "difference is",
    "sum is",
    "product is",
    "divide by",
    "multiply by",
    "try another",
    "another way",
    "that gives",
    "which gives",
    "so we get",
    "line 1",
    "line 2",
    "final expression",
    "answer is",
]


def get_default_strategic_grams() -> List[str]:
    """Return a copy of the curated default strategic grams."""
    return list(DEFAULT_STRATEGIC_GRAMS)


def get_countdown_strategic_grams() -> List[str]:
    """Return a copy of the countdown-specific strategic grams."""
    return list(COUNTDOWN_STRATEGIC_GRAMS)


# ---------------------------------------------------------------------------
# 2. N-gram extraction and document frequency
# ---------------------------------------------------------------------------

def extract_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    """
    Extract word-level n-grams from *text*.

    Tokenizes on whitespace, lowercases, and generates all n-grams with
    n in [min_n, max_n].  Returns a list (may contain duplicates if the
    same phrase appears multiple times).

    Args:
        text: Input text to extract n-grams from.
        min_n: Minimum n-gram order (inclusive). Must be >= 1.
        max_n: Maximum n-gram order (inclusive). Must be >= min_n.

    Returns:
        List of n-gram strings (lowercased, space-joined).
    """
    if not text or not text.strip():
        return []
    if min_n < 1:
        raise ValueError(f"min_n must be >= 1, got {min_n}")
    if max_n < min_n:
        raise ValueError(f"max_n ({max_n}) must be >= min_n ({min_n})")

    words = text.lower().split()
    if len(words) < min_n:
        return []

    ngrams: List[str] = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


def compute_document_frequency(
    completions: List[str],
    ngrams: List[str],
) -> Dict[str, float]:
    """
    Compute the document frequency of each *ngram* across *completions*.

    Uses word-boundary-aware matching (``\\b`` regex anchors) so that
    e.g. ``"let me"`` does **not** match inside ``"outlet member"``.
    Whitespace in the n-gram is normalized to ``\\s+`` to tolerate
    multi-space or tab-separated text.

    Args:
        completions: List of completion texts (the "documents").
        ngrams: List of n-gram strings to check.

    Returns:
        Dict mapping each n-gram to its document frequency in [0.0, 1.0].
    """
    if not completions:
        return {gram: 0.0 for gram in ngrams}

    num_docs = len(completions)
    lowered_docs = [c.lower() for c in completions]

    df: Dict[str, float] = {}
    for gram in ngrams:
        gram_lower = gram.lower()
        # Word-boundary regex: escape the gram, then replace the
        # literal "\ " (backslash-space) that re.escape inserts between
        # words with \s+ so "let  me" still matches "let me".
        #
        # Escape-level note (why r"\\s+" is correct here):
        #   replacement r"\\s+" is the 4-char string: \, \, s, +
        #   re.sub processes \\ as "emit literal backslash", then s+
        #   as literal text → output is \s+ (regex whitespace quantifier).
        #   A single-backslash r"\s+" (3 chars) would crash with
        #   re.error in Python ≥ 3.12, but that is NOT what this uses.
        escaped = re.escape(gram_lower)
        escaped = re.sub(r"\\ ", r"\\s+", escaped)
        pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        count = sum(1 for doc in lowered_docs if pattern.search(doc))
        df[gram] = count / num_docs
    return df


# ---------------------------------------------------------------------------
# 3. Full mining pipeline (requires optional deps)
# ---------------------------------------------------------------------------

def load_generations(
    jsonl_path: Union[str, Path],
    min_reward: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Load successful generations from a JSONL file.

    Each line must be a JSON object with at least a ``completion`` field.
    A ``reward`` field is used for filtering; lines without it are skipped.

    Args:
        jsonl_path: Path to the JSONL file.
        min_reward: Minimum reward threshold to keep a generation.

    Returns:
        List of dicts for generations that meet the reward threshold.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []

    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Require completion and reward fields
            if "completion" not in record or "reward" not in record:
                continue
            if record["reward"] >= min_reward:
                results.append(record)
    return results


def mine_strategic_grams(
    jsonl_path: Union[str, Path],
    min_n: int = 3,
    max_n: int = 5,
    min_df: float = 0.05,
    max_df: float = 0.8,
    min_reward: float = 0.5,
    min_completions: int = 10,
    n_clusters: int = 10,
    top_per_cluster: int = 3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """
    Full mining pipeline: load → extract → embed → cluster → filter.

    Falls back to ``DEFAULT_STRATEGIC_GRAMS`` when fewer than
    *min_completions* qualifying generations are available.

    Requires optional dependencies (``pip install textpolicy[mining]``):
    - ``sentence-transformers`` for embedding n-grams
    - ``scikit-learn`` for KMeans clustering

    Args:
        jsonl_path: Path to JSONL file with generations.
        min_n: Minimum n-gram order.
        max_n: Maximum n-gram order.
        min_df: Minimum document frequency to keep an n-gram.
        max_df: Maximum document frequency (removes overly common phrases).
        min_reward: Minimum reward for filtering generations.
        min_completions: Fall back to defaults if fewer generations.
        n_clusters: Number of KMeans clusters.
        top_per_cluster: How many n-grams to pick per cluster.
        embedding_model: SentenceTransformer model name for n-gram
                        embeddings (default ``"all-MiniLM-L6-v2"``).

    Returns:
        Dict with keys:
        - ``grams``: List[str] — the selected strategic grams.
        - ``source``: ``"mined"`` or ``"default"`` indicating origin.
        - ``num_completions``: Number of qualifying generations.
        - ``cluster_metadata``: Optional list of per-cluster info (when mined).

    Raises:
        ImportError: When sentence-transformers or scikit-learn is missing.
    """
    # Load generations
    generations = load_generations(jsonl_path, min_reward=min_reward)

    if len(generations) < min_completions:
        return {
            "grams": get_default_strategic_grams(),
            "source": "default",
            "num_completions": len(generations),
            "cluster_metadata": None,
        }

    # --- Require optional deps only on the mining path ---
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for strategic gram mining. "
            "Install it with: pip install textpolicy[mining]"
        )
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for strategic gram mining. "
            "Install it with: pip install textpolicy[mining]"
        )

    completions = [g["completion"] for g in generations]

    # Extract all n-grams and compute document frequency
    all_ngrams: set[str] = set()
    for comp in completions:
        all_ngrams.update(extract_ngrams(comp, min_n=min_n, max_n=max_n))

    ngram_list = sorted(all_ngrams)
    df = compute_document_frequency(completions, ngram_list)

    # Filter by document frequency band
    filtered = [g for g in ngram_list if min_df <= df[g] <= max_df]
    if not filtered:
        return {
            "grams": get_default_strategic_grams(),
            "source": "default",
            "num_completions": len(generations),
            "cluster_metadata": None,
        }

    # Embed and cluster
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(filtered, show_progress_bar=False)

    actual_clusters = min(n_clusters, len(filtered))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Pick top n-grams per cluster (highest document frequency)
    cluster_metadata: List[Dict[str, Any]] = []
    selected: List[str] = []
    for cluster_id in range(actual_clusters):
        members = [
            (filtered[i], df[filtered[i]])
            for i in range(len(filtered))
            if labels[i] == cluster_id
        ]
        members.sort(key=lambda x: x[1], reverse=True)
        top = [m[0] for m in members[:top_per_cluster]]
        selected.extend(top)
        cluster_metadata.append({
            "cluster_id": cluster_id,
            "size": len(members),
            "top_grams": top,
        })

    return {
        "grams": selected,
        "source": "mined",
        "num_completions": len(generations),
        "cluster_metadata": cluster_metadata,
    }


# ---------------------------------------------------------------------------
# 4. Persistence — save / load
# ---------------------------------------------------------------------------

def save_strategic_grams(data: Union[Dict[str, Any], List[str]], path: Union[str, Path]) -> None:
    """
    Save strategic grams to a JSON file.

    Accepts either:
    - A full output dict from :func:`mine_strategic_grams` (includes metadata).
    - A plain list of gram strings.

    Args:
        data: Grams or full mining output to persist.
        path: Destination file path (will be overwritten).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_strategic_grams(path: Union[str, Path]) -> List[str]:
    """
    Load strategic grams from a JSON file.

    Handles both formats:
    - A plain JSON list of strings.
    - A dict with a ``"grams"`` key (output of :func:`mine_strategic_grams`).

    Args:
        path: Path to the JSON file.

    Returns:
        List of strategic gram strings.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file format is not recognized.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "grams" in data:
        return data["grams"]
    raise ValueError(
        f"Unrecognized strategic grams format in {path}. "
        "Expected a JSON list or a dict with a 'grams' key."
    )
