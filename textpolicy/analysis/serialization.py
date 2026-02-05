# textpolicy/analysis/serialization.py
"""
JSON-safe conversion utilities and streaming JSONL writer.

Handles MLX arrays, numpy scalars, and nested structures for
serialization to JSONL format used by EmergenceLogger.
"""

import json
from pathlib import Path
from typing import Any, Union


def to_json_safe(obj: Any) -> Any:
    """Recursively convert MLX arrays, numpy scalars, etc. to JSON-native types.

    Args:
        obj: Any Python object that may contain MLX arrays or numpy types.

    Returns:
        JSON-serializable equivalent.
    """
    # MLX array → list
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return obj.tolist()

    # numpy scalar → Python scalar
    if hasattr(obj, "item") and callable(obj.item):
        return obj.item()

    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]

    # int, float, str, bool, None pass through
    return obj


class StreamingJSONLWriter:
    """Append-only JSONL writer with lazy file open and compact serialization.

    Args:
        path: Destination file path. Parent directories are created on first write.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._file = None

    def write(self, record: dict) -> None:
        """Serialize *record* as one compact JSON line, then flush."""
        if self._file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a")
        line = json.dumps(to_json_safe(record), separators=(",", ":"))
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle (idempotent)."""
        if self._file is not None:
            self._file.close()
            self._file = None
