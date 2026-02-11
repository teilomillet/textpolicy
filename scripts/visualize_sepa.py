#!/usr/bin/env python3
"""
Build a standalone HTML dashboard for SEPA/entropy/strategic-gram behavior.

Inputs:
- <run_dir>/config.json
- <run_dir>/emergence/steps.jsonl
- <run_dir>/emergence/generations.jsonl

Output:
- <run_dir>/analysis/sepa_dashboard.html (default)
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from textpolicy.analysis import get_default_strategic_grams, load_strategic_grams


ColorSpec = Tuple[str, str, List[Optional[float]]]

# Temperature palette (blue -> cyan -> yellow -> orange -> red).
_HEAT_PALETTE_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (30, 64, 175)),
    (0.25, (14, 165, 233)),
    (0.50, (254, 240, 138)),
    (0.75, (251, 146, 60)),
    (1.00, (220, 38, 38)),
]
_CHAT_MARKER_RE = re.compile(r"<\|[^<>|]+\|>")
_EXPRISH_LINE_RE = re.compile(r"^[0-9\s+\-*/()×÷.=]+$")
_EXPRISH_FRAGMENT_RE = re.compile(r"[0-9\s+\-*/()×÷.=]+")
_LINE_LABEL_RE = re.compile(
    r"^(?:line\s*\d+|line\s*[ab]|final(?: expression)?|answer|reasoning|analysis)\s*[:\-]\s*",
    re.IGNORECASE,
)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    clipped_q = min(max(q, 0.0), 1.0)
    ordered = sorted(float(v) for v in values)
    idx = int(round(clipped_q * (len(ordered) - 1)))
    return ordered[idx]


def _normalize(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    finite = [v for v in values if v is not None]
    if not finite:
        return [None] * len(values)
    lo = min(finite)
    hi = max(finite)
    if hi - lo <= 1e-12:
        return [0.5 if v is not None else None for v in values]
    return [(None if v is None else (v - lo) / (hi - lo)) for v in values]


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _interp_rgb(
    left: Tuple[int, int, int],
    right: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(left, right))  # type: ignore[return-value]


def _palette_rgb_at(t: float) -> Tuple[int, int, int]:
    clamped = min(max(t, 0.0), 1.0)
    if clamped <= _HEAT_PALETTE_STOPS[0][0]:
        return _HEAT_PALETTE_STOPS[0][1]
    if clamped >= _HEAT_PALETTE_STOPS[-1][0]:
        return _HEAT_PALETTE_STOPS[-1][1]

    for (p0, c0), (p1, c1) in zip(_HEAT_PALETTE_STOPS, _HEAT_PALETTE_STOPS[1:]):
        if p0 <= clamped <= p1:
            local_t = 0.0 if p1 <= p0 else (clamped - p0) / (p1 - p0)
            return _interp_rgb(c0, c1, local_t)
    return _HEAT_PALETTE_STOPS[-1][1]


def _color_interp(value: Optional[float], vmin: float, vmax: float) -> str:
    if value is None:
        return "#f3f4f6"
    if vmax <= vmin:
        t = 0.0
    else:
        t = (value - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    return _rgb_to_hex(_palette_rgb_at(t))


def _classify_entropy(value: Optional[float], *, low_cut: float, high_cut: float) -> str:
    if value is None:
        return "tok-unknown"
    if value <= low_cut:
        return "tok-low"
    if value >= high_cut:
        return "tok-high"
    return "tok-mid"


def _split_words_with_whitespace(text: str) -> List[str]:
    return re.findall(r"\S+|\s+", text)


def _clean_completion_text(text: str) -> str:
    cleaned = _CHAT_MARKER_RE.sub("", text)
    cleaned = cleaned.replace("\u200b", "")
    return cleaned.strip()


def _contains_gram(text: str, gram: str) -> bool:
    text_l = text.lower()
    gram_l = gram.strip().lower()
    if not gram_l:
        return False

    start = 0
    while True:
        idx = text_l.find(gram_l, start)
        if idx < 0:
            return False
        end = idx + len(gram_l)
        left_ok = idx == 0 or not text_l[idx - 1].isalnum()
        right_ok = end >= len(text_l) or not text_l[end].isalnum()
        if left_ok and right_ok:
            return True
        start = idx + 1


def _find_gram_spans(text: str, grams: Sequence[str]) -> List[Tuple[int, int, str]]:
    text_l = text.lower()
    spans: List[Tuple[int, int, str]] = []
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
                spans.append((idx, end, gram))
            start = idx + 1
    return spans


def _map_word_entropies(parts: Sequence[str], entropies: Sequence[float]) -> List[Optional[float]]:
    if not parts:
        return []
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


def _strip_line_label(line: str) -> str:
    return _LINE_LABEL_RE.sub("", line).strip()


def _extract_expression_candidate(text: str) -> str:
    cleaned = _strip_line_label(text)
    if not cleaned:
        return ""

    looks_math = any(ch.isdigit() for ch in cleaned) and any(op in cleaned for op in "+-*/×÷")
    if looks_math and _EXPRISH_LINE_RE.match(cleaned):
        return " ".join(cleaned.split())

    candidates = [
        frag.strip()
        for frag in _EXPRISH_FRAGMENT_RE.findall(cleaned)
        if frag.strip()
    ]
    valid = [
        c for c in candidates
        if any(ch.isdigit() for ch in c) and any(op in c for op in "+-*/×÷")
    ]
    if not valid:
        return ""
    return " ".join(max(valid, key=len).split())


def _split_reasoning_and_expression(text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""

    expr_from_tag = ""
    expr_idx = -1
    for idx, line in enumerate(lines):
        low = line.lower()
        if low.startswith("line 2") or low.startswith("final") or low.startswith("answer"):
            expr = _extract_expression_candidate(line)
            if expr:
                expr_from_tag = expr
                expr_idx = idx
                break

    if expr_from_tag:
        reasoning_parts = [
            _strip_line_label(line)
            for idx, line in enumerate(lines)
            if idx != expr_idx
        ]
        reasoning = "\n".join(part for part in reasoning_parts if part).strip()
        return reasoning, expr_from_tag

    last_expr = _extract_expression_candidate(lines[-1])
    if last_expr:
        reasoning = "\n".join(_strip_line_label(line) for line in lines[:-1]).strip()
        if not reasoning and len(lines) == 1:
            return "", last_expr
        return reasoning, last_expr

    return "\n".join(_strip_line_label(line) for line in lines).strip(), ""


def _build_word_signals(
    text: str,
    entropies: Sequence[float],
    grams: Sequence[str],
    *,
    low_cut: float,
    high_cut: float,
) -> List[Tuple[str, bool, Optional[float], str, bool, int]]:
    parts = _split_words_with_whitespace(text)
    mapped = _map_word_entropies(parts, entropies)
    gram_spans = _find_gram_spans(text, grams)

    def _part_overlaps_gram(part_start: int, part_end: int) -> bool:
        for gs, ge, _g in gram_spans:
            if part_start < ge and part_end > gs:
                return True
        return False

    signals: List[Tuple[str, bool, Optional[float], str, bool, int]] = []
    cursor = 0
    word_idx = 0
    for part, ent in zip(parts, mapped):
        part_start = cursor
        part_end = cursor + len(part)
        cursor = part_end

        if part.isspace():
            signals.append((part, True, None, "tok-unknown", False, -1))
            continue

        entropy_class = _classify_entropy(ent, low_cut=low_cut, high_cut=high_cut)
        is_gram = _part_overlaps_gram(part_start, part_end)
        signals.append((part, False, ent, entropy_class, is_gram, word_idx))
        word_idx += 1

    return signals


def _entropy_text_html(
    text: str,
    entropies: Sequence[float],
    grams: Sequence[str],
    *,
    low_cut: float,
    high_cut: float,
) -> str:
    signals = _build_word_signals(
        text,
        entropies,
        grams,
        low_cut=low_cut,
        high_cut=high_cut,
    )
    rendered: List[str] = []
    for part, is_space, ent, entropy_class, is_gram, word_idx in signals:
        if is_space:
            rendered.append(html.escape(part))
            continue

        classes: List[str] = ["tok-word", entropy_class]
        if is_gram:
            classes.append("tok-gram")

        title = "entropy=n/a" if ent is None else f"entropy={ent:.3f}"
        if is_gram:
            title += " | s-gram"
        title += f" | word_idx={word_idx}"
        rendered.append(
            f'<span class="{" ".join(classes)}" title="{html.escape(title)}">{html.escape(part)}</span>'
        )
    return "".join(rendered)


def _token_lens_html(
    text: str,
    entropies: Sequence[float],
    grams: Sequence[str],
    *,
    low_cut: float,
    high_cut: float,
    max_words: int = 180,
) -> str:
    signals = _build_word_signals(
        text,
        entropies,
        grams,
        low_cut=low_cut,
        high_cut=high_cut,
    )
    pills: List[str] = []
    shown = 0
    hidden = 0
    for part, is_space, ent, entropy_class, is_gram, word_idx in signals:
        if is_space:
            continue
        if shown >= max_words:
            hidden += 1
            continue

        classes = ["token-pill", entropy_class]
        if is_gram:
            classes.append("tok-gram")
        ent_txt = "n/a" if ent is None else f"{ent:.3f}"
        title = f"word {word_idx} | entropy={ent_txt}"
        if is_gram:
            title += " | s-gram"
        pills.append(
            f'<span class="{" ".join(classes)}" title="{html.escape(title)}">'
            f"{html.escape(part)}"
            "</span>"
        )
        shown += 1

    if hidden > 0:
        pills.append(f"<span class='token-pill token-pill-more'>+{hidden} more</span>")
    return "<div class='token-lens'>" + "".join(pills) + "</div>"


def _index_pills(indices: Sequence[int], *, klass: str, limit: int = 16) -> str:
    if not indices:
        return "<span class='idx-empty'>-</span>"
    items = [
        f"<span class='idx-pill {klass}'>{i}</span>"
        for i in list(indices[:limit])
    ]
    if len(indices) > limit:
        items.append(f"<span class='idx-pill {klass}'>+{len(indices) - limit}</span>")
    return "".join(items)


def _entropy_strip_svg(
    entropies: Sequence[float],
    *,
    vmin: float,
    vmax: float,
    low_cut: float,
    high_cut: float,
) -> str:
    if not entropies:
        return "<div class='note'>No per-token entropy data.</div>"

    width = 980
    height = 62
    ml = 14
    mr = 14
    mt = 12
    bar_h = 20
    plot_w = width - ml - mr

    n_tokens = len(entropies)
    max_bins = 260
    n_bins = min(n_tokens, max_bins)
    bin_size = n_tokens / float(n_bins)

    binned: List[float] = []
    for i in range(n_bins):
        start = int(round(i * bin_size))
        end = int(round((i + 1) * bin_size))
        if end <= start:
            end = min(start + 1, n_tokens)
        window = entropies[start:end]
        if not window:
            window = [entropies[min(start, n_tokens - 1)]]
        binned.append(sum(window) / len(window))

    cell_w = plot_w / float(n_bins)
    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        'aria-label="Token entropy strip">'
    ]
    for i, val in enumerate(binned):
        x = ml + i * cell_w
        color = _color_interp(val, vmin=vmin, vmax=vmax)
        parts.append(
            f'<rect x="{x:.2f}" y="{mt}" width="{cell_w + 0.25:.2f}" height="{bar_h}" '
            f'fill="{color}" stroke="#ffffff" stroke-width="0.2">'
            f"<title>bin {i}: entropy={val:.3f}</title></rect>"
        )

    def _marker_x(value: float) -> float:
        if vmax <= vmin:
            return ml
        t = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
        return ml + t * plot_w

    low_x = _marker_x(low_cut)
    high_x = _marker_x(high_cut)
    parts.append(
        f'<line x1="{low_x:.2f}" y1="{mt + bar_h + 1}" x2="{low_x:.2f}" y2="{height - 8}" '
        'stroke="#1d4ed8" stroke-width="1.2" />'
    )
    parts.append(
        f'<line x1="{high_x:.2f}" y1="{mt + bar_h + 1}" x2="{high_x:.2f}" y2="{height - 8}" '
        'stroke="#b91c1c" stroke-width="1.2" />'
    )
    parts.append(
        f'<text x="{low_x:.2f}" y="{height - 2}" text-anchor="middle" font-size="10" fill="#1e3a8a">'
        "low-cut</text>"
    )
    parts.append(
        f'<text x="{high_x:.2f}" y="{height - 2}" text-anchor="middle" font-size="10" fill="#7f1d1d">'
        "high-cut</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _heat_legend_svg(
    *,
    legend_id: str,
    title: str,
    vmin: float,
    vmid: float,
    vmax: float,
    value_fmt: str,
) -> str:
    width = 440
    height = 72
    x0 = 14
    y0 = 26
    bar_w = width - 28
    bar_h = 16
    grad_id = f"{legend_id}-grad"
    stops = "\n".join(
        (
            f'<stop offset="{int(pos * 100)}%" stop-color="{_rgb_to_hex(rgb)}" />'
            for pos, rgb in _HEAT_PALETTE_STOPS
        )
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        f'aria-label="{html.escape(title)} legend">'
        f"<defs><linearGradient id=\"{html.escape(grad_id)}\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"0%\">"
        f"{stops}</linearGradient></defs>"
        f'<text x="{x0}" y="14" font-size="12" fill="#334155">{html.escape(title)}</text>'
        f'<rect x="{x0}" y="{y0}" width="{bar_w}" height="{bar_h}" fill="url(#{html.escape(grad_id)})" '
        'stroke="#cbd5e1" stroke-width="1" rx="5" ry="5" />'
        f'<text x="{x0}" y="{y0 + 34}" font-size="11" fill="#334155">low {format(vmin, value_fmt)}</text>'
        f'<text x="{x0 + bar_w / 2:.1f}" y="{y0 + 34}" text-anchor="middle" font-size="11" fill="#334155">'
        f"mid {format(vmid, value_fmt)}</text>"
        f'<text x="{x0 + bar_w}" y="{y0 + 34}" text-anchor="end" font-size="11" fill="#334155">'
        f"high {format(vmax, value_fmt)}</text>"
        "</svg>"
    )


def _line_svg(steps: Sequence[int], series: Sequence[ColorSpec]) -> str:
    width = 980
    height = 340
    ml = 70
    mr = 30
    mt = 30
    mb = 45
    plot_w = width - ml - mr
    plot_h = height - mt - mb

    min_step = min(steps) if steps else 0
    max_step = max(steps) if steps else 1
    step_span = max(max_step - min_step, 1)

    def _x(step: int) -> float:
        return ml + ((step - min_step) / step_span) * plot_w

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        'aria-label="SEPA and entropy trajectories">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
    ]

    for i in range(6):
        y = mt + (i / 5.0) * plot_h
        parts.append(
            f'<line x1="{ml:.1f}" y1="{y:.1f}" x2="{ml + plot_w:.1f}" y2="{y:.1f}" '
            'stroke="#e2e8f0" stroke-width="1" />'
        )

    norm_series = [(name, color, _normalize(values)) for name, color, values in series]
    for name, color, values in norm_series:
        points: List[Tuple[float, float]] = []
        for step, val in zip(steps, values):
            if val is None:
                continue
            x = _x(step)
            y = mt + (1.0 - val) * plot_h
            points.append((x, y))
        if len(points) >= 2:
            path = "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in points)
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.3" '
                'stroke-linecap="round" stroke-linejoin="round" />'
            )
        elif len(points) == 1:
            x, y = points[0]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}" />')

    parts.append(
        f'<line x1="{ml:.1f}" y1="{mt + plot_h:.1f}" x2="{ml + plot_w:.1f}" '
        f'y2="{mt + plot_h:.1f}" stroke="#64748b" stroke-width="1.2" />'
    )
    parts.append(
        f'<line x1="{ml:.1f}" y1="{mt:.1f}" x2="{ml:.1f}" y2="{mt + plot_h:.1f}" '
        'stroke="#64748b" stroke-width="1.2" />'
    )

    tick_steps = sorted({min_step, (min_step + max_step) // 2, max_step})
    for s in tick_steps:
        x = _x(s)
        parts.append(
            f'<line x1="{x:.1f}" y1="{mt + plot_h:.1f}" x2="{x:.1f}" '
            f'y2="{mt + plot_h + 5:.1f}" stroke="#64748b" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{mt + plot_h + 20:.1f}" text-anchor="middle" '
            'font-size="11" fill="#334155" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">'
            f"{s}</text>"
        )

    parts.append(
        f'<text x="{ml + plot_w / 2:.1f}" y="{height - 8:.1f}" text-anchor="middle" '
        'font-size="12" fill="#334155">training step</text>'
    )
    parts.append(
        f'<text x="16" y="{mt + plot_h / 2:.1f}" text-anchor="middle" '
        'font-size="12" fill="#334155" transform="rotate(-90 16 '
        f'{mt + plot_h / 2:.1f})">normalized value</text>'
    )

    legend_x = ml
    legend_y = 14
    for i, (name, color, _vals) in enumerate(norm_series):
        x0 = legend_x + i * 165
        parts.append(
            f'<line x1="{x0:.1f}" y1="{legend_y:.1f}" x2="{x0 + 20:.1f}" y2="{legend_y:.1f}" '
            f'stroke="{color}" stroke-width="2.5" />'
        )
        parts.append(
            f'<text x="{x0 + 26:.1f}" y="{legend_y + 4:.1f}" font-size="11" fill="#1e293b">'
            f"{html.escape(name)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _heatmap_svg(
    row_labels: Sequence[str],
    col_labels: Sequence[int],
    values: Sequence[Sequence[Optional[float]]],
    *,
    title: str,
    vmin: float,
    vmax: float,
    value_fmt: str,
) -> str:
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    cell_w = 18
    cell_h = 18
    ml = 220
    mt = 28
    mb = 45
    mr = 25
    width = ml + n_cols * cell_w + mr
    height = mt + n_rows * cell_h + mb

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img" '
        f'aria-label="{html.escape(title)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff" />',
    ]

    for ridx, row in enumerate(values):
        y = mt + ridx * cell_h
        for cidx, val in enumerate(row):
            x = ml + cidx * cell_w
            color = _color_interp(val, vmin=vmin, vmax=vmax)
            title_txt = "n/a" if val is None else format(val, value_fmt)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.6">'
                f"<title>{html.escape(title_txt)}</title></rect>"
            )

    for ridx, label in enumerate(row_labels):
        y = mt + ridx * cell_h + cell_h * 0.72
        parts.append(
            f'<text x="{ml - 8}" y="{y:.1f}" text-anchor="end" '
            'font-size="11" fill="#334155">'
            f"{html.escape(label)}</text>"
        )

    tick_cols = sorted({0, n_cols // 2, max(n_cols - 1, 0)})
    for cidx in tick_cols:
        if cidx >= n_cols:
            continue
        x = ml + cidx * cell_w + cell_w / 2
        step = col_labels[cidx]
        parts.append(
            f'<line x1="{x:.1f}" y1="{mt + n_rows * cell_h:.1f}" x2="{x:.1f}" '
            f'y2="{mt + n_rows * cell_h + 4:.1f}" stroke="#475569" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{mt + n_rows * cell_h + 18:.1f}" text-anchor="middle" '
            'font-size="11" fill="#334155">'
            f"{step}</text>"
        )

    parts.append(
        f'<text x="{ml + (n_cols * cell_w) / 2:.1f}" y="{height - 8:.1f}" text-anchor="middle" '
        'font-size="12" fill="#334155">training step</text>'
    )
    parts.append(
        f'<text x="{ml}" y="16" text-anchor="start" font-size="12" fill="#0f172a">{html.escape(title)}</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _load_grams(run_dir: Path, config: Dict[str, Any], grams_override: Optional[Path]) -> List[str]:
    if grams_override is not None and grams_override.exists():
        return list(load_strategic_grams(grams_override))

    configured = config.get("strategic_grams_path")
    if isinstance(configured, str) and configured:
        cfg_path = Path(configured)
        if not cfg_path.is_absolute():
            cfg_path = run_dir / cfg_path
        if cfg_path.exists():
            return list(load_strategic_grams(cfg_path))

    return list(get_default_strategic_grams())


def _compute_lambda_series(
    steps: Sequence[int],
    step_records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Optional[float]], str]:
    sepa_steps = int(config.get("sepa_steps", 0) or 0)
    schedule = str(config.get("sepa_schedule", "linear")).strip().lower()
    sepa_enabled = sepa_steps > 0 or schedule == "auto"
    if not sepa_enabled:
        return [None for _ in steps], "SEPA disabled in config."

    by_step = {int(rec.get("step", -1)): rec for rec in step_records}
    out: List[Optional[float]] = []
    has_logged_values = False
    for step in steps:
        rec = by_step.get(step, {})
        logged = _maybe_float(rec.get("sepa_lambda"))
        if logged is not None:
            has_logged_values = True
            out.append(min(max(logged, 0.0), 1.0))
            continue

        if schedule == "linear":
            if sepa_steps <= 0:
                out.append(0.0)
            else:
                out.append(min(step / float(sepa_steps), 1.0))
        elif schedule == "auto":
            # Without persisted auto-state metrics, we can only show the
            # guaranteed linear floor when sepa_steps > 0.
            if sepa_steps > 0:
                out.append(min(step / float(sepa_steps), 1.0))
            else:
                out.append(None)
        else:
            out.append(None)

    if has_logged_values:
        return out, "Using logged sepa_lambda from steps.jsonl."
    if schedule == "auto":
        if sepa_steps > 0:
            return out, "Auto mode: plotted linear lower-bound only (auto component unavailable)."
        return out, "Auto mode: sepa_lambda unavailable (not persisted in emergence logs)."
    return out, "Linear mode: sepa_lambda derived from step/sepa_steps."


def _build_research_readout(
    generations: Sequence[Dict[str, Any]],
    grams: Sequence[str],
    step_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    if not generations:
        return "<div class='note'>No completion records available.</div>"

    total = len(generations)
    steps = sorted({int(rec.get("step", 0)) for rec in generations})
    n_steps = max(len(steps), 1)
    samples_per_step = total / float(n_steps)

    rewards: List[float] = []
    expr_count = 0
    reasoning_count = 0
    gram_count = 0
    token_counts: List[int] = []
    high_ratio_vals: List[float] = []

    entropy_vals: List[float] = []
    for rec in generations:
        reward = _maybe_float(rec.get("reward"))
        if reward is not None:
            rewards.append(reward)

        completion = _clean_completion_text(str(rec.get("completion", "")))
        reasoning, expr = _split_reasoning_and_expression(completion)
        if expr:
            expr_count += 1
        if reasoning:
            reasoning_count += 1

        if any(_contains_gram(completion, g) for g in grams):
            gram_count += 1

        raw_ent = rec.get("entropy_per_token", [])
        ent = [float(v) for v in raw_ent if _maybe_float(v) is not None] if isinstance(raw_ent, list) else []
        token_counts.append(len(ent))
        entropy_vals.extend(ent)

    high_cut = _percentile(entropy_vals, 0.85) if entropy_vals else 0.0
    for rec in generations:
        raw_ent = rec.get("entropy_per_token", [])
        ent = [float(v) for v in raw_ent if _maybe_float(v) is not None] if isinstance(raw_ent, list) else []
        if ent:
            high_ratio_vals.append(sum(1 for v in ent if v >= high_cut) / float(len(ent)))

    def _pct(count: int) -> str:
        return f"{(count / float(total) * 100.0):.1f}%"

    reward_mean = (sum(rewards) / len(rewards)) if rewards else 0.0
    reward_nonneg = sum(1 for r in rewards if r >= 0.0)
    avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
    avg_high_ratio = (sum(high_ratio_vals) / len(high_ratio_vals) * 100.0) if high_ratio_vals else 0.0
    step_gram_deltas: List[float] = []
    step_gram_match_rates: List[float] = []
    for rec in (step_records or []):
        delta = _maybe_float(rec.get("gram_entropy_delta"))
        if delta is not None:
            step_gram_deltas.append(delta)
        match_rate = _maybe_float(rec.get("strategic_gram_match_rate"))
        if match_rate is not None:
            step_gram_match_rates.append(match_rate)
    step_delta_mean = (sum(step_gram_deltas) / len(step_gram_deltas)) if step_gram_deltas else None
    step_match_mean = (
        sum(step_gram_match_rates) / len(step_gram_match_rates)
        if step_gram_match_rates
        else None
    )

    readout_cards = [
        ("Completions Logged", str(total)),
        ("Unique Steps", str(len(steps))),
        ("Samples / Step", f"{samples_per_step:.2f}"),
        ("Expression Extract Rate", _pct(expr_count)),
        ("Reasoning Present Rate", _pct(reasoning_count)),
        ("Any S-Gram Match Rate", _pct(gram_count)),
        ("Mean Reward", f"{reward_mean:.4f}"),
        ("Reward >= 0 Rate", _pct(reward_nonneg)),
        ("Avg Token Length", f"{avg_tokens:.1f}"),
        ("Avg High-Entropy Share", f"{avg_high_ratio:.1f}%"),
        (
            "S-Gram Entropy Delta (step mean)",
            ("n/a" if step_delta_mean is None else f"{step_delta_mean:.4f}"),
        ),
        (
            "S-Gram Match Rate (step mean)",
            ("n/a" if step_match_mean is None else f"{step_match_mean * 100.0:.1f}%"),
        ),
    ]
    cards_html = "".join(
        "<div class='research-card'><div class='k'>{}</div><div class='v'>{}</div></div>".format(
            html.escape(k),
            html.escape(v),
        )
        for k, v in readout_cards
    )

    warnings: List[str] = []
    if samples_per_step < 2.0:
        warnings.append(
            "Low per-step sample density (<2). Interpretation risk: trajectories may look overly deterministic."
        )
    if expr_count < int(0.6 * total):
        warnings.append(
            "Expression extraction rate is low. Consider stricter answer formatting in prompts."
        )
    if avg_tokens > 140:
        warnings.append(
            "Average completion is long. Consider lower temperature or tighter stop criteria for cleaner demos."
        )
    warning_html = (
        "<ul class='research-warnings'>"
        + "".join(f"<li>{html.escape(msg)}</li>" for msg in warnings)
        + "</ul>"
        if warnings
        else "<div class='note'>No immediate data-quality warnings detected for this snapshot.</div>"
    )

    return (
        "<div class='research-grid'>"
        f"{cards_html}"
        "</div>"
        "<div class='research-note'>"
        "These diagnostics are computed from <code>emergence/generations.jsonl</code> and are intended as quick quality gates for research review."
        "</div>"
        f"{warning_html}"
    )


def _build_completion_inspector(
    generations: Sequence[Dict[str, Any]],
    grams: Sequence[str],
    *,
    max_completions: int,
    global_entropy_low_cut: float,
    global_entropy_high_cut: float,
    global_entropy_vmin: float,
    global_entropy_vmax: float,
) -> str:
    if not generations:
        return "<div class='note'>No completion records found.</div>"

    latest_step = max(int(rec.get("step", 0)) for rec in generations)
    latest = [rec for rec in generations if int(rec.get("step", 0)) == latest_step]
    latest.sort(key=lambda rec: _maybe_float(rec.get("reward")) or 0.0, reverse=True)

    selected: List[Dict[str, Any]]
    if len(latest) >= max_completions:
        selected = latest[:max_completions]
    else:
        older = [rec for rec in generations if int(rec.get("step", 0)) != latest_step]
        older.sort(
            key=lambda rec: (int(rec.get("step", 0)), _maybe_float(rec.get("reward")) or 0.0),
            reverse=True,
        )
        selected = (latest + older)[:max_completions]

    cards: List[str] = []
    for i, rec in enumerate(selected, start=1):
        completion = _clean_completion_text(str(rec.get("completion", "")))
        reasoning_text, final_expression = _split_reasoning_and_expression(completion)
        step = int(rec.get("step", 0))
        reward = _maybe_float(rec.get("reward"))
        planning_ratio = _maybe_float(rec.get("planning_token_ratio"))
        raw_entropies = rec.get("entropy_per_token", [])
        entropies = [float(v) for v in raw_entropies if _maybe_float(v) is not None] if isinstance(raw_entropies, list) else []

        low_idx = [idx for idx, val in enumerate(entropies) if val <= global_entropy_low_cut]
        high_idx = [idx for idx, val in enumerate(entropies) if val >= global_entropy_high_cut]
        mean_entropy = (sum(entropies) / len(entropies)) if entropies else None

        matched_grams: List[str] = []
        for gram in grams:
            if _contains_gram(completion, gram):
                matched_grams.append(gram)

        text_html = _entropy_text_html(
            completion,
            entropies,
            grams,
            low_cut=global_entropy_low_cut,
            high_cut=global_entropy_high_cut,
        )
        token_lens_html = _token_lens_html(
            completion,
            entropies,
            grams,
            low_cut=global_entropy_low_cut,
            high_cut=global_entropy_high_cut,
        )
        strip_svg = _entropy_strip_svg(
            entropies,
            vmin=global_entropy_vmin,
            vmax=global_entropy_vmax,
            low_cut=global_entropy_low_cut,
            high_cut=global_entropy_high_cut,
        )

        reward_txt = "n/a" if reward is None else f"{reward:.4f}"
        planning_txt = "n/a" if planning_ratio is None else f"{planning_ratio:.4f}"
        entropy_txt = "n/a" if mean_entropy is None else f"{mean_entropy:.4f}"
        high_ratio = (len(high_idx) / len(entropies) * 100.0) if entropies else 0.0
        entropy_span = (max(entropies) - min(entropies)) if entropies else 0.0
        reasoning_preview = (
            html.escape(reasoning_text[:220] + (" ..." if len(reasoning_text) > 220 else ""))
            if reasoning_text
            else "<span class='muted-copy'>No reasoning preface, direct answer style.</span>"
        )
        final_expr_html = (
            f"<div class='final-expression'>{html.escape(final_expression)}</div>"
            if final_expression
            else "<div class='final-expression final-expression-missing'>No explicit final-expression line detected.</div>"
        )
        grams_html = (
            "".join(f"<span class='gram-chip'>{html.escape(g)}</span>" for g in matched_grams[:10])
            if matched_grams
            else "<span class='idx-empty'>none</span>"
        )

        cards.append(
            "<article class='completion-card' data-mode='both'>"
            "<div class='completion-top'>"
            f"<div class='completion-head'>Completion {i} • step {step}</div>"
            "<div class='completion-signal-grid'>"
            f"<div class='signal'><span class='k'>Reward</span><span class='v'>{reward_txt}</span></div>"
            f"<div class='signal'><span class='k'>Mean Entropy</span><span class='v'>{entropy_txt}</span></div>"
            f"<div class='signal'><span class='k'>High-Entropy Share</span><span class='v'>{high_ratio:.1f}%</span></div>"
            f"<div class='signal'><span class='k'>Entropy Spread</span><span class='v'>{entropy_span:.3f}</span></div>"
            f"<div class='signal'><span class='k'>Planning Ratio</span><span class='v'>{planning_txt}</span></div>"
            f"<div class='signal'><span class='k'>Token Count</span><span class='v'>{len(entropies)}</span></div>"
            "</div>"
            "</div>"
            f"<div class='entropy-strip'>{strip_svg}</div>"
            "<div class='completion-main'>"
            "<div class='story-col'>"
            "<div class='story-label'>Reasoning Preview</div>"
            f"<div class='story-preview'>{reasoning_preview}</div>"
            "<div class='story-label'>Annotated Completion</div>"
            f"<div class='completion-text'>{text_html}</div>"
            "<div class='story-label'>Final Expression Candidate</div>"
            f"{final_expr_html}"
            "</div>"
            "<aside class='lens-col'>"
            "<div class='story-label'>Token Lens</div>"
            f"{token_lens_html}"
            "<div class='story-label'>Matched S-Grams</div>"
            f"<div class='gram-chip-row'>{grams_html}</div>"
            "</aside>"
            "</div>"
            "<div class='completion-moments'>"
            "<div class='idx-group'>"
            "<div class='idx-label'>Low-entropy token idx</div>"
            f"<div class='idx-row'>{_index_pills(low_idx, klass='idx-low')}</div>"
            "</div>"
            "<div class='idx-group'>"
            "<div class='idx-label'>High-entropy token idx</div>"
            f"<div class='idx-row'>{_index_pills(high_idx, klass='idx-high')}</div>"
            "</div>"
            "</div>"
            "</article>"
        )

    controls = (
        "<div class='inspector-controls'>"
        "<button class='mode-btn active' data-mode='both' type='button'>Combined</button>"
        "<button class='mode-btn' data-mode='grams' type='button'>S-Grams Only</button>"
        "<button class='mode-btn' data-mode='entropy' type='button'>Entropy Only</button>"
        "<button class='mode-btn' data-mode='plain' type='button'>Plain Text</button>"
        "</div>"
    )
    legend = (
        "<div class='inspector-legend'>"
        "<span class='legend-item'><span class='swatch swatch-low'></span>Low entropy</span>"
        "<span class='legend-item'><span class='swatch swatch-high'></span>High entropy</span>"
        "<span class='legend-item'><span class='swatch swatch-gram'></span>S-gram match</span>"
        "<span class='legend-item legend-muted'>Mid-entropy tokens remain neutral for readability.</span>"
        "</div>"
    )
    return controls + legend + "<div class='completion-grid'>" + "".join(cards) + "</div>"


def _build_dashboard(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    generations: List[Dict[str, Any]],
    config: Dict[str, Any],
    grams: List[str],
    *,
    top_k: int,
    max_positions: int,
    max_completions: int,
    lambda_note: str,
    lambda_values: List[Optional[float]],
) -> str:
    sorted_steps = sorted(steps, key=lambda r: int(r.get("step", 0)))
    x_steps = [int(r.get("step", i)) for i, r in enumerate(sorted_steps)]
    step_set = set(x_steps)

    by_step_gens: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in generations:
        step = int(rec.get("step", 0))
        if step in step_set:
            by_step_gens[step].append(rec)

    entropy_mean = [_maybe_float(r.get("entropy_mean")) for r in sorted_steps]
    entropy_std = [_maybe_float(r.get("entropy_std")) for r in sorted_steps]
    planning_ratio = [_maybe_float(r.get("planning_token_ratio")) for r in sorted_steps]
    reward_mean = [_maybe_float(r.get("mean_reward")) for r in sorted_steps]
    gram_match_rate = [
        _maybe_float(r.get("strategic_gram_match_rate")) for r in sorted_steps
    ]
    gram_entropy_delta = [
        _maybe_float(r.get("gram_entropy_delta")) for r in sorted_steps
    ]

    series: List[ColorSpec] = [
        ("entropy_mean", "#0f766e", entropy_mean),
        ("entropy_std", "#b45309", entropy_std),
        ("planning_ratio", "#7c3aed", planning_ratio),
        ("sepa_lambda", "#2563eb", lambda_values),
        ("gram_match_rate", "#0ea5a3", gram_match_rate),
        ("gram_entropy_delta", "#9333ea", gram_entropy_delta),
        ("mean_reward", "#dc2626", reward_mean),
    ]

    trajectory_svg = _line_svg(x_steps, series)

    gram_coverages: Dict[str, List[float]] = {}
    for gram in grams:
        row: List[float] = []
        for step in x_steps:
            comps = [
                _clean_completion_text(str(r.get("completion", "")))
                for r in by_step_gens.get(step, [])
            ]
            if not comps:
                row.append(0.0)
                continue
            hits = sum(1 for c in comps if _contains_gram(c, gram))
            row.append(hits / float(len(comps)))
        gram_coverages[gram] = row

    ranked_grams = sorted(
        gram_coverages,
        key=lambda g: (max(gram_coverages[g]) if gram_coverages[g] else 0.0, sum(gram_coverages[g])),
        reverse=True,
    )
    top_grams = ranked_grams[: max(top_k, 1)]
    gram_matrix = [gram_coverages[g] for g in top_grams]
    gram_svg = _heatmap_svg(
        row_labels=top_grams,
        col_labels=x_steps,
        values=gram_matrix,
        title=f"Strategic-gram coverage per step (top {len(top_grams)})",
        vmin=0.0,
        vmax=1.0,
        value_fmt=".2f",
    )
    gram_legend = _heat_legend_svg(
        legend_id="gram",
        title="Temperature scale for S-gram coverage",
        vmin=0.0,
        vmid=0.5,
        vmax=1.0,
        value_fmt=".2f",
    )

    entropy_rows: List[List[Optional[float]]] = []
    for step in x_steps:
        seqs: List[List[float]] = []
        for rec in by_step_gens.get(step, []):
            raw = rec.get("entropy_per_token", [])
            if isinstance(raw, list):
                seq = [_maybe_float(v) for v in raw]
                seqs.append([float(v) for v in seq if v is not None])
        row: List[Optional[float]] = []
        for pos in range(max_positions):
            vals = [seq[pos] for seq in seqs if pos < len(seq)]
            row.append((sum(vals) / len(vals)) if vals else None)
        entropy_rows.append(row)

    entropy_vals = [v for row in entropy_rows for v in row if v is not None]
    entropy_vmin = min(entropy_vals) if entropy_vals else 0.0
    entropy_vmax = max(1.0, _percentile(entropy_vals, 0.95))
    entropy_low_cut = _percentile(entropy_vals, 0.30) if entropy_vals else 0.0
    entropy_high_cut = _percentile(entropy_vals, 0.85) if entropy_vals else entropy_vmax
    entropy_vmid = (
        _percentile(entropy_vals, 0.50) if entropy_vals else entropy_vmax / 2.0
    )
    entropy_high = entropy_high_cut
    entropy_svg = _heatmap_svg(
        row_labels=[f"step {s}" for s in x_steps],
        col_labels=list(range(max_positions)),
        values=entropy_rows,
        title=f"Entropy by token position (0..{max_positions - 1})",
        vmin=0.0,
        vmax=entropy_vmax,
        value_fmt=".2f",
    )

    completion_panel_html = _build_completion_inspector(
        generations,
        grams,
        max_completions=max(max_completions, 1),
        global_entropy_low_cut=entropy_low_cut,
        global_entropy_high_cut=entropy_high_cut,
        global_entropy_vmin=entropy_vmin,
        global_entropy_vmax=entropy_vmax,
    )
    entropy_legend = _heat_legend_svg(
        legend_id="entropy",
        title="Temperature scale for token entropy",
        vmin=0.0,
        vmid=entropy_vmid,
        vmax=entropy_vmax,
        value_fmt=".2f",
    )
    research_readout_html = _build_research_readout(
        generations,
        grams,
        step_records=sorted_steps,
    )

    final_step = sorted_steps[-1] if sorted_steps else {}
    model_id = str(config.get("model_id", "unknown"))
    schedule = str(config.get("sepa_schedule", "linear"))
    sepa_steps = config.get("sepa_steps", 0)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _fmt(v: Any) -> str:
        f = _maybe_float(v)
        if f is None:
            return "n/a"
        return f"{f:.4f}"

    def _fmt_pct(v: Any) -> str:
        f = _maybe_float(v)
        if f is None:
            return "n/a"
        return f"{100.0 * f:.1f}%"

    summary_cards = [
        ("Model", model_id),
        ("SEPA schedule", f"{schedule} (sepa_steps={sepa_steps})"),
        ("Final reward", _fmt(final_step.get("mean_reward"))),
        ("Final entropy mean", _fmt(final_step.get("entropy_mean"))),
        ("Final planning ratio", _fmt(final_step.get("planning_token_ratio"))),
        ("Final S-gram match rate", _fmt_pct(final_step.get("strategic_gram_match_rate"))),
        ("Final gram entropy delta", _fmt(final_step.get("gram_entropy_delta"))),
        ("Steps logged", str(len(sorted_steps))),
    ]
    cards_html = "".join(
        "<div class='card'><div class='k'>{}</div><div class='v'>{}</div></div>".format(
            html.escape(k),
            html.escape(v),
        )
        for k, v in summary_cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SEPA Dashboard</title>
  <style>
    :root {{
      --bg: #f4efe4;
      --bg-alt: #e9f3ef;
      --panel: #fffdf8;
      --panel-strong: #ffffff;
      --ink: #102a43;
      --muted: #486581;
      --line: #d9e2ec;
      --accent: #0f766e;
      --accent-strong: #0f4c5c;
      --gram: #7c3aed;
      --low: #0ea5e9;
      --mid: #f59e0b;
      --high: #d94848;
    }}
    body {{
      margin: 0;
      background:
        radial-gradient(1500px 500px at -10% -10%, #d0f2e6 0%, transparent 55%),
        radial-gradient(1200px 420px at 110% -5%, #f7e7bf 0%, transparent 50%),
        linear-gradient(180deg, var(--bg) 0%, #f7f3e8 100%);
      color: var(--ink);
      font-family: "Avenir Next Condensed", "Trebuchet MS", "Segoe UI", sans-serif;
    }}
    .container {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 16px 34px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0.4px;
      color: var(--accent-strong);
      text-transform: uppercase;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 16px;
    }}
    .grid {{
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      margin: 14px 0 18px;
    }}
    .card {{
      background: linear-gradient(160deg, var(--panel-strong), var(--panel));
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 6px 20px rgba(20, 66, 107, 0.06);
    }}
    .card .k {{
      font-size: 12px;
      color: #6b7d90;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }}
    .card .v {{
      font-size: 15px;
      font-weight: 600;
      line-height: 1.2;
    }}
    .panel {{
      background: linear-gradient(180deg, var(--panel-strong), var(--panel));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 12px 12px;
      margin: 12px 0;
      overflow: auto;
      box-shadow: 0 10px 30px rgba(16, 42, 67, 0.07);
    }}
    .panel h2 {{
      margin: 2px 0 8px;
      font-size: 17px;
      color: var(--accent-strong);
      letter-spacing: 0.3px;
    }}

    .inspector-controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .inspector-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      margin: 0 0 10px;
      font-size: 11px;
      color: #486581;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-muted {{
      color: #627d98;
      font-style: italic;
    }}
    .swatch {{
      width: 13px;
      height: 13px;
      border-radius: 3px;
      border: 1px solid #c9d5e3;
      display: inline-block;
    }}
    .swatch-low {{
      background: rgba(14, 165, 233, 0.20);
      border-color: rgba(14, 165, 233, 0.38);
    }}
    .swatch-high {{
      background: rgba(217, 72, 72, 0.28);
      border-color: rgba(217, 72, 72, 0.45);
    }}
    .swatch-gram {{
      background: rgba(124, 58, 237, 0.10);
      border-color: rgba(124, 58, 237, 0.45);
      box-shadow: inset 0 -2px 0 var(--gram);
    }}
    .mode-btn {{
      border: 1px solid #c5d2e0;
      background: #ffffff;
      color: #334e68;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 12px;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .mode-btn:hover {{
      border-color: #8cb3c9;
      transform: translateY(-1px);
    }}
    .mode-btn.active {{
      color: #ffffff;
      border-color: #0f766e;
      background: linear-gradient(135deg, #0f766e, #0f4c5c);
    }}

    .completion-grid {{
      display: grid;
      gap: 14px;
    }}
    .completion-card {{
      border: 1px solid #d1dce8;
      border-radius: 12px;
      padding: 12px;
      background:
        radial-gradient(800px 260px at 100% 0%, rgba(14, 165, 233, 0.08), transparent 50%),
        linear-gradient(180deg, #ffffff, #f9fbfd);
    }}
    .completion-head {{
      font-size: 14px;
      font-weight: 700;
      color: #1b3b5f;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .completion-top {{
      display: grid;
      gap: 8px;
    }}
    .completion-signal-grid {{
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }}
    .signal {{
      border: 1px solid #d9e5ef;
      border-radius: 10px;
      padding: 6px 8px;
      background: #ffffff;
      display: grid;
      gap: 2px;
    }}
    .signal .k {{
      font-size: 10px;
      color: #627d98;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }}
    .signal .v {{
      font-size: 13px;
      font-weight: 700;
      color: #243b53;
    }}
    .entropy-strip {{
      margin: 10px 0 10px;
      border: 1px solid #d9e5ef;
      border-radius: 10px;
      background: #ffffff;
      padding: 4px 6px 2px;
    }}
    .completion-main {{
      display: grid;
      gap: 10px;
      grid-template-columns: 1fr;
    }}
    .story-col, .lens-col {{
      border: 1px solid #d9e5ef;
      border-radius: 10px;
      background: #ffffff;
      padding: 10px;
    }}
    .story-label {{
      font-size: 11px;
      color: #5c7084;
      text-transform: uppercase;
      letter-spacing: 0.35px;
      margin: 4px 0 6px;
      font-weight: 700;
    }}
    .story-preview {{
      font-size: 12px;
      line-height: 1.45;
      color: #334e68;
      background: #f8fbfd;
      border: 1px solid #e6edf3;
      border-radius: 8px;
      padding: 8px;
      min-height: 36px;
    }}
    .final-expression {{
      font-family: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
      font-size: 12px;
      background: #edfdf5;
      border: 1px solid #b7f0d4;
      border-radius: 8px;
      padding: 8px;
      color: #0f5132;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .final-expression-missing {{
      background: #fff8eb;
      border-color: #f5d48f;
      color: #7a4d0b;
    }}
    .completion-text {{
      white-space: pre-wrap;
      font-family: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
      font-size: 12px;
      line-height: 1.58;
      background: #f8fbfd;
      border: 1px solid #e6edf3;
      border-radius: 8px;
      padding: 10px;
      max-height: 280px;
      overflow: auto;
      word-break: break-word;
    }}

    .tok-word {{
      border-radius: 3px;
      padding: 0 1px;
      transition: background-color 0.12s ease;
    }}
    .tok-gram {{
      box-shadow: inset 0 -2px 0 var(--gram);
    }}
    .tok-low {{
      background: rgba(14, 165, 233, 0.16);
    }}
    .tok-mid {{
      background: transparent;
    }}
    .tok-high {{
      background: rgba(217, 72, 72, 0.22);
    }}
    .tok-unknown {{
      background: transparent;
    }}

    .completion-card[data-mode="grams"] .tok-word.tok-low,
    .completion-card[data-mode="grams"] .tok-word.tok-mid,
    .completion-card[data-mode="grams"] .tok-word.tok-high,
    .completion-card[data-mode="grams"] .tok-word.tok-unknown {{
      background: transparent;
    }}
    .completion-card[data-mode="entropy"] .tok-word.tok-gram {{
      box-shadow: none;
    }}
    .completion-card[data-mode="plain"] .tok-word {{
      background: transparent;
      box-shadow: none;
    }}

    .token-lens {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-height: 220px;
      overflow: auto;
      background: #f8fbfd;
      border: 1px solid #e6edf3;
      border-radius: 8px;
      padding: 8px;
    }}
    .token-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      border: 1px solid #d9e5ef;
      padding: 2px 7px;
      font-family: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
      font-size: 11px;
      line-height: 1.3;
      background: #ffffff;
      color: #243b53;
      white-space: nowrap;
      max-width: 210px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .token-pill.tok-low {{ background: rgba(14, 165, 233, 0.14); border-color: rgba(14, 165, 233, 0.35); }}
    .token-pill.tok-mid {{ background: #ffffff; border-color: #d9e5ef; }}
    .token-pill.tok-high {{ background: rgba(217, 72, 72, 0.17); border-color: rgba(217, 72, 72, 0.38); }}
    .token-pill.tok-gram {{ box-shadow: inset 0 -2px 0 var(--gram); }}
    .token-pill-more {{
      color: #627d98;
      background: #f4f7fb;
    }}
    .completion-card[data-mode="grams"] .token-pill.tok-low,
    .completion-card[data-mode="grams"] .token-pill.tok-mid,
    .completion-card[data-mode="grams"] .token-pill.tok-high {{
      background: #ffffff;
      border-color: #d9e5ef;
    }}
    .completion-card[data-mode="entropy"] .token-pill.tok-gram {{
      box-shadow: none;
    }}
    .completion-card[data-mode="plain"] .token-pill {{
      background: #ffffff;
      border-color: #d9e5ef;
      box-shadow: none;
    }}

    .gram-chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .gram-chip {{
      border: 1px solid rgba(124, 58, 237, 0.35);
      background: rgba(124, 58, 237, 0.1);
      color: #5b21b6;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 11px;
      white-space: nowrap;
    }}
    .muted-copy {{
      color: #627d98;
      font-style: italic;
    }}
    .completion-moments {{
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }}
    .idx-group {{
      border: 1px solid #d9e5ef;
      border-radius: 9px;
      background: #ffffff;
      padding: 8px;
    }}
    .idx-label {{
      font-size: 10px;
      color: #627d98;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 6px;
      font-weight: 700;
    }}
    .idx-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .idx-pill {{
      border-radius: 999px;
      padding: 2px 7px;
      font-family: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
      font-size: 10px;
      background: #f4f7fb;
      border: 1px solid #d9e5ef;
      color: #334e68;
    }}
    .idx-low {{
      border-color: rgba(14, 165, 233, 0.35);
      background: rgba(14, 165, 233, 0.12);
    }}
    .idx-high {{
      border-color: rgba(217, 72, 72, 0.4);
      background: rgba(217, 72, 72, 0.14);
    }}
    .idx-empty {{
      font-size: 11px;
      color: #627d98;
      font-style: italic;
    }}
    .research-grid {{
      display: grid;
      gap: 8px;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      margin: 2px 0 10px;
    }}
    .research-card {{
      background: #ffffff;
      border: 1px solid #dbe7f2;
      border-radius: 10px;
      padding: 8px 10px;
    }}
    .research-card .k {{
      font-size: 10px;
      color: #627d98;
      text-transform: uppercase;
      letter-spacing: 0.35px;
      margin-bottom: 3px;
    }}
    .research-card .v {{
      font-size: 14px;
      color: #1f3d5a;
      font-weight: 700;
    }}
    .research-note {{
      font-size: 12px;
      color: #486581;
      margin: 6px 0 10px;
    }}
    .research-warnings {{
      margin: 0;
      padding-left: 18px;
      color: #8a4b08;
      font-size: 12px;
    }}
    .research-warnings li {{
      margin: 4px 0;
    }}
    .note {{
      color: var(--muted);
      font-size: 12px;
      margin: 4px 0 10px;
    }}
    code {{
      font-family: "IBM Plex Mono", "JetBrains Mono", Menlo, monospace;
      font-size: 12px;
      background: #f1f5fb;
      border-radius: 4px;
      padding: 1px 5px;
    }}

    @media (min-width: 980px) {{
      .completion-main {{
        grid-template-columns: minmax(0, 1.9fr) minmax(280px, 1fr);
      }}
    }}
    @media (max-width: 700px) {{
      h1 {{
        font-size: 24px;
      }}
      .mode-btn {{
        font-size: 11px;
        padding: 5px 10px;
      }}
      .completion-signal-grid {{
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SEPA + Strategic Grams Dashboard</h1>
    <div class="sub">Run: <code>{html.escape(str(run_dir))}</code> • Generated: {html.escape(now)}</div>
    <div class="grid">{cards_html}</div>

    <div class="panel">
      <h2>Trajectory Overview</h2>
      <div class="note">{html.escape(lambda_note)} All lines are per-series normalized to [0, 1] for shape comparison. Positive <code>gram_entropy_delta</code> means gram-matched words are higher-entropy than non-gram words.</div>
      {trajectory_svg}
    </div>

    <div class="panel">
      <h2>Strategic-Gram Heatmap</h2>
      <div class="note">Cell = fraction of completions at a step containing a gram. Blue = low, yellow = medium, red = high.</div>
      {gram_legend}
      {gram_svg}
    </div>

    <div class="panel">
      <h2>Entropy Position Heatmap</h2>
      <div class="note">Rows are steps, columns are token positions in completion, values are mean token entropy proxy (-logprob). High-entropy threshold (P85) = {entropy_high:.2f}.</div>
      {entropy_legend}
      {entropy_svg}
    </div>

    <div class="panel">
      <h2>Research Readout</h2>
      <div class="note">Quick quality gates for reviewing experiment usability and narrative strength before demoing.</div>
      {research_readout_html}
    </div>

    <div class="panel">
      <h2>Completion Inspector</h2>
      <div class="note">Demo lens: switch between combined, S-gram-only, entropy-only, and plain views. This makes hesitation moments and strategic phrases easier to narrate live.</div>
      <div class="note">Thresholds: low entropy <= {entropy_low_cut:.2f}, high entropy >= {entropy_high_cut:.2f}. Mid-entropy words are intentionally neutral. Purple underline = S-gram overlap. Final expression is extracted from tagged lines when present.</div>
      {completion_panel_html}
    </div>
  </div>
  <script>
    (function () {{
      var buttons = Array.prototype.slice.call(document.querySelectorAll(".mode-btn"));
      var cards = Array.prototype.slice.call(document.querySelectorAll(".completion-card"));
      if (!buttons.length || !cards.length) {{
        return;
      }}
      function setMode(mode) {{
        cards.forEach(function(card) {{
          card.setAttribute("data-mode", mode);
        }});
        buttons.forEach(function(btn) {{
          var on = btn.getAttribute("data-mode") === mode;
          btn.classList.toggle("active", on);
        }});
      }}
      buttons.forEach(function(btn) {{
        btn.addEventListener("click", function () {{
          setMode(btn.getAttribute("data-mode") || "both");
        }});
      }});
      setMode("both");
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an HTML dashboard for SEPA and strategic-gram behavior.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Experiment output directory containing config.json and emergence/*.jsonl",
    )
    parser.add_argument(
        "--strategic-grams",
        default=None,
        help="Optional explicit strategic-grams JSON file (list or {'grams': [...]})",
    )
    parser.add_argument("--top-grams", type=int, default=12, help="Number of strategic grams to plot")
    parser.add_argument(
        "--max-positions",
        type=int,
        default=96,
        help="Max token positions to include in entropy heatmap",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path (default: <run-dir>/analysis/sepa_dashboard.html)",
    )
    parser.add_argument(
        "--max-completions",
        type=int,
        default=8,
        help="Max full completions to render in the Completion Inspector panel",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    config = _read_json(run_dir / "config.json")
    steps = _read_jsonl(run_dir / "emergence" / "steps.jsonl")
    generations = _read_jsonl(run_dir / "emergence" / "generations.jsonl")
    if not steps:
        raise ValueError(f"No steps found at {run_dir / 'emergence' / 'steps.jsonl'}")
    if not generations:
        raise ValueError(f"No generations found at {run_dir / 'emergence' / 'generations.jsonl'}")

    grams_override = Path(args.strategic_grams).resolve() if args.strategic_grams else None
    grams = _load_grams(run_dir, config, grams_override)

    sorted_steps = sorted(steps, key=lambda r: int(r.get("step", 0)))
    x_steps = [int(r.get("step", i)) for i, r in enumerate(sorted_steps)]
    lambda_values, lambda_note = _compute_lambda_series(x_steps, sorted_steps, config)

    dashboard = _build_dashboard(
        run_dir=run_dir,
        steps=steps,
        generations=generations,
        config=config,
        grams=grams,
        top_k=max(args.top_grams, 1),
        max_positions=max(args.max_positions, 8),
        max_completions=max(args.max_completions, 1),
        lambda_note=lambda_note,
        lambda_values=lambda_values,
    )

    output_path = Path(args.output).resolve() if args.output else run_dir / "analysis" / "sepa_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard)

    print(f"Wrote dashboard: {output_path}")
    print(f"Steps: {len(steps)} | Generations: {len(generations)} | Grams: {len(grams)}")


if __name__ == "__main__":
    main()
