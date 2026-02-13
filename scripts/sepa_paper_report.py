#!/usr/bin/env python3
"""
Generate publication-style writeups from SEPA campaign outputs.

Outputs:
- arXiv-style technical draft
- blog-style narrative
- compact summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ArmStats:
    mean_reward_steps: Optional[float]
    overall_correct_rate: Optional[float]
    mean_gram_entropy_delta: Optional[float]
    mean_planning_ratio: Optional[float]
    final_reward_mean: Optional[float]
    final_planning_ratio_mean: Optional[float]


@dataclass
class CampaignSummary:
    name: str
    root: Path
    advantage_mode: str
    seed_count: int
    paired_success_count: int
    steps: Optional[int]
    episodes_per_step: Optional[int]
    baseline_total_generations: Optional[int]
    candidate_total_generations: Optional[int]
    litmus_status: Optional[str]
    significance_recommendation: Optional[str]
    correctness_delta: Optional[float]
    correctness_p_value: Optional[float]
    correctness_baseline_rate: Optional[float]
    correctness_candidate_rate: Optional[float]
    baseline: ArmStats
    candidate: ArmStats


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_flag_value(cmd: List[str], flag: str) -> Optional[str]:
    for i, token in enumerate(cmd):
        if token == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _load_final_step_metrics(run_dirs: List[str]) -> Tuple[Optional[float], Optional[float]]:
    final_rewards: List[float] = []
    final_planning: List[float] = []
    for run_dir in run_dirs:
        steps_path = Path(run_dir) / "emergence" / "steps.jsonl"
        if not steps_path.exists():
            continue
        lines = [ln for ln in steps_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        try:
            row = json.loads(lines[-1])
        except Exception:
            continue
        reward = _safe_float(row.get("mean_reward"))
        planning = _safe_float(row.get("planning_token_ratio"))
        if reward is not None:
            final_rewards.append(reward)
        if planning is not None:
            final_planning.append(planning)
    return _mean(final_rewards), _mean(final_planning)


def _load_campaign(root: Path, name: str) -> CampaignSummary:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis = manifest.get("analysis", {}) or {}

    litmus_json = _load_json(Path(analysis.get("litmus_json", ""))) if analysis.get("litmus_json") else None
    sig_json = _load_json(Path(analysis.get("significance_json", ""))) if analysis.get("significance_json") else None

    planned_runs = manifest.get("planned_runs", [])
    first_cmd = planned_runs[0]["command"] if planned_runs else []
    steps = _safe_float(_extract_flag_value(first_cmd, "--steps"))
    episodes_per_step = _safe_float(_extract_flag_value(first_cmd, "--episodes-per-step"))

    baseline_dirs = manifest.get("paired_baseline_dirs", [])
    candidate_dirs = manifest.get("paired_candidate_dirs", [])
    baseline_final_reward, baseline_final_planning = _load_final_step_metrics(baseline_dirs)
    candidate_final_reward, candidate_final_planning = _load_final_step_metrics(candidate_dirs)

    baseline_stats = litmus_json.get("baseline", {}) if litmus_json else {}
    candidate_stats = litmus_json.get("candidate", {}) if litmus_json else {}
    rate_test = sig_json.get("rate_test", {}) if sig_json else {}

    return CampaignSummary(
        name=name,
        root=root,
        advantage_mode=str(manifest.get("advantage_mode", "unknown")),
        seed_count=len(manifest.get("seeds", [])),
        paired_success_count=int(manifest.get("paired_success_count", 0)),
        steps=int(steps) if steps is not None else None,
        episodes_per_step=int(episodes_per_step) if episodes_per_step is not None else None,
        baseline_total_generations=rate_test.get("baseline_total"),
        candidate_total_generations=rate_test.get("candidate_total"),
        litmus_status=analysis.get("litmus_status"),
        significance_recommendation=analysis.get("significance_recommendation"),
        correctness_delta=_safe_float(rate_test.get("delta")),
        correctness_p_value=_safe_float(rate_test.get("p_value")),
        correctness_baseline_rate=_safe_float(rate_test.get("baseline_rate")),
        correctness_candidate_rate=_safe_float(rate_test.get("candidate_rate")),
        baseline=ArmStats(
            mean_reward_steps=_safe_float(baseline_stats.get("mean_reward_steps")),
            overall_correct_rate=_safe_float(baseline_stats.get("overall_correct_rate")),
            mean_gram_entropy_delta=_safe_float(baseline_stats.get("mean_gram_entropy_delta")),
            mean_planning_ratio=_safe_float(baseline_stats.get("mean_planning_ratio")),
            final_reward_mean=baseline_final_reward,
            final_planning_ratio_mean=baseline_final_planning,
        ),
        candidate=ArmStats(
            mean_reward_steps=_safe_float(candidate_stats.get("mean_reward_steps")),
            overall_correct_rate=_safe_float(candidate_stats.get("overall_correct_rate")),
            mean_gram_entropy_delta=_safe_float(candidate_stats.get("mean_gram_entropy_delta")),
            mean_planning_ratio=_safe_float(candidate_stats.get("mean_planning_ratio")),
            final_reward_mean=candidate_final_reward,
            final_planning_ratio_mean=candidate_final_planning,
        ),
    )


def _fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def _delta(cand: Optional[float], base: Optional[float]) -> Optional[float]:
    if cand is None or base is None:
        return None
    return cand - base


def _continuous_deltas(summary: CampaignSummary) -> Dict[str, Optional[float]]:
    return {
        "mean_reward_steps": _delta(summary.candidate.mean_reward_steps, summary.baseline.mean_reward_steps),
        "final_reward_mean": _delta(summary.candidate.final_reward_mean, summary.baseline.final_reward_mean),
        "mean_planning_ratio": _delta(summary.candidate.mean_planning_ratio, summary.baseline.mean_planning_ratio),
        "final_planning_ratio_mean": _delta(
            summary.candidate.final_planning_ratio_mean, summary.baseline.final_planning_ratio_mean
        ),
        "mean_gram_entropy_delta": _delta(
            summary.candidate.mean_gram_entropy_delta, summary.baseline.mean_gram_entropy_delta
        ),
    }


def _directionality_count(summary: CampaignSummary) -> Tuple[int, int, Dict[str, Optional[float]]]:
    deltas = _continuous_deltas(summary)
    observed = [v for v in deltas.values() if v is not None]
    positive = [v for v in observed if v > 0]
    return len(positive), len(observed), deltas


def _episodes_per_arm(summary: CampaignSummary) -> Optional[int]:
    if summary.steps is None or summary.episodes_per_step is None:
        return None
    return summary.paired_success_count * summary.steps * summary.episodes_per_step


def _artifact_paths(summary: CampaignSummary) -> List[str]:
    manifest = _load_json(summary.root / "manifest.json") or {}
    dirs = (manifest.get("paired_baseline_dirs", []) or []) + (manifest.get("paired_candidate_dirs", []) or [])
    out: List[str] = []
    for run_dir in dirs:
        p = Path(run_dir) / "wandb" / "full_completions.jsonl"
        if p.exists():
            out.append(str(p))
    return out


def _build_summary_section(summary: CampaignSummary) -> str:
    pos_count, obs_count, deltas = _directionality_count(summary)
    eps = _episodes_per_arm(summary)
    lines: List[str] = []
    lines.append(f"### {summary.name} ({summary.advantage_mode.upper()})")
    lines.append("")
    lines.append(
        f"Paired seeds completed: {summary.paired_success_count}/{summary.seed_count}. "
        f"Episodes per arm: {eps if eps is not None else 'n/a'}."
    )
    lines.append(
        f"Correctness rate: baseline {_fmt_pct(summary.correctness_baseline_rate)} vs "
        f"candidate {_fmt_pct(summary.correctness_candidate_rate)} (delta {_fmt_pct(summary.correctness_delta)}, "
        f"p={_fmt(summary.correctness_p_value, 4)})."
    )
    lines.append(
        f"Directional continuous signals: {pos_count}/{obs_count} metrics in predicted direction "
        f"(mean reward {_fmt(deltas['mean_reward_steps'])}, final reward {_fmt(deltas['final_reward_mean'])}, "
        f"mean planning {_fmt(deltas['mean_planning_ratio'])}, final planning {_fmt(deltas['final_planning_ratio_mean'])}, "
        f"mean gram-entropy {_fmt(deltas['mean_gram_entropy_delta'])})."
    )
    lines.append(
        f"Litmus={summary.litmus_status or 'n/a'}, significance recommendation={summary.significance_recommendation or 'n/a'}."
    )
    lines.append("")
    return "\n".join(lines)


def build_arxiv_report(
    maxrl: CampaignSummary,
    grpo: Optional[CampaignSummary],
    hypothesis_text: str,
) -> str:
    lines: List[str] = []
    lines.append("# MaxRL-Conditioned SEPA for Token-Level Credit Assignment")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "We evaluate whether adding SEPA to prompt-level MaxRL improves countdown reasoning performance on "
        "`arcee-ai/Trinity-Nano-Preview`. The primary endpoint is correctness rate, with reward, planning-token "
        "usage, and strategic-gram entropy as secondary diagnostics. The campaign reports below are generated "
        "directly from run artifacts and retain uncertainty as the decision criterion."
    )
    lines.append("")
    lines.append("## 1. Hypothesis")
    lines.append("")
    lines.append(hypothesis_text.strip())
    lines.append("")
    lines.append("## 2. Experimental Setup")
    lines.append("")
    lines.append(
        "Both arms use the same model, seed-paired prompts, optimizer settings, and rollout budget. "
        "The baseline arm disables SEPA (`sepa_steps=0`) while the candidate arm enables SEPA with linear schedule. "
        "In the primary comparison, both arms use MaxRL advantage normalization."
    )
    lines.append("")
    lines.append("## 3. Primary Results")
    lines.append("")
    lines.append(_build_summary_section(maxrl))

    if grpo is not None:
        lines.append("## 4. 2x2 Context (GRPO vs MaxRL)")
        lines.append("")
        lines.append(_build_summary_section(grpo))
        lines.append(
            "The MaxRL campaign is the direct test of the proposed mechanism; GRPO serves as contextual baseline "
            "to separate normalization effects from token-level shaping effects."
        )
        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "Correctness rate remains the publication gate. Directional movement in correlated continuous metrics should "
        "be treated as supporting diagnostics rather than standalone efficacy evidence."
    )
    lines.append("")
    lines.append("## 6. Reproducibility and Logging")
    lines.append("")
    lines.append(
        "Tokenizer handling applies explicit post-load regex patching in "
        "`textpolicy/generation/mlx_generation.py`. Long completions are persisted in untruncated JSONL files "
        "and uploaded as W&B dataset artifacts to avoid table-cell truncation effects."
    )
    lines.append("")
    maxrl_artifacts = _artifact_paths(maxrl)
    if maxrl_artifacts:
        lines.append("MaxRL full-completion artifacts (local paths):")
        for path in maxrl_artifacts[:8]:
            lines.append(f"- `{path}`")
        if len(maxrl_artifacts) > 8:
            lines.append(f"- `...` ({len(maxrl_artifacts) - 8} more)")
        lines.append("")

    lines.append("## 7. Limitations")
    lines.append("")
    lines.append(
        "If p-values remain non-significant, the campaign should be reported as underpowered for the observed "
        "effect size, and sample-size scaling should follow the campaign's own significance estimates before "
        "claiming improvement."
    )
    lines.append("")
    lines.append("## 8. Conclusion")
    lines.append("")
    lines.append(
        "This report supports a transparent claim: the methodology is instrumented and reproducible, while efficacy "
        "claims must remain bounded by the primary endpoint significance."
    )
    lines.append("")
    return "\n".join(lines)


def build_blog_post(maxrl: CampaignSummary, grpo: Optional[CampaignSummary], hypothesis_text: str) -> str:
    pos_count, obs_count, deltas = _directionality_count(maxrl)
    lines: List[str] = []
    lines.append("# Did SEPA Help Under MaxRL? What We Measured and What We Can Actually Claim")
    lines.append("")
    lines.append("We tested one concrete hypothesis:")
    lines.append("")
    lines.append(f"> {hypothesis_text.strip()}")
    lines.append("")
    lines.append(
        "The key rule was simple: correctness rate is the decision metric. Everything else (reward, planning ratio, "
        "entropy behavior) is diagnostic context."
    )
    lines.append("")
    lines.append(
        f"In the MaxRL campaign, correctness moved from {_fmt_pct(maxrl.correctness_baseline_rate)} to "
        f"{_fmt_pct(maxrl.correctness_candidate_rate)} (delta {_fmt_pct(maxrl.correctness_delta)}, "
        f"p={_fmt(maxrl.correctness_p_value, 4)})."
    )
    lines.append("")
    lines.append(
        f"Across correlated secondary metrics, {pos_count}/{obs_count} moved in the expected direction "
        f"(mean reward {_fmt(deltas['mean_reward_steps'])}, final reward {_fmt(deltas['final_reward_mean'])}, "
        f"mean planning {_fmt(deltas['mean_planning_ratio'])}, final planning {_fmt(deltas['final_planning_ratio_mean'])}, "
        f"gram-entropy {_fmt(deltas['mean_gram_entropy_delta'])})."
    )
    lines.append("")
    lines.append(
        "The practical takeaway: we should not over-claim from secondary signals, but we now have a clean, auditable "
        "pipeline and enough instrumentation to scale confidently to the compute budget needed for a definitive "
        "correctness answer."
    )
    lines.append("")
    if grpo is not None:
        lines.append(
            f"We also ran a GRPO context campaign (litmus={grpo.litmus_status}, "
            f"recommendation={grpo.significance_recommendation}) to complete the 2x2 picture."
        )
        lines.append("")
    lines.append(
        "For transparency, long completions are no longer limited to W&B table preview cells: each run uploads a "
        "full-completion JSONL artifact with token IDs."
    )
    lines.append("")
    return "\n".join(lines)


def build_summary(maxrl: CampaignSummary, grpo: Optional[CampaignSummary]) -> str:
    lines: List[str] = []
    lines.append("# SEPA Campaign Summary")
    lines.append("")
    lines.append(_build_summary_section(maxrl))
    if grpo is not None:
        lines.append(_build_summary_section(grpo))
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build arXiv/blog writeups from SEPA campaign outputs.")
    p.add_argument("--maxrl-campaign-root", required=True, help="Path to executed MaxRL campaign root.")
    p.add_argument("--grpo-campaign-root", default=None, help="Optional path to executed GRPO campaign root.")
    p.add_argument(
        "--hypothesis-text",
        default=(
            "Adding token-level SEPA to prompt-level MaxRL increases correctness rate relative to MaxRL+HICRA "
            "under matched rollout budgets."
        ),
        help="Single-sentence hypothesis used in reports.",
    )
    p.add_argument(
        "--arxiv-output",
        default=None,
        help="Output path for arXiv-style markdown. Default: <maxrl-root>/analysis/arxiv_report.md",
    )
    p.add_argument(
        "--blog-output",
        default=None,
        help="Output path for blog markdown. Default: <maxrl-root>/analysis/blog_post.md",
    )
    p.add_argument(
        "--summary-output",
        default=None,
        help="Output path for compact summary markdown. Default: <maxrl-root>/analysis/paper_report.md",
    )
    return p


def main() -> int:
    args = _parser().parse_args()
    maxrl_root = Path(args.maxrl_campaign_root).expanduser().resolve()
    grpo_root = Path(args.grpo_campaign_root).expanduser().resolve() if args.grpo_campaign_root else None

    maxrl = _load_campaign(maxrl_root, name="MaxRL campaign")
    grpo = _load_campaign(grpo_root, name="GRPO campaign") if grpo_root else None

    arxiv_output = (
        Path(args.arxiv_output).expanduser().resolve()
        if args.arxiv_output
        else (maxrl_root / "analysis" / "arxiv_report.md")
    )
    blog_output = (
        Path(args.blog_output).expanduser().resolve()
        if args.blog_output
        else (maxrl_root / "analysis" / "blog_post.md")
    )
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else (maxrl_root / "analysis" / "paper_report.md")
    )

    arxiv_output.parent.mkdir(parents=True, exist_ok=True)
    blog_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    arxiv_md = build_arxiv_report(maxrl=maxrl, grpo=grpo, hypothesis_text=args.hypothesis_text)
    blog_md = build_blog_post(maxrl=maxrl, grpo=grpo, hypothesis_text=args.hypothesis_text)
    summary_md = build_summary(maxrl=maxrl, grpo=grpo)

    arxiv_output.write_text(arxiv_md, encoding="utf-8")
    blog_output.write_text(blog_md, encoding="utf-8")
    summary_output.write_text(summary_md, encoding="utf-8")

    print(f"Wrote arXiv-style report: {arxiv_output}")
    print(f"Wrote blog-style report: {blog_output}")
    print(f"Wrote summary report: {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
