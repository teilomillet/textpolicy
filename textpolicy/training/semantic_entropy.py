# textpolicy/training/semantic_entropy.py
"""
Semantic-entropy tracking over planning tokens.

This module is intentionally objective-agnostic. It extracts planning-token
subsequences per completion, computes a cheap embedding proxy, and tracks
group-level dispersion with an EMA that can drive curriculum callbacks.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import mlx.core as mx  # type: ignore

_VALID_EMBEDDING_MODES = ("hash", "hidden_states")


def pool_planning_hidden_states(
    hidden_states: mx.array,
    planning_mask: mx.array,
    episode_lengths: Sequence[int],
    eps: float = 1e-8,
) -> List[Optional[List[float]]]:
    """Extract per-episode mean-pooled, L2-normalized planning embeddings.

    For each episode, slices the response-aligned hidden states, applies the
    planning mask, mean-pools over planning positions, and L2-normalizes.

    Args:
        hidden_states: Flat response-aligned ``[total_response_tokens,
            hidden_dim]`` hidden states from the model backbone.  These must
            already be aligned to response tokens (i.e. the prompt prefix has
            been stripped during extraction in ``compute_logprobs_batched``).
        planning_mask: Flat 1D ``[total_response_tokens]`` binary mask where
            1.0 = planning token.
        episode_lengths: Per-episode response token counts (sums to
            ``hidden_states.shape[0]`` and ``planning_mask.shape[0]``).
        eps: Numerical stability constant for L2 normalization.

    Returns:
        List of length ``len(episode_lengths)``.  Each entry is either a
        ``List[float]`` embedding (L2-normalized) or ``None`` when the episode
        has zero planning tokens.
    """
    if hidden_states.ndim != 2:
        raise ValueError(
            f"hidden_states must be 2D [total_response_tokens, hidden_dim], "
            f"got {hidden_states.ndim}D."
        )

    total_tokens = int(hidden_states.shape[0])
    expected_tokens = sum(int(l) for l in episode_lengths)
    if total_tokens != expected_tokens:
        raise ValueError(
            f"hidden_states has {total_tokens} rows but "
            f"sum(episode_lengths)={expected_tokens}."
        )

    mask_size = int(planning_mask.shape[0])
    if mask_size != expected_tokens:
        raise ValueError(
            f"planning_mask has {mask_size} tokens but "
            f"sum(episode_lengths)={expected_tokens}."
        )

    # Split flat planning_mask into per-episode segments.
    mask_list = planning_mask.tolist()

    results: List[Optional[List[float]]] = []
    cursor = 0
    for ep_len in episode_lengths:
        ep_len = int(ep_len)
        ep_mask = mask_list[cursor : cursor + ep_len]

        count = sum(1 for m in ep_mask if float(m) > 0.5)
        if count == 0:
            results.append(None)
            cursor += ep_len
            continue

        # Slice this episode's response-aligned hidden states: [ep_len, hidden_dim]
        h = hidden_states[cursor : cursor + ep_len, :]
        # Mask: [ep_len, 1] broadcast multiply
        mask_arr = mx.array(ep_mask, dtype=h.dtype).reshape(-1, 1)
        h_masked = h * mask_arr
        # Mean pool over planning positions
        mean_h = mx.sum(h_masked, axis=0) / float(count)
        # L2 normalize
        norm = mx.sqrt(mx.sum(mean_h * mean_h))
        mx.eval(norm)
        norm_val = norm.item()
        if norm_val > eps:
            mean_h = mean_h / norm
        mx.eval(mean_h)
        results.append(mean_h.tolist())
        cursor += ep_len

    return results


def _to_episode_rows(
    values: mx.array,
    episode_lengths: Sequence[int],
    *,
    field_name: str,
) -> List[List[float]]:
    """Convert 1D/2D token-aligned data into per-episode Python rows."""
    if values.ndim == 1:
        flat = values.tolist()
        expected = sum(int(length) for length in episode_lengths)
        if len(flat) != expected:
            raise ValueError(
                f"batch_data['{field_name}'] has {len(flat)} tokens but "
                f"sum(episode_lengths)={expected}."
            )
        rows: List[List[float]] = []
        cursor = 0
        for row_idx, length in enumerate(episode_lengths):
            if length < 0:
                raise ValueError(
                    f"episode_lengths[{row_idx}] must be >= 0, got {length}."
                )
            next_cursor = cursor + int(length)
            rows.append(flat[cursor:next_cursor])
            cursor = next_cursor
        return rows

    if values.ndim == 2:
        if values.shape[0] != len(episode_lengths):  # type: ignore[arg-type]
            raise ValueError(
                f"batch_data['{field_name}'].shape[0]={values.shape[0]} does not match "
                f"len(episode_lengths)={len(episode_lengths)}."
            )
        max_len = int(values.shape[1])  # type: ignore[index]
        matrix = values.tolist()
        rows = []
        for row_idx, length in enumerate(episode_lengths):
            if length < 0:
                raise ValueError(
                    f"episode_lengths[{row_idx}] must be >= 0, got {length}."
                )
            if int(length) > max_len:
                raise ValueError(
                    f"episode_lengths[{row_idx}]={length} exceeds padded width "
                    f"{max_len} for batch_data['{field_name}']."
                )
            rows.append(matrix[row_idx][: int(length)])
        return rows

    raise ValueError(
        f"batch_data['{field_name}'] must be 1D or 2D, got {values.ndim}D."
    )


def build_prompt_group_keys(
    full_sequences: mx.array,
    prompt_lengths: Sequence[int],
) -> List[Tuple[int, ...]]:
    """
    Build stable prompt keys for grouping GRPO completions by problem.

    Args:
        full_sequences: 2D padded [num_episodes, max_seq_len] prompt+response tokens.
        prompt_lengths: Per-episode prompt token counts.
    """
    if full_sequences.ndim != 2:
        raise ValueError(
            f"full_sequences must be 2D [N, max_seq_len], got {full_sequences.ndim}D."
        )
    n_episodes = int(full_sequences.shape[0])  # type: ignore[index]
    if len(prompt_lengths) != n_episodes:
        raise ValueError(
            f"prompt_lengths has {len(prompt_lengths)} entries but batch has "
            f"{n_episodes} episodes."
        )

    max_seq_len = int(full_sequences.shape[1])  # type: ignore[index]
    rows = full_sequences.tolist()
    keys: List[Tuple[int, ...]] = []
    for i, p_len in enumerate(prompt_lengths):
        p = int(p_len)
        if p < 0:
            raise ValueError(f"prompt_lengths[{i}] must be >= 0, got {p}.")
        if p > max_seq_len:
            raise ValueError(
                f"prompt_lengths[{i}]={p} exceeds padded sequence width {max_seq_len}."
            )
        keys.append(tuple(int(tok) for tok in rows[i][:p]))
    return keys


class SemanticEntropyTracker:
    """
    Track group-level planning-strategy dispersion with an EMA.

    "Semantic entropy" here is operationalized as mean pairwise cosine
    distance between per-completion planning embeddings within a prompt group.
    """

    def __init__(
        self,
        *,
        ema_decay: float = 0.99,
        stability_tol: float = 1e-3,
        stability_patience: int = 20,
        hash_bins: int = 256,
        positive_only: bool = True,
        reward_threshold: float = 0.5,
        eps: float = 1e-8,
        on_stable: Optional[Callable[[Dict[str, float]], None]] = None,
        embedding_mode: str = "hash",
    ) -> None:
        if not (0.0 <= ema_decay <= 1.0):
            raise ValueError(f"ema_decay must be in [0, 1], got {ema_decay}.")
        if stability_tol < 0.0:
            raise ValueError(
                f"stability_tol must be >= 0, got {stability_tol}."
            )
        if stability_patience < 1:
            raise ValueError(
                f"stability_patience must be >= 1, got {stability_patience}."
            )
        if hash_bins < 2:
            raise ValueError(f"hash_bins must be >= 2, got {hash_bins}.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}.")
        if embedding_mode not in _VALID_EMBEDDING_MODES:
            raise ValueError(
                f"embedding_mode must be one of {_VALID_EMBEDDING_MODES}, "
                f"got {embedding_mode!r}."
            )

        self.ema_decay = float(ema_decay)
        self.stability_tol = float(stability_tol)
        self.stability_patience = int(stability_patience)
        self.hash_bins = int(hash_bins)
        self.positive_only = bool(positive_only)
        self.reward_threshold = float(reward_threshold)
        self.eps = float(eps)
        self.on_stable = on_stable
        self.embedding_mode = embedding_mode

        self._ema: Optional[float] = None
        self._stable_steps: int = 0
        self._stability_fired: bool = False

    def _embed_planning_tokens(self, token_ids: Sequence[int]) -> List[float]:
        """Cheap fixed-width embedding via hashed token-frequency bins."""
        vec = [0.0] * self.hash_bins
        for tok in token_ids:
            idx = int(tok) % self.hash_bins
            vec[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > self.eps:
            inv_norm = 1.0 / norm
            vec = [v * inv_norm for v in vec]
        return vec

    def _mean_pairwise_cosine_distance(
        self,
        vectors: Sequence[Sequence[float]],
    ) -> Optional[float]:
        n = len(vectors)
        if n < 2:
            return None

        norms = [math.sqrt(sum(v * v for v in vec)) for vec in vectors]
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                ni, nj = norms[i], norms[j]
                if ni <= self.eps and nj <= self.eps:
                    distance = 0.0
                elif ni <= self.eps or nj <= self.eps:
                    distance = 1.0
                else:
                    dot = sum(a * b for a, b in zip(vectors[i], vectors[j]))
                    cosine = max(min(dot / (ni * nj), 1.0), -1.0)
                    distance = 1.0 - cosine
                total += distance
                pairs += 1
        if pairs == 0:
            return None
        return total / float(pairs)

    def update(
        self,
        *,
        actions: mx.array,
        planning_mask: mx.array,
        episode_lengths: Sequence[int],
        rewards: Optional[mx.array] = None,
        prompt_keys: Optional[Sequence[Hashable]] = None,
        group_ids: Optional[Sequence[Hashable]] = None,
        planning_embeddings: Optional[Sequence[Optional[Sequence[float]]]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Update semantic-entropy statistics from one training batch.

        Args:
            planning_embeddings: Optional pre-computed per-episode embeddings
                (e.g. from ``pool_planning_hidden_states``).  When provided and
                ``embedding_mode="hidden_states"``, these replace the hash
                embedding.  Each entry is a ``Sequence[float]`` or ``None``
                (when an episode has no planning tokens).

        Returns:
            Dict of scalar stats or None when the batch has insufficient
            grouped data (e.g. no groups with >=2 usable completions).
        """
        action_rows = _to_episode_rows(actions, episode_lengths, field_name="act")
        mask_rows = _to_episode_rows(
            planning_mask,
            episode_lengths,
            field_name="planning_mask",
        )
        n_episodes = len(action_rows)
        if len(mask_rows) != n_episodes:
            raise ValueError(
                f"planning mask rows ({len(mask_rows)}) do not match episode rows "
                f"({n_episodes})."
            )

        # Determine whether to use hidden-state embeddings or hash fallback.
        use_hidden_embeddings = (
            self.embedding_mode == "hidden_states"
            and planning_embeddings is not None
            and len(planning_embeddings) == n_episodes
        )

        # Build hash-based planning sequences only when needed.
        planning_sequences: Optional[List[List[int]]] = None
        if not use_hidden_embeddings:
            planning_sequences = []
            for tokens, mask in zip(action_rows, mask_rows):
                planning_sequences.append(
                    [
                        int(tok)
                        for tok, m in zip(tokens, mask)
                        if float(m) > 0.5
                    ]
                )

        usable_indices = list(range(n_episodes))
        if self.positive_only and rewards is not None:
            reward_values = rewards.tolist()
            if len(reward_values) != n_episodes:
                raise ValueError(
                    f"rewards has {len(reward_values)} entries but there are "
                    f"{n_episodes} episodes."
                )
            usable_indices = [
                i
                for i, reward in enumerate(reward_values)
                if float(reward) >= self.reward_threshold
            ]

        grouping_keys: Sequence[Hashable]
        if group_ids is not None:
            if len(group_ids) != n_episodes:
                raise ValueError(
                    f"group_ids has {len(group_ids)} entries but there are "
                    f"{n_episodes} episodes."
                )
            grouping_keys = group_ids
        elif prompt_keys is not None:
            if len(prompt_keys) != n_episodes:
                raise ValueError(
                    f"prompt_keys has {len(prompt_keys)} entries but there are "
                    f"{n_episodes} episodes."
                )
            grouping_keys = prompt_keys
        else:
            grouping_keys = [0] * n_episodes

        groups: Dict[Hashable, List[int]] = defaultdict(list)
        for idx in usable_indices:
            groups[grouping_keys[idx]].append(idx)

        group_dispersions: List[float] = []
        for indices in groups.values():
            if len(indices) < 2:
                continue

            if use_hidden_embeddings:
                assert planning_embeddings is not None  # for type checker
                # Use pre-computed hidden-state embeddings; skip None entries.
                embeds = [
                    planning_embeddings[i]
                    for i in indices
                    if planning_embeddings[i] is not None
                ]
            else:
                assert planning_sequences is not None  # for type checker
                embeds = [
                    self._embed_planning_tokens(planning_sequences[i])
                    for i in indices
                ]

            dispersion = self._mean_pairwise_cosine_distance(embeds)  # type: ignore[arg-type]
            if dispersion is not None and math.isfinite(dispersion):
                group_dispersions.append(dispersion)

        if not group_dispersions:
            return None

        batch_value = sum(group_dispersions) / float(len(group_dispersions))

        prev_ema = self._ema
        if self._ema is None:
            self._ema = batch_value
        else:
            d = self.ema_decay
            self._ema = d * self._ema + (1.0 - d) * batch_value

        if prev_ema is None:
            delta = 0.0
            stable_now = False
            self._stable_steps = 0
        else:
            delta = abs(self._ema - prev_ema)
            stable_now = delta <= self.stability_tol
            self._stable_steps = self._stable_steps + 1 if stable_now else 0

        stabilized = self._stable_steps >= self.stability_patience
        stats = {
            "semantic_entropy_batch": float(batch_value),
            "semantic_entropy_ema": float(self._ema),
            "semantic_entropy_delta": float(delta),
            "semantic_entropy_group_count": float(len(group_dispersions)),
            "semantic_entropy_stable": 1.0 if stabilized else 0.0,
            "semantic_entropy_stable_steps": float(self._stable_steps),
        }

        if stabilized and not self._stability_fired:
            self._stability_fired = True
            if self.on_stable is not None:
                self.on_stable(stats)
        elif not stabilized:
            self._stability_fired = False

        return stats

    def state_dict(self) -> Dict[str, Any]:
        """Serialize tracker state for restart continuity."""
        return {
            "ema_decay": self.ema_decay,
            "stability_tol": self.stability_tol,
            "stability_patience": self.stability_patience,
            "hash_bins": self.hash_bins,
            "positive_only": self.positive_only,
            "reward_threshold": self.reward_threshold,
            "eps": self.eps,
            "embedding_mode": self.embedding_mode,
            "ema": self._ema,
            "stable_steps": self._stable_steps,
            "stability_fired": self._stability_fired,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore runtime counters/EMA from a serialized payload."""
        if not isinstance(state, dict):
            raise ValueError(f"state must be a dict, got {type(state)!r}.")
        ema = state.get("ema")
        if ema is not None:
            try:
                ema = float(ema)
            except (TypeError, ValueError) as exc:
                raise ValueError("state['ema'] must be float or None.") from exc
            if not math.isfinite(ema):
                raise ValueError(f"state['ema'] must be finite, got {ema}.")

        stable_steps = state.get("stable_steps", self._stable_steps)
        try:
            stable_steps = int(stable_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("state['stable_steps'] must be an integer.") from exc
        if stable_steps < 0:
            raise ValueError(
                f"state['stable_steps'] must be >= 0, got {stable_steps}."
            )

        self._ema = ema
        self._stable_steps = stable_steps
        self._stability_fired = bool(
            state.get("stability_fired", self._stability_fired)
        )
