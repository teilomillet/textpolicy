"""
Unit tests for semantic-entropy tracking utilities.
"""

import pytest
import mlx.core as mx

from textpolicy.training.semantic_entropy import (
    SemanticEntropyTracker,
    build_prompt_group_keys,
)


@pytest.mark.unit
class TestPromptGrouping:
    def test_build_prompt_group_keys_uses_prompt_lengths(self):
        full_sequences = mx.array([
            [10, 11, 1, 2, 3],
            [10, 11, 4, 5, 6],
            [20, 21, 1, 2, 3],
        ], dtype=mx.int32)
        prompt_lengths = [2, 2, 2]

        keys = build_prompt_group_keys(full_sequences, prompt_lengths)
        assert keys == [(10, 11), (10, 11), (20, 21)]


@pytest.mark.unit
class TestSemanticEntropyTracker:
    def test_update_tracks_ema_and_stability_callback(self):
        callback_payloads = []
        tracker = SemanticEntropyTracker(
            ema_decay=0.0,
            stability_tol=1e-12,
            stability_patience=1,
            hash_bins=32,
            on_stable=lambda stats: callback_payloads.append(stats),
        )

        actions = mx.array([
            [1, 2, 3],   # group A
            [1, 2, 3],   # group A (identical)
            [4, 5, 6],   # group B
            [4, 8, 6],   # group B (different)
        ], dtype=mx.int32)
        planning_mask = mx.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=mx.float32)
        episode_lengths = [3, 3, 3, 3]
        rewards = mx.array([1.0, 1.0, 1.0, 1.0], dtype=mx.float32)
        prompt_keys = [("A",), ("A",), ("B",), ("B",)]

        stats_1 = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=rewards,
            prompt_keys=prompt_keys,
        )
        assert stats_1 is not None
        assert stats_1["semantic_entropy_batch"] > 0.0
        assert stats_1["semantic_entropy_ema"] > 0.0
        assert stats_1["semantic_entropy_stable"] == 0.0

        stats_2 = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=rewards,
            prompt_keys=prompt_keys,
        )
        assert stats_2 is not None
        assert stats_2["semantic_entropy_stable"] == 1.0
        assert len(callback_payloads) == 1

        tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=rewards,
            prompt_keys=prompt_keys,
        )
        assert len(callback_payloads) == 1

    def test_positive_only_filter_can_skip_batch(self):
        tracker = SemanticEntropyTracker(
            positive_only=True,
            reward_threshold=0.5,
        )
        actions = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
        planning_mask = mx.array([[1.0, 1.0], [1.0, 1.0]], dtype=mx.float32)

        stats = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=[2, 2],
            rewards=mx.array([1.0, 0.0], dtype=mx.float32),
            prompt_keys=[("A",), ("A",)],
        )
        assert stats is None

    def test_state_dict_roundtrip(self):
        tracker = SemanticEntropyTracker(ema_decay=0.0, hash_bins=32)
        actions = mx.array([[1, 2], [2, 3]], dtype=mx.int32)
        planning_mask = mx.array([[1.0, 1.0], [1.0, 1.0]], dtype=mx.float32)

        first = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=[2, 2],
            rewards=mx.array([1.0, 1.0], dtype=mx.float32),
            prompt_keys=[("A",), ("A",)],
        )
        assert first is not None

        restored = SemanticEntropyTracker()
        restored.load_state_dict(tracker.state_dict())
        second = restored.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=[2, 2],
            rewards=mx.array([1.0, 1.0], dtype=mx.float32),
            prompt_keys=[("A",), ("A",)],
        )
        assert second is not None
        assert second["semantic_entropy_ema"] == pytest.approx(first["semantic_entropy_ema"])
