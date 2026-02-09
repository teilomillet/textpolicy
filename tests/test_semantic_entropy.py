"""
Unit tests for semantic-entropy tracking utilities.
"""

import math

import pytest
import mlx.core as mx

from textpolicy.training.semantic_entropy import (
    SemanticEntropyTracker,
    build_prompt_group_keys,
    pool_planning_hidden_states,
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


@pytest.mark.unit
class TestPoolPlanningHiddenStates:
    """Verify pool_planning_hidden_states alignment, mean-pooling, and L2 norm."""

    def test_basic_mean_pooling_and_l2_norm(self):
        """H1: Mean-pool over planning tokens, L2 normalize."""
        # 2 episodes, 3 tokens each, hidden_dim=4
        # Episode 0: tokens [a, b, c], mask [1, 0, 1] → pool a and c
        # Episode 1: tokens [d, e, f], mask [1, 1, 0] → pool d and e
        hidden_states = mx.array([
            [1.0, 0.0, 0.0, 0.0],  # ep0 tok0 (planning)
            [0.0, 2.0, 0.0, 0.0],  # ep0 tok1 (not planning)
            [0.0, 0.0, 3.0, 0.0],  # ep0 tok2 (planning)
            [0.0, 0.0, 0.0, 4.0],  # ep1 tok0 (planning)
            [2.0, 0.0, 0.0, 0.0],  # ep1 tok1 (planning)
            [0.0, 0.0, 0.0, 0.0],  # ep1 tok2 (not planning)
        ], dtype=mx.float32)
        planning_mask = mx.array([1, 0, 1, 1, 1, 0], dtype=mx.float32)
        episode_lengths = [3, 3]

        result = pool_planning_hidden_states(
            hidden_states, planning_mask, episode_lengths,
        )
        assert len(result) == 2

        # Episode 0: mean of [1,0,0,0] and [0,0,3,0] = [0.5, 0, 1.5, 0]
        # L2 norm = sqrt(0.25 + 2.25) = sqrt(2.5)
        ep0 = result[0]
        assert ep0 is not None
        expected_mean = [0.5, 0.0, 1.5, 0.0]
        norm = math.sqrt(sum(v**2 for v in expected_mean))
        expected_normed = [v / norm for v in expected_mean]
        for got, want in zip(ep0, expected_normed):
            assert abs(got - want) < 1e-5

        # Verify L2 norm ≈ 1
        ep0_norm = math.sqrt(sum(v**2 for v in ep0))
        assert abs(ep0_norm - 1.0) < 1e-5

    def test_no_planning_tokens_returns_none(self):
        """H2: Episode with no planning tokens yields None."""
        hidden_states = mx.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=mx.float32)
        planning_mask = mx.array([0, 0], dtype=mx.float32)
        episode_lengths = [2]

        result = pool_planning_hidden_states(
            hidden_states, planning_mask, episode_lengths,
        )
        assert len(result) == 1
        assert result[0] is None

    def test_variable_length_episodes(self):
        """H3: Variable-length episodes are correctly segmented."""
        # ep0: 2 tokens, ep1: 3 tokens
        hidden_states = mx.array([
            [1.0, 0.0],  # ep0 tok0
            [0.0, 1.0],  # ep0 tok1
            [2.0, 0.0],  # ep1 tok0
            [0.0, 2.0],  # ep1 tok1
            [1.0, 1.0],  # ep1 tok2
        ], dtype=mx.float32)
        planning_mask = mx.array([1, 1, 1, 0, 1], dtype=mx.float32)
        episode_lengths = [2, 3]

        result = pool_planning_hidden_states(
            hidden_states, planning_mask, episode_lengths,
        )
        assert len(result) == 2
        assert result[0] is not None  # ep0: 2 planning tokens
        assert result[1] is not None  # ep1: 2 planning tokens (0 and 2)

    def test_shape_mismatch_raises(self):
        """H4: Mismatched sizes raise ValueError."""
        hidden_states = mx.array([[1.0, 2.0]], dtype=mx.float32)
        planning_mask = mx.array([1, 1], dtype=mx.float32)  # 2 tokens
        with pytest.raises(ValueError, match="hidden_states has 1 rows"):
            pool_planning_hidden_states(
                hidden_states, planning_mask, episode_lengths=[2],
            )

    def test_3d_input_raises(self):
        """H5: 3D hidden states should be rejected (we expect flat 2D)."""
        hidden_states = mx.zeros((2, 3, 4))
        planning_mask = mx.ones((6,))
        with pytest.raises(ValueError, match="2D"):
            pool_planning_hidden_states(
                hidden_states, planning_mask, episode_lengths=[3, 3],
            )


@pytest.mark.unit
class TestHiddenStateEmbeddingMode:
    """Verify embedding_mode='hidden_states' uses pre-computed embeddings."""

    def test_hidden_states_mode_uses_provided_embeddings(self):
        """H1: When planning_embeddings are provided, hash is skipped."""
        tracker = SemanticEntropyTracker(
            ema_decay=0.0,
            embedding_mode="hidden_states",
        )
        # Two episodes in the same group with different embeddings
        # → non-zero dispersion.
        actions = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
        planning_mask = mx.array([1, 1, 1, 1, 1, 1], dtype=mx.float32)
        episode_lengths = [3, 3]

        # Orthogonal embeddings → cosine distance = 1.0
        embed_a = [1.0, 0.0, 0.0]
        embed_b = [0.0, 1.0, 0.0]

        stats = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=mx.array([1.0, 1.0], dtype=mx.float32),
            planning_embeddings=[embed_a, embed_b],
        )
        assert stats is not None
        # Orthogonal vectors → cosine distance = 1.0
        assert stats["semantic_entropy_batch"] == pytest.approx(1.0, abs=1e-6)

    def test_hidden_states_mode_falls_back_to_hash(self):
        """H2: When no embeddings provided, tracker falls back to hash."""
        tracker = SemanticEntropyTracker(
            ema_decay=0.0,
            embedding_mode="hidden_states",
        )
        actions = mx.array([1, 2, 3, 4], dtype=mx.int32)
        planning_mask = mx.array([1, 1, 1, 1], dtype=mx.float32)
        episode_lengths = [2, 2]

        stats = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=mx.array([1.0, 1.0], dtype=mx.float32),
            # No planning_embeddings → fallback to hash
        )
        # Should still work (using hash path)
        assert stats is not None

    def test_hidden_states_mode_skips_none_entries(self):
        """H3: Episodes with None embeddings are excluded from dispersion."""
        tracker = SemanticEntropyTracker(
            ema_decay=0.0,
            embedding_mode="hidden_states",
        )
        actions = mx.array([1, 2, 3, 4, 5, 6], dtype=mx.int32)
        planning_mask = mx.array([1, 1, 0, 0, 1, 1], dtype=mx.float32)
        episode_lengths = [2, 2, 2]

        # Episode 1 has None (no planning tokens) → only 2 valid embeds
        stats = tracker.update(
            actions=actions,
            planning_mask=planning_mask,
            episode_lengths=episode_lengths,
            rewards=mx.array([1.0, 1.0, 1.0], dtype=mx.float32),
            planning_embeddings=[[1.0, 0.0], None, [0.0, 1.0]],
        )
        assert stats is not None

    def test_invalid_embedding_mode_raises(self):
        """H4: Invalid embedding_mode is rejected."""
        with pytest.raises(ValueError, match="embedding_mode"):
            SemanticEntropyTracker(embedding_mode="invalid")

    def test_state_dict_includes_embedding_mode(self):
        """H5: embedding_mode is serialized in state_dict."""
        tracker = SemanticEntropyTracker(embedding_mode="hidden_states")
        sd = tracker.state_dict()
        assert sd["embedding_mode"] == "hidden_states"
