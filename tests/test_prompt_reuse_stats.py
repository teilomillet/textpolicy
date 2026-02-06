"""Tests for repeated-prompt opportunity metrics (Issue #29)."""

import pytest
import mlx.core as mx

from textpolicy.generation.mlx_generation import compute_prompt_reuse_stats


@pytest.mark.unit
def test_prompt_reuse_stats_no_repeats():
    full_sequences = mx.array(
        [
            [10, 11, 1, 2],
            [20, 21, 3, 4],
            [30, 31, 5, 6],
        ],
        dtype=mx.int32,
    )
    prompt_lengths = [2, 2, 2]
    response_lengths = [2, 2, 2]

    stats = compute_prompt_reuse_stats(
        full_sequences,
        prompt_lengths,
        response_lengths,
    )

    assert stats["num_episodes"] == 3.0
    assert stats["unique_prompts"] == 3.0
    assert stats["duplicated_episodes"] == 0.0
    assert stats["repeat_rate"] == 0.0
    assert stats["duplicated_prompt_tokens"] == 0.0
    assert stats["prompt_token_reduction_upper_bound"] == 0.0
    assert stats["end_to_end_token_reduction_upper_bound"] == 0.0


@pytest.mark.unit
def test_prompt_reuse_stats_with_repeats():
    full_sequences = mx.array(
        [
            [10, 11, 1, 2],
            [10, 11, 3, 4],
            [20, 21, 5, 6],
        ],
        dtype=mx.int32,
    )
    prompt_lengths = [2, 2, 2]
    response_lengths = [2, 2, 2]

    stats = compute_prompt_reuse_stats(
        full_sequences,
        prompt_lengths,
        response_lengths,
    )

    assert stats["num_episodes"] == 3.0
    assert stats["unique_prompts"] == 2.0
    assert stats["repeated_prompt_groups"] == 1.0
    assert stats["duplicated_episodes"] == 1.0
    assert stats["repeat_rate"] == pytest.approx(1.0 / 3.0)
    assert stats["max_group_size"] == 2.0
    assert stats["mean_group_size"] == pytest.approx(1.5)
    assert stats["total_prompt_tokens"] == 6.0
    assert stats["duplicated_prompt_tokens"] == 2.0
    assert stats["prompt_token_reduction_upper_bound"] == pytest.approx(2.0 / 6.0)
    assert stats["total_tokens"] == 12.0
    assert stats["end_to_end_token_reduction_upper_bound"] == pytest.approx(2.0 / 12.0)


@pytest.mark.unit
def test_prompt_reuse_stats_empty_batch():
    full_sequences = mx.zeros((0, 8), dtype=mx.int32)

    stats = compute_prompt_reuse_stats(full_sequences, prompt_lengths=[], response_lengths=[])

    assert stats["num_episodes"] == 0.0
    assert stats["repeat_rate"] == 0.0
    assert stats["prompt_token_reduction_upper_bound"] == 0.0


@pytest.mark.unit
def test_prompt_reuse_stats_validation_errors():
    full_sequences = mx.array([[1, 2, 3, 4]], dtype=mx.int32)

    with pytest.raises(ValueError, match="prompt_lengths has 0 entries"):
        compute_prompt_reuse_stats(full_sequences, prompt_lengths=[])

    with pytest.raises(ValueError, match=">= 1"):
        compute_prompt_reuse_stats(full_sequences, prompt_lengths=[0])

    with pytest.raises(ValueError, match="exceeds sequence width"):
        compute_prompt_reuse_stats(full_sequences, prompt_lengths=[5])

    with pytest.raises(ValueError, match="response_lengths has 0 entries"):
        compute_prompt_reuse_stats(full_sequences, prompt_lengths=[1], response_lengths=[])
