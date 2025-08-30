"""
Pytest configuration and shared fixtures for textpolicy tests.
"""

import pytest
import mlx.core as mx
import numpy as np
from typing import List, Dict, Any


@pytest.fixture
def sample_logprobs():
    """Fixture providing sample log probabilities for testing."""
    return {
        'old_logprobs': mx.array([-1.0, -1.2, -0.8, -1.1, -0.9]),
        'new_logprobs': mx.array([-1.1, -1.0, -0.9, -1.0, -1.0]),
    }


@pytest.fixture
def sample_sequence_data():
    """Fixture providing sample sequence data for testing."""
    return {
        'sequence_lengths': [2, 3],
        'advantages': mx.array([0.5, -0.3]),
        'total_tokens': 5
    }


@pytest.fixture
def sample_reward_config():
    """Fixture providing sample reward configuration."""
    return {
        'length_weight': 1.0,
        'target_length': 15,
        'keyword_weight': 0.5,
        'keywords': ['AI', 'machine', 'learning'],
        'perplexity_weight': 0.3,
        'accuracy_weight': 0.2
    }


@pytest.fixture
def sample_text_data():
    """Fixture providing sample text data for testing."""
    return {
        'prompts': [
            "What is AI?",
            "Explain machine learning.",
            "How does artificial intelligence work?"
        ],
        'completions': [
            "AI is artificial intelligence that enables machines to think.",
            "Machine learning allows computers to learn from data automatically.", 
            "Artificial intelligence works by processing data through algorithms."
        ],
        'examples': [
            {"target_length": 15, "keywords": ["AI"], "correct_answer": "intelligence"},
            {"target_length": 20, "keywords": ["learning"], "correct_answer": "data"},
            {"target_length": 18, "keywords": ["intelligence"], "correct_answer": "algorithms"}
        ]
    }


@pytest.fixture
def mlx_test_arrays():
    """Fixture providing various MLX arrays for testing."""
    return {
        'scalar': mx.array(5.0),
        'vector': mx.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'matrix': mx.array([[1.0, 2.0], [3.0, 4.0]]),
        'tensor3d': mx.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        'zeros': mx.zeros((3, 3)),
        'ones': mx.ones((2, 4)),
        'random': mx.random.normal((10,))
    }


@pytest.fixture(scope="session")
def skip_if_no_model():
    """Fixture to skip tests that require a model if none is available."""
    def _skip_if_no_model():
        try:
            # Try to import model loading functionality
            from textpolicy.generation.mlx_generation import load_model
            return False  # Model available
        except ImportError:
            return True  # Model not available
    return _skip_if_no_model


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line(
        "markers", "algorithm: Algorithm-specific tests (GSPO, GRPO, etc.)"
    )
    config.addinivalue_line(
        "markers", "reward: Reward system tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add algorithm marker to algorithm tests
        if "gspo" in item.nodeid.lower() or "grpo" in item.nodeid.lower():
            item.add_marker(pytest.mark.algorithm)
        
        # Add reward marker to reward tests  
        if "reward" in item.nodeid.lower():
            item.add_marker(pytest.mark.reward)
        
        # Add slow marker to tests that might be slow
        if "training" in item.nodeid.lower() or "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)