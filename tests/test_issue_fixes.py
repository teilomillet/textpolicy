"""
Tests for GitHub Issues #2 and #3 fixes.

Issue #2: TextGenerationEnv cannot pass example metadata to reward functions
Issue #3: Export @verifier decorator at top level for API consistency
"""

import pytest


class TestIssue3VerifierExport:
    """Test that @verifier decorator is exported at top level (Issue #3)."""

    def test_verifier_accessible_at_top_level(self):
        """tp.verifier should be accessible without deep import."""
        import textpolicy as tp

        assert hasattr(tp, "verifier"), "verifier should be exported at top level"
        assert callable(tp.verifier), "verifier should be callable"

    def test_verifier_same_as_deep_import(self):
        """tp.verifier should be the same function as deep import."""
        import textpolicy as tp
        from textpolicy.rewards import verifier

        assert tp.verifier is verifier, "tp.verifier should be same as textpolicy.rewards.verifier"

    def test_verifier_decorator_works(self):
        """@tp.verifier should work as a decorator."""
        import textpolicy as tp

        @tp.verifier
        def my_test_verifier(prompt, completion, example, **kwargs):
            return len(completion) > 0

        # Verify it's registered and callable
        assert callable(my_test_verifier)
        result = my_test_verifier("test prompt", "test completion", {})
        assert isinstance(result, bool)

    def test_reward_and_verifier_both_at_top_level(self):
        """Both @reward and @verifier should be at top level for API consistency."""
        import textpolicy as tp

        assert hasattr(tp, "reward"), "reward should be at top level"
        assert hasattr(tp, "verifier"), "verifier should be at top level"


@pytest.mark.integration
class TestIssue2ExamplesParameter:
    """Test that TextGenerationEnv passes example metadata to reward functions (Issue #2)."""

    @pytest.fixture
    def dummy_tokenizer(self):
        """Provide a minimal tokenizer for tests."""
        class DummyTokenizer:
            def encode(self, text):
                return [ord(c) % 256 for c in text]

            def decode(self, ids):
                return "".join(chr(int(i) % 256) for i in ids)

        return DummyTokenizer()

    def test_env_accepts_examples_parameter(self, dummy_tokenizer):
        """TextGenerationEnv should accept an examples parameter."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        prompts = ["Hello", "World"]
        examples = [{"key": "value1"}, {"key": "value2"}]

        def reward_fn(prompt, completion, example, **kwargs):
            return 1.0

        # Should not raise
        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=reward_fn,
            tokenizer=dummy_tokenizer,
            examples=examples,
        )
        assert env.examples == examples

    def test_env_defaults_to_empty_dicts_when_no_examples(self, dummy_tokenizer):
        """When examples not provided, should default to empty dicts."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        prompts = ["Hello", "World"]

        def reward_fn(prompt, completion, example, **kwargs):
            return 1.0

        env = TextGenerationEnv(
            prompts=prompts, reward_fn=reward_fn, tokenizer=dummy_tokenizer
        )

        assert env.examples == [{}, {}]

    def test_env_validates_examples_length(self, dummy_tokenizer):
        """Should raise ValueError if examples length != prompts length."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        prompts = ["Hello", "World"]
        examples = [{"key": "value1"}]  # Wrong length

        def reward_fn(prompt, completion, example, **kwargs):
            return 1.0

        with pytest.raises(ValueError, match="examples length.*must match prompts length"):
            TextGenerationEnv(
                prompts=prompts,
                reward_fn=reward_fn,
                tokenizer=dummy_tokenizer,
                examples=examples,
            )

    def test_example_passed_to_reward_function(self, dummy_tokenizer):
        """Reward function should receive the correct example for each prompt."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        prompts = ["Question 1", "Question 2"]
        examples = [
            {"db_id": "database_1", "gold_sql": "SELECT 1"},
            {"db_id": "database_2", "gold_sql": "SELECT 2"},
        ]

        received_examples = []

        def capture_reward(prompt, completion, example, **kwargs):
            received_examples.append(example.copy())
            return 1.0

        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=capture_reward,
            tokenizer=dummy_tokenizer,
            examples=examples,
        )

        # Episode 0 should use examples[0]
        env.reset()
        env.step("some response")
        assert received_examples[0] == {"db_id": "database_1", "gold_sql": "SELECT 1"}

        # Episode 1 should use examples[1]
        env.reset()
        env.step("another response")
        assert received_examples[1] == {"db_id": "database_2", "gold_sql": "SELECT 2"}

    def test_reward_dict_can_emit_explicit_is_correct(self, dummy_tokenizer):
        """TextGenerationEnv should pass verifier correctness in info when provided."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        def reward_fn(prompt, completion, example, **kwargs):
            return {"reward": -0.5, "is_correct": True}

        env = TextGenerationEnv(
            prompts=["P1"],
            reward_fn=reward_fn,
            tokenizer=dummy_tokenizer,
            examples=[{}],
        )
        env.reset()
        result = env.step("response")

        assert result["reward"] == -0.5
        assert result["info"]["is_correct"] is True

    def test_examples_cycle_with_prompts(self, dummy_tokenizer):
        """Examples should cycle correctly when prompts cycle."""
        from textpolicy.environment.text_generation import TextGenerationEnv

        prompts = ["P1", "P2"]
        examples = [{"idx": 0}, {"idx": 1}]

        received_indices = []

        def capture_reward(prompt, completion, example, **kwargs):
            received_indices.append(example.get("idx"))
            return 1.0

        env = TextGenerationEnv(
            prompts=prompts,
            reward_fn=capture_reward,
            tokenizer=dummy_tokenizer,
            examples=examples,
        )

        # Run 4 episodes (should cycle through prompts twice)
        for _ in range(4):
            env.reset()
            env.step("response")

        # Should have received [0, 1, 0, 1]
        assert received_indices == [0, 1, 0, 1]

    def test_litmus_test_from_issue(self, dummy_tokenizer):
        """Run the exact litmus test from Issue #2."""
        from textpolicy.environment.text_generation import TextGenerationEnv
        import textpolicy as tp

        examples = [
            {"db_id": "concert_singer", "gold_sql": "SELECT COUNT(*) FROM singer"},
            {"db_id": "pets_1", "gold_sql": "SELECT COUNT(*) FROM pets"},
        ]
        prompts = [
            "Schema: singer(id, name)\nQuestion: How many singers?",
            "Schema: pets(id, name)\nQuestion: How many pets?",
        ]

        captured_db_ids = []

        @tp.reward
        def check_example(prompt, completion, example, **kwargs):
            db_id = example.get("db_id")
            captured_db_ids.append(db_id)
            return 1.0

        env = TextGenerationEnv(prompts, check_example, examples=examples, tokenizer=dummy_tokenizer)

        # First episode
        env.reset()
        env.step("some action")

        # Should have captured 'concert_singer'
        assert captured_db_ids[0] == "concert_singer", f"Expected 'concert_singer', got {captured_db_ids[0]}"

        # Second episode
        env.reset()
        env.step("another action")

        # Should have captured 'pets_1'
        assert captured_db_ids[1] == "pets_1", f"Expected 'pets_1', got {captured_db_ids[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
