#!/usr/bin/env python3
"""
04: Multiple Rewards - Real-World Training Quality

GOAL: Create sophisticated reward systems for production RL training

Step 4: Combining multiple reward functions creates nuanced quality scoring.
Real training needs multiple objectives: helpfulness, safety, accuracy, style.
Single rewards are too simplistic for quality model behavior.

WHY THIS MATTERS: Production models optimize for multiple objectives simultaneously.
This step shows how to balance different quality measures during training.
"""

from textpolicy.rewards import (
    reward, verifier, create_mlx_optimized_batch_processor, 
    RewardConfig
)

@reward
def coherence_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """Rate how well the completion answers the prompt."""
    # Simple coherence: check if key words from prompt appear in completion
    prompt_words = set(prompt.lower().split())
    completion_words = set(completion.lower().split())
    
    overlap = len(prompt_words & completion_words)
    return min(1.0, overlap / max(1, len(prompt_words)) * 2)

@verifier
def no_repetition(prompt: str, completion: str, example: dict, **kwargs) -> bool:
    """Verify text doesn't repeat the same phrase multiple times."""
    words = completion.split()
    if len(words) < 4:
        return True
    
    # Check for repeated 3-word phrases
    phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    return len(phrases) == len(set(phrases))  # No duplicates


if __name__ == "__main__":
    print("Multiple Rewards Example")
    print("=" * 32)
    
    # Create sophisticated quality scorer with multiple rewards
    reward_configs = [
        RewardConfig(name="length_reward", weight=0.5, params={'target_length': 20}),
        RewardConfig(name="coherence_reward", weight=0.5),
        # Add verifier that penalizes repetitive text
        RewardConfig(name="coherence_reward", weight=0.0, verifiers=['no_repetition'], verifier_penalty=-0.5)
    ]
    
    # Build the composite quality evaluator
    quality_evaluator = create_mlx_optimized_batch_processor(reward_configs)
    
    # Test different quality texts
    test_cases = [
        {
            "prompt": "What is machine learning?",
            "completion": "Machine learning is AI that learns patterns from data automatically.",
            "label": "Good: coherent, right length, clear"
        },
        {
            "prompt": "Explain clouds",
            "completion": "Clouds are made of water droplets. Clouds are made of water droplets. Clouds are made of water droplets.",
            "label": "Bad: repetitive content"
        },
        {
            "prompt": "What is photosynthesis?", 
            "completion": "Cars drive fast.",
            "label": "Bad: completely off-topic"
        },
        {
            "prompt": "How do birds fly?",
            "completion": "Birds use their wings to generate lift through airflow dynamics, allowing them to soar through the air efficiently.",
            "label": "Good: on-topic and detailed"
        }
    ]
    
    prompts = [case["prompt"] for case in test_cases]
    completions = [case["completion"] for case in test_cases]
    examples = [{} for _ in test_cases]
    
    # Evaluate all texts with our composite quality scorer
    quality_scores = quality_evaluator(prompts, completions, examples)
    
    print("Composite Quality Evaluation Results (Multi-Objective Training):")
    print("=" * 65)
    
    for i, (case, score) in enumerate(zip(test_cases, quality_scores)):
        print(f"\nTest {i+1}: {case['label']}")
        print(f"  Prompt: \"{case['prompt']}\"")
        print(f"  Completion: \"{case['completion'][:60]}{'...' if len(case['completion']) > 60 else ''}\"")
        print(f"  Quality Score: {float(score):.3f}")
    
    print(f"\nScore distribution: {[f'{float(s):.2f}' for s in quality_scores]}")
    print()
    print("SUCCESS: Multi-objective rewards enable sophisticated model training!")
    print("NEXT: See the complete textpolicy system in action...")