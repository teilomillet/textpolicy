#!/usr/bin/env python3
"""
05: TextPolicy Essence - Ready for Production Training

GOAL: Complete RL training system for language models

This is textpolicy's complete essence: decorator-registered pure functions + 
MLX batch optimization = production-ready text quality evaluation for 
reinforcement learning training.

You now understand the complete system needed to train better language models.
"""

import mlx.core as mx
from textpolicy.rewards import reward, create_mlx_optimized_batch_processor, RewardConfig

@reward  # The decorator: registers this function automatically in textpolicy's system
def quality_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """The essence: pure function that measures text quality."""
    # Length appropriateness (not too short/long) - basic quality filter
    words = len(completion.split())
    length_score = 1.0 if 10 <= words <= 40 else max(0.1, min(words/10, 40/words))
    
    # Relevance to prompt (keyword overlap) - ensures on-topic responses  
    prompt_words = set(prompt.lower().split())
    completion_words = set(completion.lower().split())
    relevance_score = len(prompt_words & completion_words) / max(1, len(prompt_words))
    
    # Coherence (no repetition) - penalizes repetitive text
    unique_words = len(set(completion.split()))
    total_words = len(completion.split())
    coherence_score = unique_words / max(1, total_words)
    
    return (length_score + relevance_score + coherence_score) / 3  # Balanced composite score


if __name__ == "__main__":
    print("TextPolicy Essence - Complete Training System")
    print("=" * 45)
    
    # Step 1: Create MLX-optimized evaluator (zero-cost abstraction)
    evaluator = create_mlx_optimized_batch_processor([
        RewardConfig(name="quality_reward", weight=1.0)
    ])
    
    # Step 2: Define training scenarios (what models will learn to optimize)
    scenarios = [
        ("What is AI?", "AI is artificial intelligence that learns from data."),
        ("Explain gravity", "Gravity attracts objects toward each other with force."),
        ("How do plants grow?", "Plants grow using sunlight, water and nutrients from soil.")
    ]
    
    prompts, completions = zip(*scenarios)
    examples = [{} for _ in scenarios]
    
    # Step 3: Evaluate with Apple Silicon acceleration (training-ready)
    scores = evaluator(list(prompts), list(completions), list(examples))
    
    # Step 4: Training-ready results
    print("Training-Ready Quality Evaluation:")
    for i, ((prompt, completion), score) in enumerate(zip(scenarios, scores)):
        print(f"  {i+1}. {prompt}")
        print(f"     -> \"{completion}\"")
        print(f"     Quality: {float(score):.3f}")
    
    print(f"\nAverage Quality: {float(mx.mean(scores)):.3f}")
    print(f"MLX Acceleration: {'Active' if mx.default_device().type == 'gpu' else 'CPU mode'}")
    
    print("\n" + "="*60)
    print("COMPLETE: TextPolicy system ready for RL training!")
    print("Components mastered:")
    print("  - Pure functions (@reward decorator)")  
    print("  - MLX Apple Silicon optimization")
    print("  - Zero abstraction cost design")
    print("  - Production-ready for reinforcement learning")
    print("="*60)