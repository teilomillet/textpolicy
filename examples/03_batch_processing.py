#!/usr/bin/env python3
"""
03: Batch Processing - Training Performance at Scale

GOAL: Process thousands of texts efficiently during model training

Step 3: MLX-optimized batch processing evaluates multiple texts simultaneously.
During RL training, models generate hundreds of completions per batch - 
processing them one-by-one would be impossibly slow.

WHY THIS MATTERS: Training requires evaluating ~1000s of texts per training step.
Batch processing with MLX makes this feasible on Apple Silicon.
"""

import mlx.core as mx
from textpolicy.rewards import reward, create_mlx_optimized_batch_processor, RewardConfig

@reward
def clarity_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """Rate text clarity based on sentence structure and word choice."""
    sentences = completion.split('.')
    avg_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len(sentences))
    
    # Optimal sentence length: 10-20 words
    if 10 <= avg_length <= 20:
        return 1.0
    elif avg_length < 10:
        return avg_length / 10.0
    else:
        return max(0.0, 1.0 - (avg_length - 20) / 20.0)


if __name__ == "__main__":
    print("Batch Processing Example")
    print("=" * 30)
    
    # Create batch processor - textpolicy's MLX optimization
    configs = [
        RewardConfig(name="clarity_reward", weight=0.6),
        RewardConfig(name="length_reward", weight=0.4, params={'target_length': 15})
    ]
    
    # This function processes multiple texts simultaneously with MLX
    batch_processor = create_mlx_optimized_batch_processor(configs)
    
    # Test data - multiple prompts and completions
    prompts = [
        "Explain photosynthesis",
        "What is gravity?", 
        "How do computers work?"
    ]
    
    completions = [
        "Plants use sunlight to make food through photosynthesis.",
        "Gravity is the force that attracts objects toward each other.",
        "Computers process information using electronic circuits and software programs."
    ]
    
    examples = [{}, {}, {}]  # Empty example data for this demo
    
    # Process all at once - this is what happens during training!
    print("Processing 3 texts simultaneously with MLX...")
    rewards = batch_processor(prompts, completions, examples)
    
    print(f"Batch rewards (MLX array): {rewards}")
    print(f"Individual scores: {[float(r) for r in rewards]}")
    print()
    
    # Show training-scale processing - process more texts
    print("Scaling to 10 texts (typical training batch)...")
    big_prompts = prompts * 3 + ["What is AI?"]
    big_completions = completions * 3 + ["AI helps solve complex problems."]
    big_examples = [{} for _ in big_prompts]
    
    big_rewards = batch_processor(big_prompts, big_completions, big_examples)
    print(f"Processed {len(big_rewards)} texts: avg={float(mx.mean(big_rewards)):.3f}")
    print()
    
    print("SUCCESS: MLX batch processing enables efficient RL training!")
    print("NEXT: Learn how to combine multiple rewards for sophisticated training...")