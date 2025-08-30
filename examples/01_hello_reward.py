#!/usr/bin/env python3
"""
01: Hello Reward - The Foundation of RL Training

GOAL: Train better language models using reinforcement learning

This is step 1: creating reward functions that tell your model what "good" text 
looks like. Without rewards, RL can't train - this is the foundation!

WHY THIS MATTERS: Your model will optimize for these rewards during training.
Good rewards = better model behavior.
"""

def simple_length_reward(prompt: str, completion: str, example: dict) -> float:
    """
    Rate completions based on reasonable length (not too short, not too long).
    
    This is textpolicy's fundamental pattern: pure functions that score text.
    """
    words = len(completion.split())
    
    # Reward reasonable length (10-50 words)
    if 10 <= words <= 50:
        return 1.0  # Perfect score
    elif words < 10:
        return words / 10.0  # Too short penalty  
    else:
        return max(0.0, 1.0 - (words - 50) / 50.0)  # Too long penalty


if __name__ == "__main__":
    # Test our reward function
    test_cases = [
        ("What is AI?", "AI is artificial intelligence."),  # Too short
        ("Explain AI", "AI is artificial intelligence that can learn from data and make decisions."),  # Good
        ("Tell me about AI", "AI is a very long explanation that goes on and on with lots of details about machine learning, deep learning, neural networks, and many other complex topics that make this response way too verbose."),  # Too long
    ]
    
    print("Hello Reward Example")
    print("=" * 30)
    
    for i, (prompt, completion) in enumerate(test_cases, 1):
        reward = simple_length_reward(prompt, completion, {})
        words = len(completion.split())
        print(f"Test {i}: {words} words → Reward: {reward:.2f}")
        print(f"  Text: \"{completion[:50]}{'...' if len(completion) > 50 else ''}\"")
        print()
    
    print("SUCCESS: You've learned textpolicy's core concept: text -> reward score!")
    print("NEXT: Learn how to make rewards part of a training system...")
    