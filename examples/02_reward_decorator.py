#!/usr/bin/env python3
"""
02: Reward Decorator - Building a Training System

GOAL: Make rewards modular for scalable RL training

Step 2: The @reward decorator registers your functions automatically, making 
them discoverable by the training system. This modularity is essential when 
training models - you need many different reward functions working together.

WHY THIS MATTERS: During training, you'll combine multiple rewards (length, 
quality, safety, etc.). The decorator system makes this seamless.
"""

from textpolicy.rewards import reward

@reward  # This decorator registers our function automatically!
def length_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """Rate text based on reasonable length - now registered in textpolicy!"""
    words = len(completion.split())
    target = kwargs.get('target_length', 30)
    tolerance = kwargs.get('tolerance', 0.3)
    
    deviation = abs(words - target) / target
    if deviation <= tolerance:
        return 1.0
    else:
        return max(0.0, 1.0 - deviation)

@reward(name="politeness")  # Custom name for the reward
def politeness_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """Rate text based on politeness markers."""
    polite_words = ['please', 'thank you', 'appreciate', 'kindly']
    text_lower = completion.lower()
    
    polite_count = sum(1 for word in polite_words if word in text_lower)
    return min(1.0, polite_count * 0.5)  # Max reward with 2+ polite words


if __name__ == "__main__":
    # Import functions to see the registry system
    from textpolicy.rewards import list_registered_functions, get_reward_function
    
    print("Reward Decorator Example")
    print("=" * 35)
    
    # Show registered functions
    registered = list_registered_functions()
    print(f"Registered rewards: {registered['rewards']}")
    print()
    
    # Test using registered functions - this is how training systems discover rewards
    test_text = "Thank you for your help! I really appreciate your kindness."
    
    # Get functions from registry (training systems do this automatically)
    length_func = get_reward_function('length_reward')
    politeness_func = get_reward_function('politeness')
    
    length_score = length_func("", test_text, {}, target_length=12)
    politeness_score = politeness_func("", test_text, {})
    
    print(f"Test text: \"{test_text}\"")
    print(f"Length reward: {length_score:.2f}")
    print(f"Politeness reward: {politeness_score:.2f}")
    print()
    
    print("SUCCESS: You've learned the @reward decorator - the key to modular training!")
    print("NEXT: Learn how to process multiple texts efficiently for training...")
    