#!/usr/bin/env python3
"""
08: Real RL Training - Proper GRPO Implementation

This example demonstrates proper RL training using TextPolicy's GRPO algorithm
with rollout collection, buffer management, and stable training.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from textpolicy.generation.mlx_generation import load_model, create_policy
from textpolicy.rollout import RolloutCoordinator
from textpolicy.buffer import Buffer
from textpolicy.training import Trainer
from textpolicy.algorithms import grpo
import wandb


def stable_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """
    Stable reward function with proper baselines and smoothing.
    
    This reward function is designed for training stability:
    - Uses continuous scoring instead of sharp thresholds
    - Includes baseline rewards to prevent extreme swings
    - Smooths transitions between reward levels
    """
    if not completion.strip():
        return 0.1  # Small positive reward for attempts
    
    words = completion.split()
    word_count = len(words)
    
    # Smooth length reward with sigmoid-like function
    # Target: 15-25 words, smooth falloff outside this range
    target_length = 20
    length_diff = abs(word_count - target_length)
    length_score = max(0.1, 1.0 - (length_diff / 15.0))
    
    # Semantic relevance using word overlap with smoothing
    prompt_words = set(prompt.lower().split())
    completion_words = set(completion.lower().split())
    
    if len(prompt_words) == 0:
        relevance = 0.5  # Neutral relevance if no prompt words
    else:
        overlap = len(prompt_words & completion_words)
        relevance = min(1.0, overlap / len(prompt_words))
        # Smooth relevance to prevent sharp transitions
        relevance = 0.3 + 0.7 * relevance
    
    # Quality indicators (prevent repetitive/meaningless responses)
    quality_penalty = 0.0
    
    # Penalize excessive repetition
    if len(set(words)) < len(words) * 0.7:  # More than 30% repeated words
        quality_penalty += 0.2
    
    # Penalize very short responses
    if word_count < 5:
        quality_penalty += 0.3
    
    # Penalize responses that are just the prompt repeated
    if completion.lower().strip() == prompt.lower().strip():
        quality_penalty += 0.5
    
    # Combine scores with baseline and smoothing
    base_reward = 0.3  # Baseline reward for any reasonable response
    combined_score = (length_score * 0.4 + relevance * 0.4 + base_reward) - quality_penalty
    
    # Clamp to reasonable range and apply smoothing
    final_reward = max(0.1, min(1.0, combined_score))
    
    return final_reward


def run_training():
    """Main training function using proper TextPolicy architecture."""
    print(" Starting proper GRPO training with Qwen3-0.6B...")
    
    # Setup
    wandb.init(project="textpolicy-grpo", name="qwen3-grpo-stable")
    
    # Load model and create policy
    model, tokenizer = load_model("Qwen/Qwen3-0.6B")
    
    # Create policy function with stable generation parameters
    policy_fn = create_policy(
        model=model,
        tokenizer=tokenizer,
        generation_params={
            'max_tokens': 25,
            'temperature': 0.7,        # Stable temperature
            'top_p': 0.9,             # Focused sampling
            'repetition_penalty': 1.1  # Prevent repetition
        }
    )
    
    # Create environment function (simple text generation environment)
    def create_env():
        from textpolicy.environment.text_generation import TextGenerationEnv
        return TextGenerationEnv(
            prompts=[
                "What is AI?",
                "How does machine learning work?", 
                "Explain neural networks",
                "What is deep learning?",
                "What are the benefits of deep learning?",
                "How do neural networks learn?",
                "What is supervised learning?",
                "Explain unsupervised learning"
            ],
            reward_fn=stable_reward,
            max_tokens=25,
            tokenizer=tokenizer  # Pass tokenizer for MLX compatibility
        )
    
    # Create rollout coordinator with GRPO strategy
    rollout_coordinator = RolloutCoordinator(
        env_fn=create_env,
        policy_fn=lambda: policy_fn,
        algorithm='grpo',
        num_workers=0,  # Single process for simplicity
        max_steps=100,
        max_episodes=20
    )
    
    # Create buffer for episode storage
    buffer = Buffer(max_episodes=20)
    
    # Create trainer with GRPO algorithm
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages,      # GRPO advantage computation
        loss_fn=grpo.policy_loss,                  # GRPO policy loss
        optimizer=optim.Adam(learning_rate=5e-6),  # Lower learning rate for stability
        max_grad_norm=0.5,                         # Gradient clipping
        buffer=buffer,
        data_selector_fn=grpo.select_recent_data,  # Use recent episodes
        compile_training=True                       # MLX compilation for efficiency
    )
    
    # Training loop with proper rollout collection
    for step in range(50):
        print(f"\n--- Training Step {step} ---")
        
        # Collect rollout data
        print("Collecting rollout data...")
        rollout_buffer = rollout_coordinator.collect()
        
        # Add to training buffer
        for episode in rollout_buffer.episodes:
            buffer.add_episode_from_dict(episode.to_dict())
        
        # Train on collected data
        print("Training on collected data...")
        metrics = trainer.train()  # Uses buffer with GRPO data selection
        
        # Log metrics
        wandb.log({
            "step": step,
            "loss": metrics['loss'],
            "episodes_collected": len(rollout_buffer.episodes),
            "total_episodes": len(buffer.episodes)
        })
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}: loss={metrics['loss']:.4f}")
            print(f"  Episodes in buffer: {len(buffer.episodes)}")
            
            # Show example generation
            env = create_env()
            obs, info = env.reset()
            action, action_info = policy_fn(obs, deterministic=True)
            # Decode action back to text for display
            from textpolicy.generation.mlx_generation import decode
            action_text = decode(tokenizer, action)
            prompt_text = info.get('prompt_text', str(obs))
            print(f"  Example: '{prompt_text}' -> '{action_text}'")
    
    print("GRPO training complete!")
    wandb.finish()


if __name__ == "__main__":
    run_training()