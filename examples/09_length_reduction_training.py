#!/usr/bin/env python3
"""
09: Length Reduction Training - Test if GRPO Actually Learns

This example tests whether the model actually learns from RL training by:
1. Starting with long max_tokens (128) to allow verbose responses
2. Rewarding shorter, more concise answers
3. Tracking response length over time to validate learning

Expected behavior: Response lengths should decrease over training steps.
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


def length_reduction_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """
    SCALING reward function that gets better as we approach the target.
    
    This provides a smooth percentage-based reward signal:
    - Scales from 0.2 (very long) to 1.0 (perfect target)
    - Smooth gradient based on distance from target
    - Bonus for being under target length
    """
    if not completion.strip():
        return 0.2
    
    words = completion.split()
    word_count = len(words)
    
    # SCALING REWARD BASED ON DISTANCE TO TARGET
    target_length = 20  # Our goal
    max_penalty_length = 120  # Worst case (original max_tokens limit)
    
    if word_count <= target_length:
        # BONUS for being at or under target
        # Scale from 1.0 (exactly target) to 1.2 (very short)
        if word_count <= 5:  # Too short
            reward = 0.8  # Slight penalty for being too terse
        else:
            # Reward gets better as we approach target from below
            distance_from_target = target_length - word_count
            bonus = min(0.2, distance_from_target * 0.02)  # Up to +20% bonus
            reward = 1.0 + bonus
    else:
        # PENALTY for being over target - scales smoothly
        excess = word_count - target_length
        max_excess = max_penalty_length - target_length  # 100 words of possible excess
        
        # Percentage of how far we are from target to worst case
        excess_percentage = min(1.0, excess / max_excess)
        
        # Scale reward from 1.0 (at target) down to 0.2 (at max penalty)
        reward = 1.0 - (excess_percentage * 0.8)  # Smooth scaling
    
    # Ensure reward stays in reasonable bounds
    reward = max(0.2, min(1.2, reward))
    
    # Log for analysis
    print(f"REWARD: {word_count:2d} words -> {reward:.3f} (target: {target_length}, dist: {abs(word_count - target_length)})")
    
    return reward


def run_length_training():
    """Main training function to test length reduction learning."""
    print("Starting Length Reduction Training with Qwen3-0.6B...")
    print("Goal: Train model to generate shorter, more concise responses")
    print("Starting max_tokens: 128 -> Target: ~12 words\n")
    
    # Setup
    wandb.init(project="textpolicy-length", name="qwen3-length-reduction")
    
    # Load model and create policy with LONG max_tokens
    model, tokenizer = load_model("Qwen/Qwen3-0.6B")
    
    # Create policy function with long generation to test learning
    policy_fn = create_policy(
        model=model,
        tokenizer=tokenizer,
        generation_params={
            'max_tokens': 128,           # LONG - gives room for verbose responses
            'temperature': 0.8,          # Higher temperature for more variation
            'top_p': 0.95,              # Allow diverse vocabulary
            'repetition_penalty': 1.1   # Prevent loops
        }
    )
    
    # Create environment with clear, simple prompts
    def create_env():
        from textpolicy.environment.text_generation import TextGenerationEnv
        return TextGenerationEnv(
            prompts=[
                "What is AI?",
                "Define machine learning",
                "Explain neural networks", 
                "What is deep learning?",
                "How do computers learn?",
                "What is supervised learning?",
                "Define artificial intelligence",
                "Explain data science"
            ],
            reward_fn=length_reduction_reward,
            max_tokens=128,  # Match policy max_tokens
            tokenizer=tokenizer
        )
    
    # Create rollout coordinator
    rollout_coordinator = RolloutCoordinator(
        env_fn=create_env,
        policy_fn=lambda: policy_fn,
        algorithm='grpo',
        num_workers=0,
        max_steps=100,
        max_episodes=20
    )
    
    # Create buffer for episode storage
    buffer = Buffer(max_episodes=20)
    
    # Create trainer with Dr. GRPO algorithm (bias-corrected GRPO)
    trainer = Trainer(
        model=model,
        advantage_fn=grpo.compute_advantages_dr_grpo,  # Dr. GRPO: bias-corrected advantages
        loss_fn=grpo.policy_loss,                      # Same policy loss (no bias here)
        optimizer=optim.Adam(learning_rate=1e-5),      # Higher learning rate for better signal
        max_grad_norm=0.5,                             # Gradient clipping
        buffer=buffer,
        data_selector_fn=grpo.select_recent_data,      # Use recent episodes
        compile_training=True                          # MLX compilation for efficiency
    )
    
    # Track learning metrics over time
    episode_rewards_history = []  # Track average reward per episode batch
    episode_lengths_history = []  # Track average length per episode batch
    
    # Training loop with detailed episode tracking
    for step in range(100):  # More steps to see learning
        print(f"\n--- Training Step {step} ---")
        
        # Collect rollout data
        rollout_buffer = rollout_coordinator.collect()
        
        # Track episode-level metrics for this step
        step_rewards = []
        step_lengths = []
        
        for episode in rollout_buffer.episodes:
            buffer.add_episode_from_dict(episode.to_dict())
            
            # Extract reward and estimate length from episode
            if hasattr(episode, 'rew') and len(episode.rew) > 0:
                episode_reward = float(episode.rew[0])
                step_rewards.append(episode_reward)
                
                # Estimate word count from reward (reverse engineering our reward function)
                # This is approximate but gives us a length trend
                if episode_reward >= 1.0:
                    estimated_length = 20  # At or below target
                elif episode_reward >= 0.2:
                    # Reverse the scaling: reward = 1.0 - (excess_percentage * 0.8)
                    excess_percentage = (1.0 - episode_reward) / 0.8
                    excess_words = excess_percentage * 100  # max_excess = 100
                    estimated_length = 20 + excess_words
                else:
                    estimated_length = 120  # Max length
                
                step_lengths.append(estimated_length)
        
        # Calculate episode batch statistics
        batch_avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
        batch_avg_length = sum(step_lengths) / len(step_lengths) if step_lengths else 120
        
        episode_rewards_history.append(batch_avg_reward)
        episode_lengths_history.append(batch_avg_length)
        
        # Train on collected data
        metrics = trainer.train()
        
        # Log metrics including episode-level trends
        wandb.log({
            "step": step,
            "loss": metrics['loss'],
            "batch_avg_reward": batch_avg_reward,
            "batch_avg_length": batch_avg_length,
            "episodes_collected": len(rollout_buffer.episodes),
            "total_episodes": len(buffer.episodes)
        })
        
        # Print episode batch summary for this step
        print(f"Episode Batch Summary:")
        print(f"   Average Reward: {batch_avg_reward:.3f}")
        print(f"   Average Length: {batch_avg_length:.1f} words (target: 20)")
        print(f"   Episodes: {len(step_rewards)}")
        
        # Detailed progress logging every 10 steps
        if step % 10 == 0:
            print(f"\nTraining Progress - Step {step}:")
            print(f"   Loss: {metrics['loss']:.4f}")
            print(f"   Buffer: {len(buffer.episodes)} total episodes")
            
            # Show example generation to track length
            env = create_env()
            obs, info = env.reset()
            action, action_info = policy_fn(obs, deterministic=True)
            
            # Decode action back to text for analysis
            from textpolicy.generation.mlx_generation import decode
            action_text = decode(tokenizer, action)
            prompt_text = info.get('prompt_text', str(obs))
            word_count = len(action_text.split())
            
            print(f"Example Generation:")
            print(f"   Prompt: '{prompt_text}'")
            print(f"   Response ({word_count} words): '{action_text[:100]}{'...' if len(action_text) > 100 else ''}'")
            
            # Track learning trends using episode-level metrics
            if len(episode_rewards_history) >= 10:
                # Compare recent episodes vs early episodes
                recent_rewards = episode_rewards_history[-5:]  # Last 5 episode batches
                early_rewards = episode_rewards_history[2:7] if len(episode_rewards_history) >= 7 else episode_rewards_history[:3]
                
                recent_lengths = episode_lengths_history[-5:]  # Last 5 episode batches  
                early_lengths = episode_lengths_history[2:7] if len(episode_lengths_history) >= 7 else episode_lengths_history[:3]
                
                if recent_rewards and early_rewards:
                    reward_improvement = sum(recent_rewards) / len(recent_rewards) - sum(early_rewards) / len(early_rewards)
                    length_improvement = sum(early_lengths) / len(early_lengths) - sum(recent_lengths) / len(recent_lengths)
                    
                    print(f"\nLEARNING TRENDS:")
                    print(f"   Reward change: {reward_improvement:+.3f}")
                    print(f"   Length reduction: {length_improvement:+.1f} words")
                    
                    if reward_improvement > 0.05 and length_improvement > 0:
                        print("SUCCESS: Model is learning to reduce length!")
                    elif reward_improvement > 0.02:
                        print("PROGRESS: Positive learning trend detected")
                    elif reward_improvement < -0.02:
                        print("CONCERN: Negative trend - check hyperparameters")
                    else:
                        print("STABLE: Learning in progress...")
    
    print("\nLength Reduction Training Complete!")
    
    # Final analysis
    if len(step_rewards) >= 20:
        early_avg = sum(step_rewards[:10]) / 10
        late_avg = sum(step_rewards[-10:]) / 10
        total_improvement = late_avg - early_avg
        
        print(f"\nLEARNING ANALYSIS:")
        print(f"   Early average reward: {early_avg:.3f}")
        print(f"   Late average reward:  {late_avg:.3f}")
        print(f"   Total improvement:    {total_improvement:+.3f}")
        
        if total_improvement > 0.1:
            print("SUCCESS: Model learned to generate shorter responses!")
        elif total_improvement > 0.05:
            print("PROGRESS: Model shows learning trend")
        else:
            print("LIMITED: Weak learning signal - may need tuning")
    
    wandb.finish()


if __name__ == "__main__":
    run_length_training()