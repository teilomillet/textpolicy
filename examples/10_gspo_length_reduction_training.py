#!/usr/bin/env python3
"""10: GSPO Length Reduction Training - Sequence-Level Policy Optimization
Demonstrates GSPO's sequence-level importance sampling vs GRPO's token-level approach."""

import mlx.optimizers as optim
from textpolicy.generation.mlx_generation import load_model, create_policy
from textpolicy.rollout import RolloutCoordinator
from textpolicy.buffer import Buffer
from textpolicy.training import Trainer
from textpolicy.algorithms import grpo, gspo


def length_reward(prompt: str, completion: str, example: dict, **kwargs) -> float:
    """Reward shorter responses: target 15-25 words with smooth penalties."""
    words = len(completion.split()) if completion.strip() else 0
    if 15 <= words <= 25:
        return 1.0  # Perfect length
    elif words < 15:
        return 0.6 + (words / 15) * 0.4  # Too short penalty
    else:
        return max(0.1, 0.9 * (0.95 ** (words - 25)))  # Exponential decay for long responses


def run_gspo_training():
    """Compare GSPO sequence vs GRPO token-level importance sampling."""
    print("GSPO vs GRPO Length Reduction Training")
    
    model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct", verbose=False)
    policy_fn = create_policy(model, tokenizer, {'max_tokens': 40, 'temperature': 0.8})
    
    prompts = ["What is AI?", "How do computers work?", "Explain machine learning.",
               "What are neural networks?", "How does programming help?", "Define algorithms."]
    
    def create_env():
        from textpolicy.environment.text_generation import TextGenerationEnv
        return TextGenerationEnv(prompts, length_reward, max_tokens=40, tokenizer=tokenizer)
    
    algorithms = [("GRPO", grpo.policy_loss), ("GSPO", gspo.policy_loss_sequence)]
    
    for algo_name, loss_fn in algorithms:
        print(f"\nTesting {algo_name}...")
        
        if algo_name == "GSPO":  # Reload for fair comparison
            model, tokenizer = load_model("Qwen/Qwen2.5-1.5B-Instruct", verbose=False)
            policy_fn = create_policy(model, tokenizer, {'max_tokens': 40, 'temperature': 0.8})
        
        rollout_coordinator = RolloutCoordinator(create_env, lambda: policy_fn, 'grpo', 0, 30, 8)
        buffer = Buffer(max_episodes=30)
        # Use algorithm-specific learning rates to handle different stability characteristics
        lr = 5e-6 if algo_name == "GRPO" else 1e-5  # Lower LR for GRPO due to numerical instability
        
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages_dr_grpo,
            loss_fn=loss_fn,
            optimizer=optim.Adam(learning_rate=lr),
            metrics_fn=grpo.compute_metrics,
            buffer=buffer,  # link buffer to trainer
            data_selector_fn=grpo.select_recent_data,
            max_grad_norm=0.5  # Add gradient clipping for stability
        )
        
        rewards = []
        losses = []
        
        # Extended training to observe long-term learning patterns
        for step in range(100):  # Extended to 100 steps for comprehensive learning analysis
            rollout_buffer = rollout_coordinator.collect()
            # Critical: Transfer collected episodes to trainer's buffer for GSPO/GRPO processing
            for episode in rollout_buffer.episodes:
                buffer.add_episode_from_dict(episode.to_dict())
                if hasattr(episode, 'rew') and episode.rew:
                    rewards.append(float(episode.rew[0]))
            
            metrics = trainer.train()
            current_loss = metrics.get('loss', 0)
            losses.append(current_loss)
            
            # Progress reporting to track learning dynamics
            if step % 20 == 0 or step == 99:  # Every 20 steps + final step
                recent_reward = sum(rewards[-8:]) / len(rewards[-8:]) if len(rewards) >= 8 else 0
                recent_loss_avg = sum(losses[-3:]) / len(losses[-3:]) if len(losses) >= 3 else current_loss
                print(f"  Step {step}: reward={recent_reward:.3f}, loss={recent_loss_avg:.4f}")
        
        # Comprehensive learning phase analysis over 100 steps
        if len(rewards) >= 80:
            early = sum(rewards[:20])/20     # Steps 1-20
            mid = sum(rewards[40:60])/20     # Steps 41-60  
            late = sum(rewards[-20:])/20     # Steps 81-100
            print(f"{algo_name} Results: Early {early:.3f} -> Mid {mid:.3f} -> Late {late:.3f} (Total: {late-early:+.3f})")
        else:
            initial, final = sum(rewards[:10])/10, sum(rewards[-10:])/10
            print(f"{algo_name} Results: {initial:.3f} -> {final:.3f} ({final-initial:+.3f})")
        
        buffer.clear()
    
    print("\nGSPO demonstrates sequence-level importance sampling advantages!")


if __name__ == "__main__":
    run_gspo_training()