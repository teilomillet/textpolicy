# textpolicy/generation/reload.py
"""
Pure functions for handling LoRA model reloading after training updates.

Critical for RL training: After each training step, the LoRA adapters are updated.
For the next rollout generation, we need to ensure the policy uses the updated model.

Following our design principles:
- Pure functions only
- No state management 
- Zero abstraction cost
"""

from typing import Dict, Tuple, Any, Optional, Callable
import mlx.core as mx # type: ignore
import mlx.nn as nn # type: ignore
from .mlx_generation import load_model# type: ignore


def save_adapters(
    model: nn.Module,
    adapter_path: str
) -> None:
    """
    Pure function to save LoRA adapter weights.
    
    After training updates, save only the LoRA parameters to disk.
    This is much faster than saving the full model.
    
    Args:
        model: Model with trained LoRA adapters
        adapter_path: Path to save adapters
    """
    # Extract LoRA parameters
    lora_params = {}
    if hasattr(model, "named_parameters"):
        for name, param in model.named_parameters():
            if 'lora_' in name.lower() and getattr(param, "requires_grad", True):
                lora_params[name] = param
    else:
        from mlx.utils import tree_flatten
        for name, param in tree_flatten(model.trainable_parameters()):
            if 'lora' in name.lower() or 'adapter' in name.lower():
                lora_params[name] = param

    if not lora_params:
        print("Warning: no LoRA adapter parameters found — skipping save.")
        return

    # Save using MLX
    mx.save_safetensors(adapter_path, lora_params)
    print(f"✓ Saved LoRA adapters to {adapter_path}")


def reload_model(
    base_model_path: str,
    adapter_path: str,
    tokenizer_config: Optional[Dict] = None
) -> Tuple[nn.Module, Any]:
    """
    Pure function to reload model with updated LoRA adapters.
    
    This is called after training to get an updated model for the next rollout.
    Much more efficient than reloading the full model.
    
    Args:
        base_model_path: Path to base model
        adapter_path: Path to updated LoRA adapters
        tokenizer_config: Optional tokenizer config
        
    Returns:
        (updated_model, tokenizer): Model with updated adapters
    """
    # Load fresh model with updated adapters
    model, tokenizer = load_model(
        model_path=base_model_path,
        adapter_path=adapter_path,
        tokenizer_config=tokenizer_config
    )
    
    print("✓ Reloaded model with updated adapters")
    return model, tokenizer


def create_reloadable_policy(
    base_model_path: str,
    initial_adapter_path: Optional[str],
    generation_params: Dict[str, Any]
) -> Tuple[Callable, Callable]:
    """
    Create a policy function that can be reloaded with updated LoRA adapters.
    
    Returns both the policy function and a reload function for efficiency.
    
    Args:
        base_model_path: Path to base model
        initial_adapter_path: Initial LoRA adapter path (can be None)
        generation_params: Generation parameters
        
    Returns:
        (policy_fn, reload_fn): Policy function and reload function
    """
    # Load initial model
    current_model, tokenizer = load_model(
        model_path=base_model_path,
        adapter_path=initial_adapter_path
    )
    
    # Store current state
    current_state = {
        'model': current_model,
        'tokenizer': tokenizer,
        'base_path': base_model_path,
        'generation_params': generation_params
    }
    
    def policy_fn(obs: mx.array, deterministic: bool = False) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Policy function that always uses the current model."""
        from .mlx_generation import generate_tokens
        
        # Use current model (updated after each reload)
        model = current_state['model']
        tokenizer = current_state['tokenizer']
        
        # Adjust temperature
        temp = 0.0 if deterministic else generation_params.get("temperature", 0.8)
        
        # Generate tokens
        response_tokens, response_info = generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=obs,
            max_tokens=generation_params.get("max_tokens", 50),
            temperature=temp,
            top_p=generation_params.get("top_p", 0.95)
        )
        
        # Extract logprobs from response info
        logprobs = response_info.get('logprob', mx.array([]))
        
        extras = {
            "logprob": logprobs,
            "entropy": mx.mean(logprobs) if len(logprobs) > 0 else mx.array(0.0)
        }
        
        return response_tokens, extras
    
    def reload_fn(adapter_path: str) -> None:
        """Reload the model with updated adapters."""
        updated_model, updated_tokenizer = reload_model(
            base_model_path=current_state['base_path'],
            adapter_path=adapter_path
        )
        
        # Update current state
        current_state['model'] = updated_model
        current_state['tokenizer'] = updated_tokenizer
        
        print(f"✓ Policy reloaded with adapters from {adapter_path}")
    
    return policy_fn, reload_fn


def create_training_loop_with_reload(
    base_model_path: str,
    adapter_save_path: str,
    generation_params: Dict[str, Any],
    trainer
) -> Tuple[Callable, Callable]:
    """
    Create a complete training loop that handles LoRA reloading.
    
    This solves the LoRA update problem by automatically saving and reloading
    adapters after each training step.
    
    Args:
        base_model_path: Path to base model
        adapter_save_path: Path to save LoRA adapters
        generation_params: Generation parameters
        trainer: GRPO trainer instance
        
    Returns:
        (train_and_reload_fn, policy_fn): Training function and policy
    """
    # Create reloadable policy
    policy_fn, reload_fn = create_reloadable_policy(
        base_model_path=base_model_path,
        initial_adapter_path=None,  # Start with base model
        generation_params=generation_params
    )
    
    def train_and_reload_fn(rollout_data) -> Dict[str, float]:
        """
        Train the model and reload policy with updated adapters.
        
        Args:
            rollout_data: Rollout data for training
            
        Returns:
            Training metrics
        """
        # 1. Train the model (updates LoRA adapters in-place)
        metrics = trainer.train(rollout_data)
        
        # 2. Save updated LoRA adapters
        save_adapters(trainer.model, adapter_save_path)
        
        # 3. Reload policy with updated adapters
        reload_fn(adapter_save_path)
        
        print("✓ Training step complete, policy updated")
        return metrics
    
    return train_and_reload_fn, policy_fn


# Simple wrapper for the most common use case
def create_auto_reload_setup(
    model_path: str,
    adapter_save_path: str = "./lora_adapters.safetensors",
    **generation_params
) -> Tuple[Callable, Callable, nn.Module, Any]:
    """
    Create complete auto-reloading setup for LoRA training.
    
    This is the main function most users should call.
    
    Args:
        model_path: Path to base model
        adapter_save_path: Where to save LoRA adapters
        **generation_params: Generation parameters
        
    Returns:
        (policy_fn, reload_fn, model, tokenizer): Complete setup
    """
    from .lora import create_lora_setup
    
    # Load and setup LoRA model
    base_model, tokenizer = load_model(model_path)
    
    lora_config = {
        "lora_layers": generation_params.pop("lora_layers", 8),
        "lora_rank": generation_params.pop("lora_rank", 8),
        "lora_scale": generation_params.pop("lora_scale", 20.0)
    }
    
    lora_model, memory_stats = create_lora_setup(
        model=base_model,
        lora_config=lora_config
    )
    
    # Create reloadable policy
    policy_fn, reload_fn = create_reloadable_policy(
        base_model_path=model_path,
        initial_adapter_path=None,
        generation_params=generation_params
    )
    
    print("✓ Auto-reload setup complete")
    print(f"   Memory savings: {memory_stats['memory_savings_percent']:.1f}%")
    print(f"   Adapter save path: {adapter_save_path}")
    
    return policy_fn, reload_fn, lora_model, tokenizer
