# textpolicy/rewards/adapters.py
"""
Core configuration models and utility functions for MLX-optimized reward system.

This module provides essential patterns for building modular reward systems:
- Configuration models for reward system setup
- Sample types for data handling
- Math utilities for text processing
- Async patterns for external model integration
"""

import mlx.core as mx
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re

# Optional dependencies
try:
    import aiohttp # type: ignore
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    aiohttp = None

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback simple config class
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(default=None, **kwargs):
        return default


# ==========================================
# CONFIGURATION MODELS
# ==========================================

class MLXRewardConfig(BaseModel):
    """Configuration for individual reward functions."""
    weight: float = Field(1.0, description="Weight of this reward function")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the reward function")
    verifiers: Optional[List[str]] = Field(None, description="List of verifier names")
    verifier_penalty: float = Field(0.0, description="Penalty if verifiers fail")
    enable_mlx_compilation: bool = Field(True, description="Enable MLX compilation")


class MLXRewardSystemConfig(BaseModel):
    """Configuration for complete reward system setup."""
    reward_configs: Dict[str, MLXRewardConfig] = Field(
        default_factory=dict, 
        description="MLX-optimized reward configurations"
    )
    batch_size: int = Field(32, description="Batch size for MLX processing")
    max_workers: int = Field(4, description="Max workers for async processing")
    enable_external_rewards: bool = Field(False, description="Enable external reward models")


# ==========================================
# SAMPLE TYPES
# ==========================================

@dataclass
class MLXSample:
    """Lightweight sample type for text generation data."""
    prompt: str = ""
    response: str = ""
    label: Optional[str] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        FAILED = "failed"
    
    status: Status = Status.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing."""
        return {
            "prompt": self.prompt,
            "completion": self.response,  # Map to our naming convention
            "example": {
                "label": self.label,
                "metadata": self.metadata,
                **self.metadata  # Flatten metadata
            }
        }


# ==========================================
# MATH UTILITIES
# ==========================================

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract LaTeX boxed answer from text."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    
    i = idx
    brace_count = 0
    while i < len(text):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return text[idx:i+1]
        i += 1
    return None


def normalize_math_answer(answer: str) -> str:
    """Normalize mathematical answer for comparison."""
    answer = answer.split("=")[-1].strip()
    
    # Remove common expressions
    removals = ["square", "dollars", "units", "\\text{}", "^\\circ"]
    for removal in removals:
        answer = answer.replace(removal, "")
    
    # Normalize fractions and roots
    answer = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", answer)
    answer = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", answer)
    answer = answer.replace("$", "").strip()
    
    return answer


def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    
    if not truth_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common) / len(truth_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


# ==========================================
# EXTERNAL REWARD MODELS
# ==========================================

class MLXExternalRewardModel:
    """Async client for external reward model APIs."""
    
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout
    
    async def get_reward(self, sample: MLXSample) -> float:
        """Get reward from external model."""
        if not HAS_AIOHTTP or aiohttp is None:
            raise ImportError("aiohttp is required for external reward models. Install with: uv add aiohttp")
        
        payload = {
            "prompt": sample.prompt,
            "response": sample.response,
            "label": sample.label,
            "metadata": sample.metadata
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return float(result.get("reward", 0.0))
                    else:
                        return 0.0
        except Exception:
            return 0.0
    
    async def get_batch_rewards(self, samples: List[MLXSample]) -> List[float]:
        """Get rewards for batch of samples."""
        tasks = [self.get_reward(sample) for sample in samples]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)


# ==========================================
# MLX-NATIVE BRIDGE ADAPTERS
# ==========================================

def create_mlx_batch_adapter(
    reward_configs: Dict[str, MLXRewardConfig],
    external_models: Optional[Dict[str, MLXExternalRewardModel]] = None
) -> Union[Callable[[List[str], List[str], List[Dict[str, Any]]], mx.array], 
           Callable[[List[str], List[str], List[Dict[str, Any]]], Any]]:
    """
    Create MLX-native batch adapter for processing reward configurations.
    
    This function takes configuration patterns and creates MLX-optimized processors.
    """
    from .registry import create_configured_reward_function, RewardConfig
    
    # Convert MLX configs to our internal format
    internal_configs = []
    for name, config in reward_configs.items():
        internal_config = RewardConfig(
            name=name,
            weight=config.weight,
            params=config.params,
            verifiers=config.verifiers or [],
            verifier_penalty=config.verifier_penalty
        )
        internal_configs.append(internal_config)
    
    # Create base MLX processor
    from .mlx_batch_processor import create_mlx_optimized_batch_processor
    base_processor = create_mlx_optimized_batch_processor(internal_configs)
    
    # Add external model support if needed
    if external_models:
        async def async_processor(
            prompts: List[str],
            completions: List[str], 
            examples: List[Dict[str, Any]]
        ) -> mx.array:
            # Process local rewards
            local_rewards = base_processor(prompts, completions, examples)
            
            # Process external rewards
            external_rewards = []
            samples = [
                MLXSample(prompt=p, response=c, metadata=e) 
                for p, c, e in zip(prompts, completions, examples)
            ]
            
            for name, model in external_models.items():
                batch_rewards = await model.get_batch_rewards(samples)
                external_rewards.append(mx.array(batch_rewards))
            
            # Combine all rewards
            if external_rewards:
                all_rewards = mx.stack([local_rewards] + external_rewards, axis=1)
                return mx.mean(all_rewards, axis=1)  # Simple average
            else:
                return local_rewards
        
        return async_processor
    else:
        return base_processor


def samples_to_mlx_format(samples: List[MLXSample]) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Convert MLXSample list to our MLX processing format."""
    prompts = [s.prompt for s in samples]
    completions = [s.response for s in samples]
    examples = [s.to_dict()["example"] for s in samples]
    return prompts, completions, examples


def mlx_format_to_samples(
    prompts: List[str], 
    completions: List[str], 
    examples: List[Dict[str, Any]],
    rewards: mx.array
) -> List[MLXSample]:
    """Convert MLX format back to MLXSample list."""
    samples = []
    for i, (prompt, completion, example) in enumerate(zip(prompts, completions, examples)):
        sample = MLXSample(
            prompt=prompt,
            response=completion,
            label=example.get("label"),
            reward=float(rewards[i]),
            metadata=example.get("metadata", {}),
            status=MLXSample.Status.COMPLETED
        )
        samples.append(sample)
    return samples


# ==========================================
# ESSENTIAL MATH REWARD FUNCTIONS
# ==========================================

from .registry import reward

@reward(name="math_accuracy")
def math_accuracy_reward(
    prompt: str,
    completion: str, 
    example: Dict[str, Any],
    extract_boxed: bool = True,
    **kwargs
) -> float:
    """Math accuracy reward using boxed answer extraction."""
    ground_truth = example.get("label") or example.get("ground_truth")
    if not ground_truth:
        return 0.0
    
    # Extract answer if needed
    prediction = completion
    if extract_boxed:
        boxed = extract_boxed_answer(completion)
        if boxed:
            # Remove \boxed{} wrapper
            prediction = boxed[7:-1] if boxed.startswith("\\boxed{") and boxed.endswith("}") else boxed
    
    # Normalize both answers
    pred_normalized = normalize_math_answer(prediction)
    truth_normalized = normalize_math_answer(ground_truth)
    
    # Exact match
    if pred_normalized == truth_normalized:
        return 1.0
    
    # Fallback to F1 score
    return compute_f1_score(pred_normalized, truth_normalized)


@reward(name="f1_score")
def f1_score_reward(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    **kwargs
) -> float:
    """F1 score reward for text overlap measurement."""
    ground_truth = example.get("label") or example.get("ground_truth", "")
    return compute_f1_score(completion, ground_truth)


# ==========================================
# CONFIGURATION HELPERS
# ==========================================

def create_mlx_system_from_config(config_dict: Dict[str, Any]) -> Callable:
    """
    Create complete MLX reward system from configuration dictionary.
    
    This function creates MLX-optimized reward processors from configuration.
    """
    # Parse configuration
    reward_configs = {}
    external_models = {}
    
    for name, config_data in config_dict.items():
        if config_data.get("external_url"):
            # External model
            external_models[name] = MLXExternalRewardModel(
                url=config_data["external_url"],
                timeout=config_data.get("timeout", 30.0)
            )
        else:
            # Local MLX reward
            reward_configs[name] = MLXRewardConfig(**config_data)
    
    # Create adapter
    return create_mlx_batch_adapter(reward_configs, external_models if external_models else None)


def get_available_adapters() -> Dict[str, List[str]]:
    """List available adapter components."""
    return {
        "reward_functions": ["math_accuracy", "f1_score"],
        "math_utilities": ["extract_boxed_answer", "normalize_math_answer", "compute_f1_score"],
        "external_models": ["MLXExternalRewardModel"],
        "config_models": ["MLXRewardConfig", "MLXRewardSystemConfig"],
        "sample_types": ["MLXSample"]
    }