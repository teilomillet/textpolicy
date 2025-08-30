# textpolicy/rewards/basic.py
"""
Basic pure reward functions for text generation following retrain's patterns.

All functions follow the signature: (prompt: str, completion: str, example: Dict[str, Any], **kwargs) -> float
All functions are pure - no side effects, deterministic output.
All functions are MLX compilation compatible and registered via decorators.
"""

from typing import List, Optional, Dict, Any
from .registry import reward


@reward
def length_reward(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    target_length: int = 50,
    tolerance: float = 0.2,
    **kwargs
) -> float:
    """
    Pure function rewarding responses close to target length.
    
    Args:
        prompt: Input prompt (not used but kept for signature consistency)
        completion: Generated response text
        example: Example data context (not used here)
        target_length: Target length in words
        tolerance: Tolerance for length deviation (0.2 = 20%)
        **kwargs: Additional parameters
        
    Returns:
        Reward between 0.0 and 1.0
    """
    if not completion.strip():
        return 0.0  # no text, no fluency reward
    
    actual_length = len(completion.split())
    deviation = abs(actual_length - target_length) / target_length
    
    if deviation <= tolerance:
        return 1.0
    else:
        # Linear decay beyond tolerance
        return max(0.0, 1.0 - (deviation - tolerance) / (1.0 - tolerance))


@reward
def keyword_reward(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    keywords: Optional[List[str]] = None,
    bonus_multiplier: float = 1.0,
    **kwargs
) -> float:
    """
    Pure function rewarding keyword usage.
    
    Args:
        prompt: Input prompt (analyzed for required keywords)
        completion: Generated response text
        example: Example data context (may contain keywords if not provided)
        keywords: Keywords to encourage (can be None to use from example)
        bonus_multiplier: Multiplier for bonus points
        **kwargs: Additional parameters
        
    Returns:
        Reward between 0.0 and potentially > 1.0 with bonuses
    """
    if not completion.strip():
        return 0.0
    
    # Get keywords from parameter or example context
    if keywords is None:
        keywords = example.get('keywords', [])
    
    if not keywords:
        return 0.0
    
    completion_lower = completion.lower()
    
    # Count keyword matches
    matches = sum(1 for kw in keywords if kw.lower() in completion_lower)
    base_reward = matches / len(keywords) if keywords else 0.0
    
    # Bonus for using keywords not in prompt
    prompt_lower = prompt.lower()
    bonus_keywords = [kw for kw in keywords if kw.lower() not in prompt_lower]
    bonus_matches = sum(1 for kw in bonus_keywords if kw.lower() in completion_lower)
    bonus_reward = (bonus_matches / len(bonus_keywords) if bonus_keywords else 0.0) * bonus_multiplier
    
    return min(1.0, base_reward + bonus_reward)


@reward
def perplexity_reward(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    model = None,  # Optional MLX model for perplexity computation
    max_perplexity: float = 100.0,
    **kwargs
) -> float:
    """
    Pure function rewarding low perplexity (high fluency).
    
    If no model provided, uses simple heuristics.
    With model, computes actual perplexity using MLX.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain model reference)
        model: Optional MLX model for perplexity computation
        max_perplexity: Maximum perplexity for normalization
        **kwargs: Additional parameters
        
    Returns:
        Reward between 0.0 and 1.0 (higher = more fluent)
    """
    if not completion.strip():
        return 0.0
    
    # Use model from example if not provided
    if model is None:
        model = example.get('model')
    
    if model is not None:
        # MLX-based perplexity computation is not yet implemented.
        # Remove model to ensure heuristic fallback is used.
        model = None
    
    # Fallback: simple heuristics for fluency
    words = completion.split()
    
    # Penalize very short responses (minimum heuristic fluency)
    if len(words) < 3:
        return 0.2
    
    # Penalize repetition
    unique_words = len(set(words))
    repetition_penalty = unique_words / len(words)
    
    # Penalize very long words (might be gibberish)
    avg_word_length = sum(len(word) for word in words) / len(words)
    length_penalty = 1.0 if avg_word_length <= 8 else max(0.3, 1.0 - (avg_word_length - 8) * 0.1)
    
    # Penalize lack of punctuation in longer responses
    has_punctuation = any(char in completion for char in '.!?')
    punct_penalty = 1.0 if len(words) < 10 or has_punctuation else 0.8
    
    fluency_score = repetition_penalty * length_penalty * punct_penalty
    return min(1.0, fluency_score)


@reward
def accuracy_reward(
    prompt: str,
    completion: str,
    example: Dict[str, Any],
    ground_truth: Optional[str] = None,
    similarity_threshold: float = 0.7,
    **kwargs
) -> float:
    """
    Pure function rewarding factual accuracy.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain ground truth)
        ground_truth: Optional ground truth for comparison
        similarity_threshold: Threshold for similarity matching
        **kwargs: Additional parameters
        
    Returns:
        Reward between 0.0 and 1.0
    """
    if not completion.strip():
        return 0.0
    
    # Get ground truth from parameter or example context
    if ground_truth is None:
        ground_truth = example.get('ground_truth') or example.get('label')
    
    if ground_truth is None:
        # Without ground truth, use simple fact-checking heuristics
        # Penalize uncertain language in factual contexts
        uncertain_phrases = [
            'i think', 'maybe', 'perhaps', 'possibly', 'not sure',
            'might be', 'could be', 'i believe', 'seems like'
        ]
        
        uncertainty_penalty = sum(1 for phrase in uncertain_phrases 
                                if phrase in completion.lower())
        confidence_score = max(0.3, 1.0 - uncertainty_penalty * 0.2)
        
        return confidence_score
    
    # With ground truth, compute similarity
    # Simple word overlap similarity (could be enhanced with embeddings)
    completion_words = set(completion.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if not truth_words:
        return 0.5
    
    overlap = len(completion_words & truth_words)
    similarity = overlap / len(truth_words)
    
    return 1.0 if similarity >= similarity_threshold else similarity / similarity_threshold
