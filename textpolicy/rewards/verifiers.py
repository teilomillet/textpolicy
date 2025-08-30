# textpolicy/rewards/verifiers.py
"""
Text quality verifiers following retrain's decorator-based pattern.

Verifiers provide boolean pre-filtering for reward functions,
following retrain's philosophy of efficient quality control.

All verifiers follow the signature: (prompt: str, completion: str, example: Dict[str, Any]) -> bool
"""

import warnings
from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass
from enum import Enum
from .registry import verifier


warnings.warn(
    "VerificationResult and VerificationReport are deprecated; use boolean verifiers instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Legacy types for backward compatibility (deprecated)
class VerificationResult(Enum):
    """Result of verification check (deprecated - use boolean verifiers now)."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class VerificationReport:
    """Report from verification check (deprecated - use boolean verifiers now)."""
    result: VerificationResult
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    message: str


@verifier
def length_verifier(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    min_length: int = 10,
    max_length: int = 500,
    **kwargs
) -> bool:
    """
    Verifies response length appropriateness following retrain's pattern.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain length constraints)
        min_length: Minimum required word count
        max_length: Maximum allowed word count
        **kwargs: Additional parameters
        
    Returns:
        True if length is appropriate, False otherwise
    """
    # Get constraints from example if not provided
    if 'min_length' in example:
        min_length = example['min_length']
    if 'max_length' in example:
        max_length = example['max_length']
    
    word_count = len(completion.split())
    return min_length <= word_count <= max_length


@verifier
def toxicity_verifier(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    **kwargs
) -> bool:
    """
    Verifies response is not toxic or inappropriate following retrain's pattern.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain custom toxic patterns)
        **kwargs: Additional parameters
        
    Returns:
        True if content is non-toxic, False if toxic content detected
    """
    # Default toxic patterns - simple keyword-based detection
    # In practice, use a proper toxicity classifier
    toxic_patterns = [
        r'\b(hate|kill|die|stupid|idiot|racist|sexist)\b',
        r'\b(fuck|shit|damn|hell)\b',
        r'\b(violence|abuse|harassment)\b'
    ]
    
    # Allow custom patterns from example
    if 'toxic_patterns' in example:
        toxic_patterns.extend(example['toxic_patterns'])
    
    toxic_regex = re.compile('|'.join(toxic_patterns), re.IGNORECASE)
    matches = toxic_regex.findall(completion.lower())
    
    return len(matches) == 0


@verifier
def coherence_verifier(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    min_coherence_score: float = 0.5,
    **kwargs
) -> bool:
    """
    Verifies response coherence and logical flow following retrain's pattern.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain coherence requirements)
        min_coherence_score: Minimum coherence score threshold
        **kwargs: Additional parameters
        
    Returns:
        True if coherent enough, False otherwise
    """
    # Get threshold from example if specified
    if 'min_coherence_score' in example:
        min_coherence_score = example['min_coherence_score']
    
    # Simple coherence heuristics
    coherence_indicators = [
        r'\b(therefore|thus|however|moreover|furthermore)\b',
        r'\b(first|second|third|finally|in conclusion)\b',
        r'\b(because|since|as a result|consequently)\b'
    ]
    
    coherence_regex = re.compile('|'.join(coherence_indicators), re.IGNORECASE)
    
    # Check for logical connectors
    connectors = len(coherence_regex.findall(completion))
    
    # Check sentence structure
    sentences = re.split(r'[.!?]+', completion)
    valid_sentences = [s for s in sentences if s.strip()]
    if not valid_sentences:
        return False
        
    avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
    
    # Simple coherence score
    connector_score = min(1.0, connectors / 2.0)  # 2+ connectors is good
    sentence_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.5
    
    coherence_score = (connector_score + sentence_score) / 2.0
    
    return coherence_score >= min_coherence_score


@verifier
def factual_verifier(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    min_factual_score: float = 0.6,
    max_uncertainty_count: int = 2,
    **kwargs
) -> bool:
    """
    Verifies factual accuracy and consistency following retrain's pattern.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain factual requirements)
        min_factual_score: Minimum factual confidence score
        max_uncertainty_count: Maximum allowed uncertainty phrases
        **kwargs: Additional parameters
        
    Returns:
        True if factually sound, False otherwise
    """
    # Get thresholds from example if specified
    if 'min_factual_score' in example:
        min_factual_score = example['min_factual_score']
    if 'max_uncertainty_count' in example:
        max_uncertainty_count = example['max_uncertainty_count']
    
    # Uncertainty indicators that suggest factual issues
    uncertainty_phrases = [
        r'\b(i think|maybe|perhaps|possibly|not sure)\b',
        r'\b(might be|could be|i believe|seems like)\b',
        r'\b(probably|likely|unclear|unknown)\b'
    ]
    
    uncertainty_regex = re.compile('|'.join(uncertainty_phrases), re.IGNORECASE)
    
    # Count uncertainty phrases
    uncertainty_count = len(uncertainty_regex.findall(completion.lower()))
    
    # Check for contradictory statements (simple heuristic)
    contradictions = 0
    if 'yes' in completion.lower() and 'no' in completion.lower():
        contradictions += 1
    if 'true' in completion.lower() and 'false' in completion.lower():
        contradictions += 1
    
    # Calculate factual confidence
    uncertainty_penalty = min(1.0, uncertainty_count * 0.2)
    contradiction_penalty = contradictions * 0.3
    
    factual_score = max(0.0, 1.0 - uncertainty_penalty - contradiction_penalty)
    
    return factual_score >= min_factual_score and uncertainty_count <= max_uncertainty_count


# Additional verifiers following retrain's patterns

@verifier
def has_greeting(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    required_greeting: str = "hello",
    **kwargs
) -> bool:
    """
    Verifies that the completion contains a greeting.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may specify required greeting)
        required_greeting: The greeting phrase to look for
        **kwargs: Additional parameters
        
    Returns:
        True if greeting is present, False otherwise
    """
    if 'required_greeting' in example:
        required_greeting = example['required_greeting']
    
    return required_greeting.lower() in completion.lower()


@verifier
def no_empty_response(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    **kwargs
) -> bool:
    """
    Verifies that the completion is not empty or whitespace-only.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context
        **kwargs: Additional parameters
        
    Returns:
        True if completion has content, False if empty
    """
    return bool(completion.strip())


@verifier
def contains_keywords(
    prompt: str, 
    completion: str, 
    example: Dict[str, Any],
    required_keywords: Optional[List[str]] = None,
    **kwargs
) -> bool:
    """
    Verifies that the completion contains required keywords.
    
    Args:
        prompt: Input prompt
        completion: Generated response text
        example: Example data context (may contain required_keywords)
        required_keywords: List of keywords that must be present
        **kwargs: Additional parameters
        
    Returns:
        True if all required keywords are present, False otherwise
    """
    if required_keywords is None:
        required_keywords = example.get('required_keywords', [])
    
    if not required_keywords:
        return True  # No requirements, always pass
    
    completion_lower = completion.lower()
    return all(keyword.lower() in completion_lower for keyword in required_keywords)


# Legacy compatibility functions (deprecated)
def create_default_verifier_pipeline():
    """Create default verification pipeline (deprecated - use registry-based verifiers)."""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("create_default_verifier_pipeline is deprecated. Use registry-based verifiers with apply_verifiers_to_reward instead.")
    return None


def create_custom_verifier_pipeline(verifier_configs: List[Dict[str, Any]]):
    """Create custom verification pipeline (deprecated - use registry-based verifiers)."""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("create_custom_verifier_pipeline is deprecated. Use registry-based verifiers with apply_verifiers_to_reward instead.")
    return None


# Legacy class aliases for backward compatibility
class TextVerifier:
    """Base class for text quality verifiers (deprecated)."""
    def verify(self, prompt: str, response: str):
        # Legacy method: deprecated in favour of registry-based boolean verifiers
        warnings.warn(
            "TextVerifier.verify is deprecated; use registry-based @verifier functions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return False


class LengthVerifier(TextVerifier):
    """Legacy length verifier (deprecated - use length_verifier function)."""
    pass


class ToxicityVerifier(TextVerifier):
    """Legacy toxicity verifier (deprecated - use toxicity_verifier function)."""
    pass


class CoherenceVerifier(TextVerifier):
    """Legacy coherence verifier (deprecated - use coherence_verifier function)."""
    pass


class FactualVerifier(TextVerifier):
    """Legacy factual verifier (deprecated - use factual_verifier function)."""
    pass


class VerificationPipeline:
    """Legacy verification pipeline (deprecated - use registry-based approach)."""
    def __init__(self, verifiers):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("VerificationPipeline is deprecated. Use registry-based verifiers with apply_verifiers_to_reward instead.")
    
    def verify_batch(self, prompts, responses):
        # Legacy method: deprecated in favour of registry-based verification pipeline
        warnings.warn(
            "VerificationPipeline.verify_batch is deprecated; use registry-based verifiers instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []
