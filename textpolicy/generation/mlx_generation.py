# textpolicy/generation/mlx_generation.py
"""
Complete MLX-LM text generation functions for RL training.

This module provides proper integration with MLX-LM for text generation RL,
including correct logprob extraction for policy gradient training.

Key functions:
- load_model: Load MLX model and tokenizer
- generate_tokens: Generate text with logprob tracking  
- compute_logprobs: Extract logprobs for RL training
- create_policy: Create policy function for rollout collection
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Any, Callable
import mlx.core as mx
import mlx.nn as nn
try:
    from mlx_lm import load, generate
    HAS_MLX_LM = True
except ImportError:
    HAS_MLX_LM = False
    print("Warning: mlx_lm not found. Using fallback implementations.")

try:
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
# sampling utilities fallback when sample_utils is unavailable
except ImportError:
    make_sampler = None
    make_logits_processors = None


def _get_eos_configs_for_model(
    model_path: str,
    tokenizer_config: Optional[Dict]
) -> Tuple[Optional[Dict], Dict[str, Any]]:
    """
    Determine tokenizer_config and model_config for proper EOS handling based on model type.
    """
    model_config: Dict[str, Any] = {}
    if tokenizer_config is None and "Qwen" in model_path:
        tokenizer_config = {}
    if "Qwen" in model_path:
        # For Qwen Instruct variants, let tokenizer.eos_token_id (<|im_end|>) prevail;
        # override only for base Qwen to use <|endoftext|> (151643) as EOS.
        if "Instruct" not in model_path:
            eos_id = 151643
            model_config["eos_token_id"] = eos_id
    return tokenizer_config, model_config


def _prepare_tokenizer(tokenizer: Any, verbose: bool) -> None:
    """
    Configure tokenizer verbosity and ensure EOS token IDs for stopping.
    """
    tokenizer.verbose = verbose
    # Force tokenizer's EOS to Qwen's natural <|endoftext|> when available
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is not None:
        # Override tokenizer.eos_token to match eos_token_id for natural stopping
        tokenizer.eos_token_id = eos_id
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(eos_id)
        tokenizer.eos_token_ids = [eos_id]
        # Align pad_token to EOS to ensure MLX-LM uses EOS for padding/stopping
        tokenizer.pad_token_id = eos_id
        tokenizer.pad_token = tokenizer.eos_token


def _make_eos_safe_sampler(temp: float, top_p: float) -> Any:
    """
    Build a sampler that does not prune low-probability tokens (e.g., EOS) and encourages natural stopping.
    """
    if make_sampler is not None:
        # Use more conservative sampling parameters to encourage natural EOS generation
        # Lower min_p and ensure we keep more tokens in consideration
        return make_sampler(
            temp=temp,
            top_p=top_p,
            min_p=0.0,  # Don't filter out low-probability tokens like EOS
            min_tokens_to_keep=2,  # Keep at least 2 tokens to ensure EOS has a chance
        )
    else:
        # Fallback implementation when mlx_lm.sample_utils is not available
        return None


def _make_logits_processors(repetition_penalty: float) -> Any:
    """
    Create logits processors to enforce repetition penalty.
    """
    if make_logits_processors is not None:
        return make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=20,
        )
    else:
        # Fallback implementation when mlx_lm.sample_utils is not available
        return None


def _extract_response_tokens(
    response: Any,
    prompt_list: Any,
    tokenizer: Any,
) -> mx.array:
    """
    Extract response token IDs from a raw generation output (string or list).
    Enhanced to better handle EOS tokens and edge cases.
    """
    if isinstance(response, list):
        return mx.array(response)
    try:
        full_tokens = tokenizer.encode(response)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        
        # First, try to find EOS token and include it in response
        if eos_id is not None and eos_id in full_tokens:
            idx = full_tokens.index(eos_id)
            # Include EOS token in response for proper reward calculation
            resp = full_tokens[len(prompt_list): idx + 1]
        else:
            # No EOS found - extract response portion without EOS
            try:
                prompt_text = tokenizer.decode(prompt_list)
                if response.startswith(prompt_text):
                    tail = response[len(prompt_text):]
                else:
                    tail = response
                resp = tokenizer.encode(tail.strip()) if tail.strip() else []
            except Exception:
                # Fallback: encode the whole response and hope for the best
                resp = tokenizer.encode(response) if response else []
        
        return mx.array(resp) if resp else mx.array([])
    except Exception:
        return mx.array([])


def load_model(
    model_path: str, 
    adapter_path: Optional[str] = None,
    tokenizer_config: Optional[Dict] = None,
    verbose: bool = False
) -> Tuple[nn.Module, Any]:
    """
    Load MLX model and tokenizer for RL training.
    
    This function properly loads MLX-LM models with support for LoRA adapters
    and ensures compatibility with our training system. Automatically configures
    proper EOS tokens for Qwen models to ensure correct generation stopping.
    
    Args:
        model_path: Path or HuggingFace model ID  
        adapter_path: Optional LoRA adapter path
        tokenizer_config: Optional tokenizer configuration for EOS tokens
        verbose: Enable debug logging for chat template application
        
    Returns:
        (model, tokenizer): MLX model and tokenizer instances
    """
    if not HAS_MLX_LM:
        raise ImportError("mlx_lm is required. Install with: pip install mlx-lm")
    
    print(f"Loading MLX model: {model_path}")
    if adapter_path:
        print(f"Loading with LoRA adapters: {adapter_path}")
    
    # Configure model & tokenizer for EOS handling based on model type
    tokenizer_config, model_config = _get_eos_configs_for_model(model_path, tokenizer_config)
    model, tokenizer = load(
        path_or_hf_repo=model_path,
        adapter_path=adapter_path,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        lazy=False,
    )
    _prepare_tokenizer(tokenizer, verbose)
    print("✓ Model loaded successfully")
    return model, tokenizer


def generate_tokens(
    model: nn.Module,
    tokenizer: Any,
    prompt_tokens: mx.array,
    max_tokens: int = 50,
    temperature: float = 0.7,  # Lower default temperature for more stable generation
    top_p: float = 0.9,        # Lower top_p for more focused sampling
    repetition_penalty: float = 1.1  # Add repetition penalty to prevent loops
) -> Tuple[mx.array, Dict[str, Any]]:
    """Generate response tokens with proper MLX-LM integration and EOS token support."""
    if not HAS_MLX_LM:
        return _simple_generate(model, prompt_tokens, max_tokens, temperature)
    
    prompt_list = prompt_tokens.tolist()
    response_token_list: list = []
    response_logprob_list: list = []

    # Use stream_generate instead of generate to get proper EOS token handling
    # This is the core fix - stream_generate respects EOS tokens, generate() does not
    try:
        from mlx_lm import stream_generate
        
        # EOS-safe sampling with reduced temperature for more predictable stopping
        optimized_temperature = min(temperature, 0.7)
        sampler = _make_eos_safe_sampler(optimized_temperature, top_p)
        logits_processors = _make_logits_processors(repetition_penalty) if _make_logits_processors is not None else None


        # Use stream_generate to get token-by-token generation with EOS detection
        response_segments = list(stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_list,  # type: ignore
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ))
        
        # Extract tokens and logprobs from response segments
        for segment in response_segments:
            response_token_list.append(segment.token)
            # Capture per-token logprob inline to avoid a redundant forward pass
            if segment.logprobs is not None:
                response_logprob_list.append(float(segment.logprobs[segment.token]))
            # Check if this segment indicates natural stopping (EOS token)
            if hasattr(segment, 'finish_reason') and segment.finish_reason == "stop":
                break

        # Convert to MLX array
        response_tokens = mx.array(response_token_list) if response_token_list else mx.array([])


    except ImportError:
        # Fallback to original generate method if stream_generate unavailable
        print("WARNING: stream_generate not available, using fallback generate method")
        optimized_temperature = min(temperature, 0.7)
        sampler = _make_eos_safe_sampler(optimized_temperature, top_p)
        logits_processors = _make_logits_processors(repetition_penalty) if _make_logits_processors is not None else None

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_list,  # type: ignore
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        
        response_tokens = _extract_response_tokens(response, prompt_list, tokenizer)
    
    # Use inline logprobs captured during generation when available,
    # falling back to a full forward pass only if logprobs were missing.
    if response_logprob_list and len(response_logprob_list) == len(response_token_list):
        logprobs = mx.array(response_logprob_list)
    else:
        logprobs = compute_logprobs(model, prompt_tokens, response_tokens)
    return response_tokens, {'logprob': logprobs}


def _truncate_repetitive_text(text: str, max_repetitions: int = 3) -> str:
    """
    Truncate text if it contains excessive repetitions.
    
    This helps prevent the model from generating endless loops of the same tokens.
    """
    words = text.split()
    if len(words) < 4:
        return text
    
    # Check for word repetition
    for i in range(len(words) - max_repetitions):
        if len(set(words[i:i+max_repetitions])) == 1:
            # Found repetition, truncate here
            return ' '.join(words[:i])
    
    # Check for character repetition (like "5555555")
    for i in range(len(text) - max_repetitions):
        if len(set(text[i:i+max_repetitions])) == 1:
            # Found character repetition, truncate here
            return text[:i]
    
    return text


def _simple_generate(
    model: nn.Module, 
    prompt_tokens: mx.array, 
    max_tokens: int, 
    temperature: float
) -> Tuple[mx.array, Dict[str, Any]]:
    """
    Simple fallback generation for development without MLX-LM.
    
    This provides basic autoregressive generation for testing when
    MLX-LM is not available.
    """
    current_tokens = prompt_tokens
    generated = []
    generated_logprobs = []

    for _ in range(max_tokens):
        # Model forward pass
        logits = model(current_tokens[None])  # Add batch dimension
        next_token_logits = logits[0, -1, :]  # Last token logits

        # Temperature scaling
        if temperature > 0:
            scaled_logits = next_token_logits / temperature
        else:
            scaled_logits = next_token_logits

        # Sample next token
        probs = mx.softmax(scaled_logits)
        next_token = mx.random.categorical(probs[None])[0]

        # Capture logprob inline: log_softmax of the *unscaled* logits at the selected token
        log_probs = next_token_logits - mx.logsumexp(next_token_logits)
        generated_logprobs.append(float(log_probs[next_token]))

        # Add to sequence
        generated.append(next_token)
        current_tokens = mx.concatenate([current_tokens, next_token[None]])

        # Stop on EOS (approximate) - avoid .item() calls
        if len(generated) > 5 and next_token < 5:  # Simple stop condition
            break

    response_tokens = mx.array(generated) if generated else mx.array([2])

    # Use inline logprobs captured during generation (avoids redundant forward pass)
    if generated_logprobs and len(generated_logprobs) == len(generated):
        logprobs = mx.array(generated_logprobs)
    else:
        logprobs = compute_logprobs(model, prompt_tokens, response_tokens)

    return response_tokens, {'logprob': logprobs}


def compute_logprobs(
    model: nn.Module,
    prompt_tokens: mx.array,
    response_tokens: mx.array
) -> mx.array:
    """
    Extract log-probabilities of response_tokens under model via teacher-forcing.
    Raises on dimension mismatch or invalid (nan/inf/positive) values.
    """
    if len(response_tokens) == 0:
        return mx.array([])
    
    full_sequence = mx.concatenate([prompt_tokens, response_tokens])
    model_input = full_sequence[None] if full_sequence.ndim == 1 else full_sequence
    logits = model(model_input)
    prompt_len, response_len = len(prompt_tokens), len(response_tokens)
    prediction_logits = logits[0, prompt_len-1:prompt_len-1+response_len, :]
    if prediction_logits.shape[0] != response_len:
        raise ValueError(
            f"Logits/tokens mismatch: {prediction_logits.shape[0]} vs {response_len}"
        )
    
    log_probs = prediction_logits - mx.logsumexp(prediction_logits, axis=-1, keepdims=True)
    selected = log_probs[mx.arange(response_len), response_tokens]
    if mx.any(mx.isnan(selected)) or mx.any(mx.isinf(selected)):
        raise ValueError("Invalid logprobs (nan/inf)")
    if mx.any(selected > 0):
        print("Warning: positive logprobs detected")
    return selected


def compute_logprobs_batched(
    model: nn.Module,
    full_sequences: mx.array,
    response_tokens: mx.array,
    prompt_lengths: list,
    response_lengths: list,
) -> mx.array:
    """
    Batched log-probability extraction: single forward pass for N episodes.

    Replaces N sequential ``compute_logprobs`` calls with one batched
    ``model(full_sequences)`` call, converting O(N) serial model invocations
    into a single parallel operation.

    Args:
        model: Causal language model.
        full_sequences: ``[N, max_seq_len]`` right-padded prompt+response tokens.
        response_tokens: ``[N, max_resp_len]`` right-padded response-only tokens.
        prompt_lengths: Per-episode prompt token count (Python list of ints).
        response_lengths: Per-episode response token count (Python list of ints).

    Returns:
        Flat 1D unpadded logprobs — ``shape[0] == sum(response_lengths)``.

    Safety notes:
        - Right-padding is safe for causal models (padded tokens are right of
          real tokens and cannot influence them via causal attention).
        - The Python loop over episodes is cheap indexing, not model calls.
        - No ``.item()`` or ``mx.eval()`` calls — safe inside ``mx.compile``.
    """
    n_episodes = full_sequences.shape[0]

    if n_episodes == 0:
        return mx.array([], dtype=mx.float32)

    # Defensive shape check: all episode-indexed inputs must have N entries.
    # A mismatch here means _pack_episodes dropped rows (e.g. filtering out
    # empty episodes before stacking) — catch it early with a clear message.
    if len(prompt_lengths) != n_episodes:
        raise ValueError(
            f"prompt_lengths has {len(prompt_lengths)} entries but "
            f"full_sequences has {n_episodes} rows. They must match."
        )
    if len(response_lengths) != n_episodes:
        raise ValueError(
            f"response_lengths has {len(response_lengths)} entries but "
            f"full_sequences has {n_episodes} rows. They must match."
        )
    if response_tokens.shape[0] != n_episodes:
        raise ValueError(
            f"response_tokens has {response_tokens.shape[0]} rows but "
            f"full_sequences has {n_episodes} rows. They must match."
        )

    # Single batched forward pass: [N, max_seq_len] → [N, max_seq_len, vocab]
    logits = model(full_sequences)

    # Extract per-episode logprobs (cheap indexing, not model calls)
    per_episode = []
    for i in range(n_episodes):
        p_len = prompt_lengths[i]
        r_len = response_lengths[i]

        if r_len == 0:
            continue

        # Guard: p_len == 0 would make the slice index (p_len - 1) wrap to
        # -1 (the last token), silently corrupting logprobs. A prompt of
        # length 0 is invalid for causal logprob extraction — you need at
        # least one token to condition on.
        if p_len == 0:
            raise ValueError(
                f"Episode {i} has prompt_length=0. Causal logprob extraction "
                f"requires at least 1 prompt token (the model needs context "
                f"to predict response tokens)."
            )

        # Prediction logits: same slicing as compute_logprobs
        # logits at position (p_len-1) predicts the first response token
        prediction_logits = logits[i, p_len - 1 : p_len - 1 + r_len, :]

        # Log-softmax → select the actual response token at each position
        log_probs = prediction_logits - mx.logsumexp(
            prediction_logits, axis=-1, keepdims=True
        )
        selected = log_probs[mx.arange(r_len), response_tokens[i, :r_len]]
        per_episode.append(selected)

    if not per_episode:
        return mx.array([], dtype=mx.float32)

    return mx.concatenate(per_episode)


def encode(tokenizer: Any, text: str) -> mx.array:
    """
    Convert text to MLX token array.
    
    Args:
        tokenizer: MLX tokenizer
        text: Input text string
        
    Returns:
        Token array as MLX array
    """
    tokens = tokenizer.encode(text)
    return mx.array(tokens, dtype=mx.int32)


def decode(tokenizer: Any, tokens: mx.array) -> str:
    """
    Convert MLX token array to text.
    
    Args:
        tokenizer: MLX tokenizer  
        tokens: Token array
        
    Returns:
        Decoded text string
    """
    token_list = tokens.tolist()
    return tokenizer.decode(token_list)


def create_policy(
    model: nn.Module,
    tokenizer: Any,
    generation_params: Optional[Dict[str, Any]] = None
) -> Callable[[mx.array], Tuple[mx.array, Dict[str, Any]]]:
    """
    Create a policy function for RL training with automatic chat template support.
    
    This returns a pure function that can be used by rollout systems
    to generate responses and collect the data needed for training.
    
    Automatically applies chat templates for instruction models
    to enable proper EOS token generation and natural stopping behavior.
    
    Args:
        model: MLX model
        tokenizer: MLX tokenizer
        generation_params: Generation parameters (max_tokens, temperature, etc.)
        
    Returns:
        Policy function: (prompt_tokens) -> (response_tokens, info)
    """
    params = generation_params or {}
    max_tokens = params.get('max_tokens', 50)
    temperature = params.get('temperature', 0.8)
    top_p = params.get('top_p', 0.95)
    
    def policy_fn(prompt_tokens: mx.array, deterministic: bool = False) -> Tuple[mx.array, Dict[str, Any]]:
        """
        Policy function that generates responses for RL training with automatic chat template support.
        
        Automatically applies chat templates for instruction models to enable
        proper EOS token generation. This allows models to naturally end responses with
        appropriate end-of-sequence tokens instead of being artificially truncated.
        
        Args:
            prompt_tokens: Input prompt tokens
            deterministic: Whether to use deterministic generation
            
        Returns:
            (response_tokens, generation_info): Response and metadata for training
        """
        # Auto-apply chat template for instruction models
        processed_tokens = prompt_tokens
        
        try:
            # Decode tokens to check if chat template is needed
            if hasattr(tokenizer, 'decode'):
                raw_prompt = tokenizer.decode(prompt_tokens.tolist())
            else:
                # Fallback for tokenizers without decode method
                raw_prompt = str(prompt_tokens.tolist())
            
            # Let the tokenizer decide if chat template is needed
            # This works for ANY instruction model (Qwen, Llama, Mistral, etc.)
            needs_formatting = (
                hasattr(tokenizer, 'apply_chat_template') and
                # Only apply if not already formatted (avoid double-formatting)
                not any(marker in raw_prompt for marker in ['<|im_start|>', '<|endoftext|>', '<|assistant|>'])
            )
            
            if needs_formatting:
                # Convert to messages format and apply chat template
                # This uses the tokenizer's built-in knowledge of its own chat format
                messages = [{"role": "user", "content": raw_prompt.strip()}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # Adds <|im_start|>assistant\n for response generation
                )
                
                # Re-encode with proper formatting for EOS generation
                if hasattr(tokenizer, 'encode'):
                    processed_tokens = mx.array(tokenizer.encode(formatted_prompt))
                else:
                    # Fallback if tokenizer doesn't have encode method
                    processed_tokens = prompt_tokens
                
                # Debug logging (only in verbose mode to avoid noise)
                if hasattr(tokenizer, 'verbose') and tokenizer.verbose:
                    print(f"Applied chat template: '{formatted_prompt[:100]}...'")  # Show first 100 chars for debugging
            
        except Exception:
            # Fallback to original tokens if formatting fails
            # This ensures robustness and backward compatibility
            pass
        
        # Generate response with processed tokens
        temp = 0.0 if deterministic else temperature
        return generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=processed_tokens,  # Use formatted tokens for proper EOS generation
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p
        )
    
    return policy_fn


def compute_reward(
    prompt: str,
    response: str,
    reward_type: str = "length",
    **kwargs
) -> float:
    """
    Simple reward computation for RL training.
    
    This provides basic reward functions for testing. In practice,
    you would use the sophisticated reward system from textpolicy.rewards.
    
    Args:
        prompt: Input prompt text
        response: Generated response text
        reward_type: Type of reward to compute
        **kwargs: Additional parameters for reward computation
        
    Returns:
        Reward score
    """
    if reward_type == "length":
        target_length = kwargs.get('target_length', 30)
        actual_length = len(response.split())
        # Simple length-based reward
        diff = abs(actual_length - target_length)
        return max(0.0, 1.0 - diff / target_length)
    
    elif reward_type == "keyword":
        keywords = kwargs.get('keywords', ['good', 'great', 'excellent'])
        count = sum(1 for kw in keywords if kw.lower() in response.lower())
        return count / len(keywords)
    
    else:
        # Default: simple response quality heuristic
        if len(response.strip()) == 0:
            return 0.0
        if len(response.split()) < 5:
            return 0.2
        return 0.5


# Convenience function for complete setup
def create_setup(
    model_path: str,
    generation_params: Optional[Dict[str, Any]] = None,
    adapter_path: Optional[str] = None
) -> Tuple[Callable, nn.Module, Any]:
    """
    Complete setup for MLX-LM RL training.
    
    This function combines model loading and policy creation for
    convenient setup of RL training systems.
    
    Args:
        model_path: Path or HuggingFace model ID
        generation_params: Generation parameters
        adapter_path: Optional LoRA adapter path
        
    Returns:
        (policy_fn, model, tokenizer): Complete setup for RL training
    """
    # Load model and tokenizer
    model, tokenizer = load_model(model_path, adapter_path)
    
    # Create policy function
    policy_fn = create_policy(model, tokenizer, generation_params)
    
    return policy_fn, model, tokenizer
