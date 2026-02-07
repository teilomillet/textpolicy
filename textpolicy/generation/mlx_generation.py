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
import importlib
from typing import Dict, List, Optional, Tuple, Any, Callable
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


def _maybe_apply_chat_template(tokenizer: Any, prompt_tokens: mx.array) -> mx.array:
    """Apply chat template to a prompt when tokenizer support is available."""
    processed_tokens = prompt_tokens

    try:
        if hasattr(tokenizer, "decode"):
            raw_prompt = tokenizer.decode(prompt_tokens.tolist())
        else:
            raw_prompt = str(prompt_tokens.tolist())

        needs_formatting = (
            hasattr(tokenizer, "apply_chat_template")
            and not any(
                marker in raw_prompt
                for marker in ["<|im_start|>", "<|endoftext|>", "<|assistant|>"]
            )
        )

        if needs_formatting:
            messages = [{"role": "user", "content": raw_prompt.strip()}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if hasattr(tokenizer, "encode"):
                processed_tokens = mx.array(tokenizer.encode(formatted_prompt))
            if hasattr(tokenizer, "verbose") and tokenizer.verbose:
                print(f"Applied chat template: '{formatted_prompt[:100]}...'")
    except Exception:
        # Keep raw prompt tokens on any formatting failure.
        pass

    return processed_tokens


def _resolve_pad_token_id(tokenizer: Any) -> int:
    """Resolve pad token id with robust fallback."""
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    return int(pad_token_id)


def _resolve_eos_token_ids(tokenizer: Any) -> List[int]:
    """Resolve EOS token ids as a Python list."""
    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids is not None:
        return [int(t) for t in eos_token_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return [int(eos_token_id)]
    return []


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
    response_tokens: mx.array,
    *,
    _compiled: bool = False,
    return_token_entropies: bool = False,
) -> Any:
    """
    Extract log-probabilities of response_tokens under model via teacher-forcing.

    Raises on dimension mismatch or invalid (nan/inf/positive) values when
    ``_compiled=False`` (the default).  When ``_compiled=True``, NaN/Inf
    values are replaced with ``finfo(dtype).min`` using compile-safe
    ``mx.where`` instead — Python branching on ``mx.any(...)`` is illegal
    inside ``mx.compile`` traced functions.

    Args:
        _compiled: When True, skip ``mx.any``-based validation (uses
            ``mx.where`` sanitization instead).  Set by the Trainer when
            ``_loss_fn`` is wrapped with ``mx.compile``.
        return_token_entropies: When True, return a tuple
            ``(logprobs, token_entropies)`` where token entropies are
            Shannon entropy values per response token.
    """
    if len(response_tokens) == 0:
        empty = mx.array([])
        if return_token_entropies:
            return empty, empty
        return empty

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
    token_entropies = -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)

    if _compiled:
        # Inside mx.compile: Python ``if`` on ``mx.any(...)`` is illegal
        # (triggers ``.item()``).  Use ``mx.where`` to sanitize NaN/Inf
        # on the computation graph instead.  See _default_get_logprobs
        # in trainer.py for the same pattern.
        sentinel = mx.array(mx.finfo(selected.dtype).min, dtype=selected.dtype)
        selected = mx.where(
            mx.isnan(selected) | mx.isinf(selected),
            sentinel,
            selected,
        )
        token_entropies = mx.where(
            mx.isnan(token_entropies) | mx.isinf(token_entropies),
            mx.array(0.0, dtype=token_entropies.dtype),
            token_entropies,
        )
    else:
        # Outside compilation: eager validation with clear error messages.
        # The mx.any() calls also act as implicit sync barriers, which
        # preserves evaluation ordering for sequential callers.
        if mx.any(mx.isnan(selected)) or mx.any(mx.isinf(selected)):
            raise ValueError("Invalid logprobs (nan/inf)")
        if mx.any(selected > 0):
            print("Warning: positive logprobs detected")
        if mx.any(mx.isnan(token_entropies)) or mx.any(mx.isinf(token_entropies)):
            raise ValueError("Invalid token entropies (nan/inf)")

    if return_token_entropies:
        return selected, token_entropies
    return selected


def compute_prompt_reuse_stats(
    full_sequences: mx.array,
    prompt_lengths: List[int],
    response_lengths: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Quantify repeated-prompt opportunity in a batched workload.

    The returned metrics are designed for issue #29 triage:
    they estimate how much prompt-side work could be removed if identical
    prompts were prefetched once and reused.

    Args:
        full_sequences: 2D tensor ``[N, max_seq_len]`` containing right-padded
            prompt+response tokens.
        prompt_lengths: Prompt token count for each episode.
        response_lengths: Optional response token count for each episode.

    Returns:
        Dict with repeat-rate and upper-bound token-savings metrics.
    """
    if full_sequences.ndim != 2:
        raise ValueError(
            f"full_sequences must be 2D [N, max_seq_len], got {full_sequences.ndim}D "
            f"with shape {full_sequences.shape}."
        )

    n_episodes = int(full_sequences.shape[0])
    if len(prompt_lengths) != n_episodes:
        raise ValueError(
            f"prompt_lengths has {len(prompt_lengths)} entries but "
            f"full_sequences has {n_episodes} rows. They must match."
        )
    if any(p <= 0 for p in prompt_lengths):
        raise ValueError("All prompt_lengths must be >= 1.")

    max_seq_len = int(full_sequences.shape[1])
    if any(p > max_seq_len for p in prompt_lengths):
        raise ValueError(
            f"At least one prompt length exceeds sequence width {max_seq_len}."
        )

    if response_lengths is not None:
        if len(response_lengths) != n_episodes:
            raise ValueError(
                f"response_lengths has {len(response_lengths)} entries but "
                f"full_sequences has {n_episodes} rows. They must match."
            )
        if any(r < 0 for r in response_lengths):
            raise ValueError("All response_lengths must be >= 0.")

    if n_episodes == 0:
        result: Dict[str, float] = {
            "num_episodes": 0.0,
            "unique_prompts": 0.0,
            "repeated_prompt_groups": 0.0,
            "duplicated_episodes": 0.0,
            "repeat_rate": 0.0,
            "max_group_size": 0.0,
            "mean_group_size": 0.0,
            "total_prompt_tokens": 0.0,
            "duplicated_prompt_tokens": 0.0,
            "prompt_token_reduction_upper_bound": 0.0,
        }
        if response_lengths is not None:
            result["total_response_tokens"] = 0.0
            result["total_tokens"] = 0.0
            result["end_to_end_token_reduction_upper_bound"] = 0.0
        return result

    # Group episodes by the exact unpadded prompt token tuple.
    prompt_counts: Dict[Tuple[int, ...], int] = {}
    prompt_token_lengths: Dict[Tuple[int, ...], int] = {}
    for i, p_len in enumerate(prompt_lengths):
        prompt_key = tuple(full_sequences[i, :p_len].tolist())
        prompt_counts[prompt_key] = prompt_counts.get(prompt_key, 0) + 1
        prompt_token_lengths[prompt_key] = p_len

    unique_prompts = len(prompt_counts)
    duplicated_episodes = n_episodes - unique_prompts
    repeated_prompt_groups = sum(1 for c in prompt_counts.values() if c > 1)
    max_group_size = max(prompt_counts.values())
    mean_group_size = n_episodes / unique_prompts

    total_prompt_tokens = int(sum(prompt_lengths))
    duplicated_prompt_tokens = int(
        sum(
            (count - 1) * prompt_token_lengths[prompt_key]
            for prompt_key, count in prompt_counts.items()
            if count > 1
        )
    )
    prompt_token_reduction_upper_bound = (
        duplicated_prompt_tokens / total_prompt_tokens
        if total_prompt_tokens > 0
        else 0.0
    )

    result = {
        "num_episodes": float(n_episodes),
        "unique_prompts": float(unique_prompts),
        "repeated_prompt_groups": float(repeated_prompt_groups),
        "duplicated_episodes": float(duplicated_episodes),
        "repeat_rate": duplicated_episodes / n_episodes,
        "max_group_size": float(max_group_size),
        "mean_group_size": float(mean_group_size),
        "total_prompt_tokens": float(total_prompt_tokens),
        "duplicated_prompt_tokens": float(duplicated_prompt_tokens),
        "prompt_token_reduction_upper_bound": float(prompt_token_reduction_upper_bound),
    }

    if response_lengths is not None:
        total_response_tokens = int(sum(response_lengths))
        total_tokens = total_prompt_tokens + total_response_tokens
        end_to_end_token_reduction_upper_bound = (
            duplicated_prompt_tokens / total_tokens if total_tokens > 0 else 0.0
        )
        result["total_response_tokens"] = float(total_response_tokens)
        result["total_tokens"] = float(total_tokens)
        result["end_to_end_token_reduction_upper_bound"] = float(
            end_to_end_token_reduction_upper_bound
        )

    return result


def compute_logprobs_batched(
    model: nn.Module,
    full_sequences: mx.array,
    response_tokens: mx.array,
    prompt_lengths: list,
    response_lengths: list,
    return_token_entropies: bool = False,
) -> Any:
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
        When ``return_token_entropies=True``, returns a tuple
        ``(logprobs, token_entropies)`` with matching flat 1D shapes.

    Safety notes:
        - Right-padding is safe for causal models (padded tokens are right of
          real tokens and cannot influence them via causal attention).
        - The Python loop over episodes is cheap indexing, not model calls.
        - No ``.item()`` or ``mx.eval()`` calls — safe inside ``mx.compile``.
    """
    if full_sequences.ndim != 2:
        raise ValueError(
            f"full_sequences must be 2D [N, max_seq_len], got {full_sequences.ndim}D "
            f"with shape {full_sequences.shape}."
        )
    if response_tokens.ndim != 2:
        raise ValueError(
            f"response_tokens must be 2D [N, max_resp_len], got {response_tokens.ndim}D "
            f"with shape {response_tokens.shape}."
        )

    n_episodes = full_sequences.shape[0]

    if n_episodes == 0:
        empty = mx.array([], dtype=mx.float32)
        if return_token_entropies:
            return empty, empty
        return empty

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
    max_r_len = max(response_lengths) if response_lengths else 0
    if max_r_len > 0 and response_tokens.shape[1] < max_r_len:
        raise ValueError(
            f"response_tokens has {response_tokens.shape[1]} columns but "
            f"max(response_lengths)={max_r_len}. The response_tokens tensor "
            f"must be wide enough to hold the longest response."
        )

    # Single batched forward pass: [N, max_seq_len] → [N, max_seq_len, vocab]
    logits = model(full_sequences)

    # Extract per-episode logprobs (cheap indexing, not model calls)
    per_episode = []
    per_episode_entropies = []
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
        entropies = -mx.sum(mx.exp(log_probs) * log_probs, axis=-1)
        per_episode.append(selected)
        per_episode_entropies.append(entropies)

    if not per_episode:
        empty = mx.array([], dtype=mx.float32)
        if return_token_entropies:
            return empty, empty
        return empty

    result = mx.concatenate(per_episode)
    entropy_result = mx.concatenate(per_episode_entropies)

    # Sanitize NaN/Inf — compile-safe (no Python branching on array
    # values).  Matches the handling in compute_logprobs(_compiled=True)
    # and _default_get_logprobs so all three extraction paths behave
    # consistently.
    sentinel = mx.array(mx.finfo(result.dtype).min, dtype=result.dtype)
    result = mx.where(
        mx.isnan(result) | mx.isinf(result),
        sentinel,
        result,
    )
    entropy_result = mx.where(
        mx.isnan(entropy_result) | mx.isinf(entropy_result),
        mx.array(0.0, dtype=entropy_result.dtype),
        entropy_result,
    )

    if return_token_entropies:
        return result, entropy_result
    return result


##############################################################################
# Batched generation — mask helpers and core decode loop
##############################################################################


def _create_batched_prefill_mask(
    prompt_lengths: List[int],
    max_prompt_len: int,
) -> mx.array:
    """Create a combined causal + left-padding mask for batched prefill.

    Left-padding places each prompt at the *right* of the ``max_prompt_len``
    window so that the last prompt token always sits at column
    ``max_prompt_len - 1``.  Padding positions (to the left) must be masked
    out so attention never reads them.

    Args:
        prompt_lengths: Per-sequence prompt token counts (Python list).
        max_prompt_len: Width of the padded prompt tensor.

    Returns:
        Bool mask of shape ``[B, 1, max_prompt_len, max_prompt_len]``.
        ``True`` = attend, ``False`` = block.
    """
    if max_prompt_len <= 0:
        raise ValueError("max_prompt_len must be positive")
    if len(prompt_lengths) == 0:
        raise ValueError("prompt_lengths must be non-empty")
    if any(p <= 0 for p in prompt_lengths):
        raise ValueError("All prompt lengths must be >= 1")

    batch_size = len(prompt_lengths)
    positions = mx.arange(max_prompt_len, dtype=mx.int32)

    # Causal mask shared across batch: [1, 1, L, L]
    causal = positions[:, None] >= positions[None, :]
    causal = causal.reshape(1, 1, max_prompt_len, max_prompt_len)

    # Per-sequence key validity mask: [B, 1, 1, L]
    pad_offsets = mx.array(
        [max_prompt_len - pl for pl in prompt_lengths],
        dtype=mx.int32,
    )
    key_valid = positions.reshape(1, 1, 1, max_prompt_len) >= pad_offsets.reshape(
        batch_size, 1, 1, 1
    )
    return mx.logical_and(causal, key_valid)


def _create_batched_decode_mask(
    prompt_lengths: List[int],
    max_prompt_len: int,
    decode_offset: int,
) -> mx.array:
    """Create a decode-step mask that blocks left-padding KV positions.

    During autoregressive decoding the query length is 1 (the newly generated
    token).  The KV length equals ``max_prompt_len + decode_offset`` (all
    cached positions).  We only need to block positions ``k < pad_offset_i``
    which correspond to left-padding tokens.

    Args:
        prompt_lengths: Per-sequence prompt token counts.
        max_prompt_len: Width of the original padded prompt.
        decode_offset: Number of decode steps completed so far (0-indexed).

    Returns:
        Bool mask ``[B, 1, 1, kv_len]`` — ``True`` = attend.
    """
    if max_prompt_len <= 0:
        raise ValueError("max_prompt_len must be positive")
    if decode_offset < 0:
        raise ValueError("decode_offset must be >= 0")
    if len(prompt_lengths) == 0:
        raise ValueError("prompt_lengths must be non-empty")
    if any(p <= 0 for p in prompt_lengths):
        raise ValueError("All prompt lengths must be >= 1")

    batch_size = len(prompt_lengths)
    kv_len = max_prompt_len + decode_offset

    positions = mx.arange(kv_len, dtype=mx.int32)
    pad_offsets = mx.array(
        [max_prompt_len - pl for pl in prompt_lengths],
        dtype=mx.int32,
    )
    return positions.reshape(1, 1, 1, kv_len) >= pad_offsets.reshape(
        batch_size, 1, 1, 1
    )


def _make_prompt_cache_if_available(model: nn.Module) -> Optional[Any]:
    """Best-effort prompt-cache construction across mlx_lm module layouts."""
    if not HAS_MLX_LM:
        return None

    for module_name in ("mlx_lm.cache", "mlx_lm.cache_prompt"):
        try:
            cache_module = importlib.import_module(module_name)
        except Exception:
            continue

        make_prompt_cache = getattr(cache_module, "make_prompt_cache", None)
        if not callable(make_prompt_cache):
            continue

        try:
            return make_prompt_cache(model)
        except Exception:
            return None

    return None


def _model_forward_with_optional_mask_and_cache(
    model: nn.Module,
    input_tokens: mx.array,
    *,
    mask: Optional[mx.array] = None,
    cache_obj: Optional[Any] = None,
) -> Tuple[mx.array, bool]:
    """Forward model with best-effort support for (mask, cache) kwargs."""
    if cache_obj is not None and mask is not None:
        try:
            return model(input_tokens, mask=mask, cache=cache_obj), True
        except TypeError:
            pass
    if cache_obj is not None:
        try:
            return model(input_tokens, cache=cache_obj), True
        except TypeError:
            pass
    if mask is not None:
        try:
            return model(input_tokens, mask=mask), False
        except TypeError:
            pass
    return model(input_tokens), False


def batch_generate_tokens(
    model: nn.Module,
    tokenizer: Any,
    prompt_token_lists: List[mx.array],
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
) -> List[Tuple[mx.array, Dict[str, mx.array]]]:
    """Generate responses for multiple prompts in a single batched pass.

    This bypasses ``stream_generate`` and operates on the model directly,
    amortising weight-load cost across all sequences in the batch — the key
    optimisation for memory-bandwidth-bound Apple Silicon inference.

    Algorithm
    ---------
    1. Left-pad all prompts to ``[N, max_prompt_len]`` with ``pad_token_id``.
    2. Single prefill forward pass with a combined causal + padding mask.
    3. Autoregressive decode loop (up to ``max_tokens`` steps): one forward
       pass per step for all sequences, with per-sequence EOS tracking.
    4. Return per-sequence ``(response_tokens, {'logprob': logprobs})``.

    Logprob convention: computed from *unscaled* logits (before temperature),
    matching the existing ``_simple_generate`` pattern.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer with ``eos_token_id`` and ``pad_token_id``.
        prompt_token_lists: List of N token-ID lists (variable length).
        max_tokens: Maximum new tokens to generate per sequence.
        temperature: Sampling temperature (>0).
        top_p: Nucleus sampling threshold.
        repetition_penalty: Penalty factor for repeating tokens (>1.0
            discourages repetition).  ``None`` or ``1.0`` disables.
        repetition_context_size: Number of recent tokens to consider for
            repetition penalty.  Default ``20`` (matches ``mlx_lm``).

    Returns:
        List of N ``(response_tokens, {'logprob': logprobs})`` tuples.
        ``response_tokens`` includes EOS tokens when sampled; no tokens are
        emitted after EOS for a finished sequence.
    """
    if repetition_penalty is not None and repetition_penalty <= 0:
        raise ValueError(
            f"repetition_penalty must be a positive float, got {repetition_penalty}"
        )

    if max_tokens <= 0:
        return [
            (mx.array([], dtype=mx.int32), {"logprob": mx.array([], dtype=mx.float32)})
            for _ in prompt_token_lists
        ]

    prompts = [mx.array(p, dtype=mx.int32) for p in prompt_token_lists]
    if len(prompts) == 0:
        return []
    for i, prompt in enumerate(prompts):
        if prompt.ndim != 1:
            raise ValueError(
                f"Prompt at index {i} must be 1D tokens, got shape {prompt.shape}."
            )
        if prompt.shape[0] == 0:
            raise ValueError(
                f"Prompt at index {i} is empty. Batched generation requires at least 1 token."
            )

    batch_size = len(prompts)
    pad_id = _resolve_pad_token_id(tokenizer)
    eos_ids = set(_resolve_eos_token_ids(tokenizer))

    # 1) Left-pad prompts for prefill.
    prompt_lengths = [int(p.shape[0]) for p in prompts]
    max_prompt_len = max(prompt_lengths)
    padded_rows: List[mx.array] = []
    prompt_token_lists_py = [p.tolist() for p in prompts]
    for prompt in prompts:
        pad_count = max_prompt_len - int(prompt.shape[0])
        if pad_count > 0:
            row = mx.concatenate(
                [mx.full((pad_count,), pad_id, dtype=mx.int32), prompt]
            )
        else:
            row = prompt
        padded_rows.append(row)
    prompt_batch = mx.stack(padded_rows)  # [B, max_prompt_len]

    # 2) Create prompt cache when available.
    cache_obj = _make_prompt_cache_if_available(model)

    # 3) Prefill pass: one model call for all prompts.
    prefill_mask = _create_batched_prefill_mask(prompt_lengths, max_prompt_len)
    prefill_logits, used_cache = _model_forward_with_optional_mask_and_cache(
        model,
        prompt_batch,
        mask=prefill_mask,
        cache_obj=cache_obj,
    )
    next_logits = prefill_logits[:, max_prompt_len - 1, :]  # [B, vocab]

    # 4) Decode loop (with per-sequence EOS tracking).
    sampler = _make_eos_safe_sampler(temperature, top_p) if temperature > 0 else None
    finished = [False] * batch_size
    per_seq_tokens: List[List[int]] = [[] for _ in range(batch_size)]
    per_seq_logprobs: List[List[float]] = [[] for _ in range(batch_size)]
    arange_batch = mx.arange(batch_size)

    use_rep_penalty = repetition_penalty is not None and repetition_penalty != 1.0
    # Persistent per-sequence context for repetition penalty.  Initialized
    # from the *original* prompt tokens (immutable copy) so the non-cache
    # fallback path — which mutates prompt_token_lists_py — cannot introduce
    # duplicate tokens into the penalty window.
    if use_rep_penalty:
        rep_ctx: List[List[int]] = [list(p) for p in prompt_token_lists_py]

    for decode_step in range(max_tokens):
        # Logprobs from unscaled logits (matches _simple_generate convention).
        # These are recorded for training and are always unpenalized.
        log_probs = next_logits - mx.logsumexp(next_logits, axis=-1, keepdims=True)

        # Apply repetition penalty for sampling only (paper: arXiv 1909.05858).
        # MLX arrays are immutable values so __setitem__ creates new arrays
        # without aliasing next_logits.
        if use_rep_penalty:
            sample_logits = next_logits
            for i in range(batch_size):
                if finished[i]:
                    continue
                ctx = rep_ctx[i][-repetition_context_size:]
                if ctx:
                    tok_ids = mx.array(ctx, dtype=mx.int32)
                    sel = sample_logits[i, tok_ids]
                    sample_logits[i, tok_ids] = mx.where(
                        sel < 0,
                        sel * repetition_penalty,
                        sel / repetition_penalty,
                    )
            sample_log_probs = sample_logits - mx.logsumexp(
                sample_logits, axis=-1, keepdims=True
            )
        else:
            sample_logits = next_logits
            sample_log_probs = log_probs

        if temperature <= 0:
            sampled = mx.argmax(sample_log_probs, axis=-1)
        elif sampler is not None:
            sampled = sampler(sample_log_probs)
        else:
            scaled_logits = sample_logits / temperature
            probs = mx.softmax(scaled_logits, axis=-1)
            sampled = mx.random.categorical(mx.log(probs))

        selected_logprobs = log_probs[arange_batch, sampled]
        sampled_list = sampled.tolist()
        logprob_list = selected_logprobs.tolist()

        for i in range(batch_size):
            if finished[i]:
                continue
            tok = int(sampled_list[i])
            per_seq_tokens[i].append(tok)
            per_seq_logprobs[i].append(float(logprob_list[i]))
            if use_rep_penalty:
                rep_ctx[i].append(tok)
            if tok in eos_ids:
                finished[i] = True

        if all(finished):
            break
        if decode_step == max_tokens - 1:
            break

        # 5) Next-step logits.
        if used_cache:
            # Cache-backed one-token decode.
            feed_tokens = [
                pad_id if finished[i] else int(sampled_list[i])
                for i in range(batch_size)
            ]
            tokens_batch = mx.array(feed_tokens, dtype=mx.int32).reshape(batch_size, 1)
            decode_mask = _create_batched_decode_mask(
                prompt_lengths,
                max_prompt_len,
                decode_offset=decode_step + 1,
            )
            step_logits, _ = _model_forward_with_optional_mask_and_cache(
                model,
                tokens_batch,
                mask=decode_mask,
                cache_obj=cache_obj,
            )
            next_logits = step_logits[:, -1, :]
        else:
            # Fallback for tiny models that do not implement cache:
            # rebuild full sequences and do one batched full forward.
            for i in range(batch_size):
                if not finished[i]:
                    prompt_token_lists_py[i].append(int(sampled_list[i]))

            max_len = max(len(seq) for seq in prompt_token_lists_py)
            rows: List[mx.array] = []
            lengths: List[int] = []
            for seq in prompt_token_lists_py:
                seq_arr = mx.array(seq, dtype=mx.int32)
                lengths.append(len(seq))
                pad_len = max_len - len(seq)
                if pad_len > 0:
                    seq_arr = mx.concatenate(
                        [seq_arr, mx.full((pad_len,), pad_id, dtype=mx.int32)]
                    )
                rows.append(seq_arr)

            full_batch = mx.stack(rows)
            full_logits, _ = _model_forward_with_optional_mask_and_cache(
                model,
                full_batch,
                mask=None,
                cache_obj=None,
            )
            next_logits = full_logits[arange_batch, mx.array(lengths, dtype=mx.int32) - 1, :]

    # 6) Package results.
    results: List[Tuple[mx.array, Dict[str, mx.array]]] = []
    for tokens, logprobs in zip(per_seq_tokens, per_seq_logprobs):
        response_tokens = mx.array(tokens, dtype=mx.int32) if tokens else mx.array([], dtype=mx.int32)
        response_logprobs = mx.array(logprobs, dtype=mx.float32) if logprobs else mx.array([], dtype=mx.float32)
        results.append((response_tokens, {"logprob": response_logprobs}))
    return results


def create_batched_policy(
    model: nn.Module,
    tokenizer: Any,
    generation_params: Optional[Dict[str, Any]] = None,
) -> Callable[[List[mx.array]], List[Tuple[mx.array, Dict[str, Any]]]]:
    """Create a batched policy function for RL rollout collection.

    This is the batched counterpart of :func:`create_policy`.  Instead of
    processing one prompt at a time, it accepts a list of prompt token arrays
    and generates all responses in a single batched forward pass.

    Chat-template application follows the same logic as ``create_policy``
    (lines 558-600): decode → detect if formatting needed → apply template →
    re-encode.

    Args:
        model: Causal language model.
        tokenizer: Tokenizer with chat template support.
        generation_params: Generation parameters (max_tokens, temperature, …).

    Returns:
        Function ``(List[mx.array]) -> List[(response_tokens, info)]``.
    """
    params = generation_params or {}
    max_tokens = params.get("max_tokens", 50)
    temperature = params.get("temperature", 0.8)
    top_p = params.get("top_p", 0.95)
    rep_penalty = params.get("repetition_penalty", None)

    def batched_policy_fn(
        prompt_tokens_list: List[mx.array],
        deterministic: bool = False,
    ) -> List[Tuple[mx.array, Dict[str, Any]]]:
        """Generate responses for all prompts in one batched decode call."""
        processed = [
            _maybe_apply_chat_template(tokenizer, mx.array(pt, dtype=mx.int32))
            for pt in prompt_tokens_list
        ]
        temp = 0.0 if deterministic else temperature
        return batch_generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt_token_lists=processed,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=rep_penalty,
        )

    # Attach metadata so rollout coordination can derive batched policy automatically.
    setattr(batched_policy_fn, "_tp_model", model)
    setattr(batched_policy_fn, "_tp_tokenizer", tokenizer)
    setattr(batched_policy_fn, "_tp_generation_params", dict(params))
    setattr(batched_policy_fn, "_tp_is_batched", True)
    return batched_policy_fn


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
        processed_tokens = _maybe_apply_chat_template(
            tokenizer,
            mx.array(prompt_tokens, dtype=mx.int32),
        )
        
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
    
    # Attach metadata so rollout coordination can derive batched policy automatically.
    setattr(policy_fn, "_tp_model", model)
    setattr(policy_fn, "_tp_tokenizer", tokenizer)
    setattr(policy_fn, "_tp_generation_params", dict(params))
    setattr(policy_fn, "_tp_is_batched", False)
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
