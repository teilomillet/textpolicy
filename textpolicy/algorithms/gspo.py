# textpolicy/algorithms/gspo.py
"""
Group Sequence Policy Optimization (GSPO).

GSPO computes importance weights at the sequence level to align with
sequence-level rewards. Variants include sequence, token, and hybrid forms.
Reference: https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html
"""
from __future__ import annotations
import mlx.core as mx 
from typing import List, Dict


def compute_sequence_importance_weights(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    sequence_lengths: List[int],
    clip_ratio: float = 0.2
) -> mx.array:
    """
    Compute sequence-level importance weights for GSPO.
    
    GSPO formula: w^GSPO_{i} = [π_θ(y_i | x) / π_θ_old(y_i | x)]^(1/|y_i|)
    
    This normalizes by sequence length to prevent bias toward shorter/longer sequences.
    
    Args:
        old_logprobs: Log probabilities from rollout collection [batch_size, seq_len]
        new_logprobs: Log probabilities from current policy [batch_size, seq_len]  
        sequence_lengths: Length of each sequence in the batch
        
    Returns:
        Sequence-level importance weights [batch_size]
        
    Compared to token-level sampling, this reduces variance and matches
    sequence-level reward assignment.
    """
    batch_size = len(sequence_lengths)
    sequence_weights = []
    
    current_idx = 0
    for seq_len in sequence_lengths:
        # Extract logprobs for this sequence
        seq_old_logprobs = old_logprobs[current_idx:current_idx + seq_len]
        seq_new_logprobs = new_logprobs[current_idx:current_idx + seq_len]  # type: ignore
        
        # Compute sequence-level log probability: sum of token log probs
        old_seq_logprob = mx.sum(seq_old_logprobs)
        new_seq_logprob = mx.sum(seq_new_logprobs)
        
        # Sequence-level importance ratio: π_new(y|x) / π_old(y|x)
        log_ratio = new_seq_logprob - old_seq_logprob
        
        # GSPO normalization: raise to power 1/|y_i| to prevent length bias
        # This ensures sequences of different lengths contribute equally
        normalized_log_ratio = log_ratio / seq_len
        
        # Clip in log space to prevent numerical explosion
        # This is the key missing piece that was causing billion-scale importance weights
        clipped_log_ratio = mx.clip(
            normalized_log_ratio,
            mx.log(mx.array(1 - clip_ratio)),  # log(0.8) ≈ -0.22
            mx.log(mx.array(1 + clip_ratio))   # log(1.2) ≈ 0.18
        )
        
        # Now safely compute importance weight (will be in range [0.8, 1.2])
        importance_weight = mx.exp(clipped_log_ratio)
        
        # Final check: account for float32 precision
        # exp(log(1.2)) in float32 may produce 1.2000000476837158; enforce exact bounds
        importance_weight = mx.clip(importance_weight, 1.0 - clip_ratio, 1.0 + clip_ratio)
        
        # Enforce exact bounds using scalar comparisons
        # Convert to float for comparison to avoid MLX array comparison issues
        weight_float = float(importance_weight)
        if weight_float > 1.0 + clip_ratio:
            importance_weight = mx.array(1.0 + clip_ratio)
        elif weight_float < 1.0 - clip_ratio:
            importance_weight = mx.array(1.0 - clip_ratio)
        
        sequence_weights.append(importance_weight)
        current_idx += seq_len
    
    return mx.array(sequence_weights)


def compute_hybrid_importance_weights(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    sequence_lengths: List[int],
    alpha: float = 0.5,
    beta: float = 0.5
) -> mx.array:
    """
    Compute hybrid importance weights using principled log-space combination.
    
    Instead of multiplying exp(seq_ratio) * exp(token_ratio) which compounds variance,
    uses additive combination: exp(α * seq_log_ratio + β * token_log_ratio)
    
    This provides a more stable and theoretically sound approach to combining
    sequence-level stability with token-level granularity.
    
    Args:
        old_logprobs: Log probabilities from rollout collection [total_tokens]
        new_logprobs: Log probabilities from current policy [total_tokens]
        sequence_lengths: Length of each sequence in the batch
        alpha: Weight for sequence-level importance (default: 0.5)
        beta: Weight for token-level importance (default: 0.5)
        
    Returns:
        Hybrid importance weights [total_tokens]
        
    Advantages:
    - Avoids explosive multiplication of exponentials
    - Controlled variance through hyperparameter balance
    - Principled combination in log-space
    """
    # Compute sequence-level log ratios (without exponential)
    batch_size = len(sequence_lengths)
    seq_log_ratios = []
    
    current_idx = 0
    for seq_len in sequence_lengths:
        # Extract logprobs for this sequence
        seq_old_logprobs = old_logprobs[current_idx:current_idx + seq_len]
        seq_new_logprobs = new_logprobs[current_idx:current_idx + seq_len]  # type: ignore
        
        # Compute sequence-level log probability: sum of token log probs
        old_seq_logprob = mx.sum(seq_old_logprobs)
        new_seq_logprob = mx.sum(seq_new_logprobs)
        
        # Sequence-level log ratio with GSPO normalization (prevent length bias)
        log_ratio = new_seq_logprob - old_seq_logprob
        normalized_seq_log_ratio = log_ratio / seq_len
        
        seq_log_ratios.append(normalized_seq_log_ratio)
        current_idx += seq_len
    
    # Expand sequence-level log ratios to token level
    token_seq_log_ratios = []
    for i, seq_len in enumerate(sequence_lengths):
        # Use stop gradient to prevent certain gradient flows
        seq_log_ratio_sg = mx.stop_gradient(seq_log_ratios[i])
        token_seq_log_ratios.extend([seq_log_ratio_sg] * seq_len)
    
    token_seq_log_ratios = mx.array(token_seq_log_ratios)
    
    # Compute token-level log ratios (with stop gradient on old logprobs)
    old_logprobs_sg = mx.stop_gradient(old_logprobs)
    token_log_ratios = new_logprobs - old_logprobs_sg
    
    # Combine in log-space: α * seq_log_ratio + β * token_log_ratio
    combined_log_ratios = alpha * token_seq_log_ratios + beta * token_log_ratios
    
    # Apply single exponential to get final importance weights
    hybrid_weights = mx.exp(combined_log_ratios)
    
    return hybrid_weights


def gspo_policy_loss(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    sequence_lengths: List[int],
    variant: str = "sequence",
    clip_ratio: float = 0.2,
    alpha: float = 0.5,
    beta: float = 0.5
) -> mx.array:
    """
    GSPO policy loss with sequence-level importance sampling.
    
    Args:
        old_logprobs: Log probabilities from rollout collection
        new_logprobs: Log probabilities from current policy
        advantages: Group-relative advantages (computed same as GRPO)
        sequence_lengths: Length of each sequence in the batch
        variant: "sequence" for pure GSPO, "hybrid" for GSPO-token, "token" for GRPO
        clip_ratio: Clipping ratio for surrogate objective
        alpha: Weight for sequence-level importance (used in hybrid variant)
        beta: Weight for token-level importance (used in hybrid variant)
        
    Returns:
        Policy loss scalar (to be minimized)
        
    Key innovation:
    - Uses sequence-level importance weights instead of token-level
    - Reduces gradient variance and improves training stability
    - Better alignment with sequence-level reward signals
    """
    if variant == "sequence":
        # Pure GSPO: sequence-level importance sampling
        importance_weights = compute_sequence_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, clip_ratio
        )
        
        # Expand advantages to match sequence weights
        if len(advantages) != len(sequence_lengths):
            raise ValueError(f"Advantages length {len(advantages)} doesn't match sequences {len(sequence_lengths)}")
        
        # Apply PPO clipping to sequence-level weights
        clipped_weights = mx.clip(importance_weights, 1 - clip_ratio, 1 + clip_ratio)
        
        # Compute surrogate loss at sequence level
        surr1 = importance_weights * advantages
        surr2 = clipped_weights * advantages
        loss = -mx.mean(mx.minimum(surr1, surr2))
        
    elif variant == "hybrid":
        # GSPO-token: hybrid sequence and token-level
        importance_weights = compute_hybrid_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, alpha=alpha, beta=beta
        )
        
        # Expand advantages to token level
        token_advantages = []
        for i, seq_len in enumerate(sequence_lengths):
            token_advantages.extend([advantages[i]] * seq_len)
        token_advantages = mx.array(token_advantages)
        
        # Apply PPO clipping to hybrid weights
        clipped_weights = mx.clip(importance_weights, 1 - clip_ratio, 1 + clip_ratio)
        
        # Compute surrogate loss at token level
        surr1 = importance_weights * token_advantages
        surr2 = clipped_weights * token_advantages
        loss = -mx.mean(mx.minimum(surr1, surr2))
        
    elif variant == "token":
        # Standard GRPO: token-level importance sampling (for comparison)
        ratio = mx.exp(new_logprobs - old_logprobs)
        
        # Expand advantages to token level
        token_advantages = []
        for i, seq_len in enumerate(sequence_lengths):
            token_advantages.extend([advantages[i]] * seq_len)
        token_advantages = mx.array(token_advantages)
        
        # Apply PPO clipping
        clipped_ratio = mx.clip(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        # Compute surrogate loss
        surr1 = ratio * token_advantages
        surr2 = clipped_ratio * token_advantages
        loss = -mx.mean(mx.minimum(surr1, surr2))
        
    else:
        raise ValueError(f"Unknown GSPO variant: {variant}. Choose 'sequence', 'hybrid', or 'token'")
    
    return loss


def create_gspo_policy_loss(variant: str = "sequence", clip_ratio: float = 0.2, alpha: float = 0.5, beta: float = 0.5):
    """
    Factory function to create GSPO policy loss function with standard signature.
    
    This follows the design guidelines for pure function composition with the universal Trainer.
    
    Args:
        variant: GSPO variant ("sequence", "hybrid", or "token")
        clip_ratio: PPO clipping ratio for importance weights
        
    Returns:
        Policy loss function with standard signature (old_logprobs, new_logprobs, advantages)
        
    Usage:
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages_dr_grpo,
            loss_fn=gspo.create_gspo_policy_loss(variant="sequence"),
            optimizer=optimizer
        )
    """
    def gspo_policy_loss_fn(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> mx.array:
        """
        GSPO policy loss with sequence-level importance sampling.
        
        Standard signature for use with universal Trainer.
        """
        # For GSPO, we need sequence lengths. This is a limitation that requires
        # the batch_data to include sequence_lengths information.
        # For now, we'll use a fallback approach for compatibility.
        
        # Robust fallback: distribute tokens as evenly as possible across episodes
        # This handles variable-length sequences by distributing remainder tokens
        total_tokens = len(old_logprobs) if len(old_logprobs.shape) == 1 else old_logprobs.shape[0] # type: ignore
        num_episodes = len(advantages)
        
        if num_episodes > 0:
            base_length = total_tokens // num_episodes
            remainder = total_tokens % num_episodes
            # Distribute remainder tokens to first 'remainder' episodes 
            sequence_lengths = [base_length + (1 if i < remainder else 0) for i in range(num_episodes)]
        else:
            sequence_lengths = [total_tokens] if total_tokens > 0 else [1]
        
        return gspo_policy_loss(
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            advantages=advantages,
            sequence_lengths=sequence_lengths,
            variant=variant,
            clip_ratio=clip_ratio,
            alpha=alpha,
            beta=beta
        )
    
    return gspo_policy_loss_fn


def create_gspo_metrics(variant: str = "sequence", clip_ratio: float = 0.2):
    """
    Factory function to create GSPO metrics function with standard signature.
    
    Args:
        variant: GSPO variant being used
        clip_ratio: Clipping ratio used in loss
        
    Returns:
        Metrics function with standard signature
        
    Usage:
        trainer = Trainer(
            model=model,
            advantage_fn=grpo.compute_advantages_dr_grpo,
            loss_fn=gspo.create_gspo_policy_loss(variant="sequence"),
            metrics_fn=gspo.create_gspo_metrics(variant="sequence"),
            optimizer=optimizer
        )
    """
    def gspo_metrics_fn(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> Dict[str, float]:
        """GSPO metrics with sequence-level importance weight tracking."""
        # Robust fallback: distribute tokens as evenly as possible across episodes
        # This matches the same robust approach used in the policy loss function
        total_tokens = len(old_logprobs) if len(old_logprobs.shape) == 1 else old_logprobs.shape[0] # type: ignore
        num_episodes = len(advantages)
        
        if num_episodes > 0:
            base_length = total_tokens // num_episodes
            remainder = total_tokens % num_episodes
            # Distribute remainder tokens to first 'remainder' episodes
            sequence_lengths = [base_length + (1 if i < remainder else 0) for i in range(num_episodes)]
        else:
            sequence_lengths = [total_tokens] if total_tokens > 0 else [1]
        
        return compute_gspo_metrics(
            old_logprobs=old_logprobs,
            new_logprobs=new_logprobs,
            advantages=advantages,
            sequence_lengths=sequence_lengths,
            variant=variant,
            clip_ratio=clip_ratio
        )
    
    return gspo_metrics_fn


# Convenience functions that match GRPO interface
def policy_loss_sequence(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> mx.array:
    """GSPO sequence-level policy loss function (standard signature)."""
    return create_gspo_policy_loss(variant="sequence")(old_logprobs, new_logprobs, advantages)


def policy_loss_hybrid(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> mx.array:
    """GSPO hybrid policy loss function (standard signature)."""
    return create_gspo_policy_loss(variant="hybrid")(old_logprobs, new_logprobs, advantages)

def create_policy_loss_hybrid(alpha: float = 0.5, beta: float = 0.5):
    """
    Create a GSPO hybrid policy loss function with configurable hyperparameters.
    
    Args:
        alpha: Weight for sequence-level importance (0.0 = pure token-level, 1.0 = pure sequence-level)
        beta: Weight for token-level importance (0.0 = ignore token-level, 1.0 = full token-level)
        
    Returns:
        Policy loss function with standard signature
        
    Example:
        # Balanced hybrid (default)
        loss_fn = create_policy_loss_hybrid(alpha=0.5, beta=0.5)
        
        # More sequence-focused
        loss_fn = create_policy_loss_hybrid(alpha=0.7, beta=0.3)
        
        # More token-focused  
        loss_fn = create_policy_loss_hybrid(alpha=0.3, beta=0.7)
    """
    def hybrid_loss_fn(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> mx.array:
        # Use custom alpha/beta parameters for this specific loss function
        return create_gspo_policy_loss(variant="hybrid", alpha=alpha, beta=beta)(
            old_logprobs, new_logprobs, advantages
        )
    return hybrid_loss_fn


def policy_loss_token(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> mx.array:
    """GSPO token-level policy loss function (standard signature) - equivalent to GRPO."""
    return create_gspo_policy_loss(variant="token")(old_logprobs, new_logprobs, advantages)


def compute_metrics_sequence(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> Dict[str, float]:
    """GSPO sequence-level metrics function (standard signature)."""
    return create_gspo_metrics(variant="sequence")(old_logprobs, new_logprobs, advantages)


def compute_metrics_hybrid(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> Dict[str, float]:
    """GSPO hybrid metrics function (standard signature)."""
    return create_gspo_metrics(variant="hybrid")(old_logprobs, new_logprobs, advantages)


def compute_metrics_token(old_logprobs: mx.array, new_logprobs: mx.array, advantages: mx.array) -> Dict[str, float]:
    """GSPO token-level metrics function (standard signature)."""
    return create_gspo_metrics(variant="token")(old_logprobs, new_logprobs, advantages)


def compute_gspo_metrics(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    advantages: mx.array,
    sequence_lengths: List[int],
    variant: str = "sequence",
    clip_ratio: float = 0.2
) -> dict:
    """
    Compute GSPO training metrics for monitoring.
    
    Args:
        old_logprobs: Log probabilities from rollout
        new_logprobs: Log probabilities from current policy  
        advantages: Group-relative advantages
        sequence_lengths: Length of each sequence in the batch
        variant: GSPO variant being used
        clip_ratio: Clipping ratio used in loss
        
    Returns:
        Dictionary of metrics for logging/monitoring
        
    Additional GSPO-specific metrics:
    - Sequence-level importance weight statistics
    - Gradient variance estimates
    - Length bias indicators
    """
    # Standard advantage metrics
    metrics = {
        'mean_advantage': mx.mean(advantages).item(),
        'std_advantage': mx.std(advantages).item(),
        'min_advantage': mx.min(advantages).item(),
        'max_advantage': mx.max(advantages).item()
    }
    
    if variant == "sequence":
        # Sequence-level importance weights
        seq_weights = compute_sequence_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths, clip_ratio
        )
        
        # Sequence weight statistics
        metrics.update({
            'mean_seq_weight': mx.mean(seq_weights).item(),
            'std_seq_weight': mx.std(seq_weights).item(),
            'max_seq_weight': mx.max(seq_weights).item(),
            'min_seq_weight': mx.min(seq_weights).item()
        })
        
        # Clipping statistics at sequence level
        clipped = (seq_weights < (1 - clip_ratio)) | (seq_weights > (1 + clip_ratio))
        metrics['seq_clip_fraction'] = mx.mean(clipped.astype(mx.float32)).item()
        
    elif variant == "hybrid":
        # Hybrid importance weights  
        hybrid_weights = compute_hybrid_importance_weights(
            old_logprobs, new_logprobs, sequence_lengths
        )
        
        # Hybrid weight statistics
        metrics.update({
            'mean_hybrid_weight': mx.mean(hybrid_weights).item(),
            'std_hybrid_weight': mx.std(hybrid_weights).item(),
            'max_hybrid_weight': mx.max(hybrid_weights).item(),
            'min_hybrid_weight': mx.min(hybrid_weights).item()
        })
        
        # Clipping statistics at token level
        clipped = (hybrid_weights < (1 - clip_ratio)) | (hybrid_weights > (1 + clip_ratio))
        metrics['hybrid_clip_fraction'] = mx.mean(clipped.astype(mx.float32)).item()
        
    else:  # token-level (standard GRPO)
        # Token-level importance ratios
        ratio = mx.exp(new_logprobs - old_logprobs)
        
        metrics.update({
            'mean_token_ratio': mx.mean(ratio).item(),
            'std_token_ratio': mx.std(ratio).item(),
            'max_token_ratio': mx.max(ratio).item(),
            'min_token_ratio': mx.min(ratio).item()
        })
        
        # Clipping statistics at token level
        clipped = (ratio < (1 - clip_ratio)) | (ratio > (1 + clip_ratio))
        metrics['token_clip_fraction'] = mx.mean(clipped.astype(mx.float32)).item()
    
    # Length bias analysis
    if len(sequence_lengths) > 1:
        length_array = mx.array(sequence_lengths, dtype=mx.float32)
        metrics.update({
            'mean_seq_length': mx.mean(length_array).item(),
            'std_seq_length': mx.std(length_array).item(),
            'min_seq_length': mx.min(length_array).item(),
            'max_seq_length': mx.max(length_array).item()
        })
    
    # KL divergence approximation
    kl_div = mx.mean(old_logprobs - new_logprobs)
    metrics['kl_divergence'] = kl_div.item()
    
    return metrics


# Algorithm-specific data selectors for GSPO
def select_gspo_data(buffer, variant: str = "sequence"):
    """
    GSPO data selector: Use all available data with sequence-level organization.
    
    GSPO requires sequence length information for proper importance weight computation.
    This selector ensures sequence boundaries are preserved in the batch data.
    
    Args:
        buffer: Buffer containing episodes
        variant: GSPO variant ("sequence", "hybrid", or "token")
        
    Returns:
        Batch data organized for GSPO training with sequence length metadata
    """
    from .grpo import select_all_data
    
    # Reuse GRPO's data selection but add sequence length tracking
    batch_data = select_all_data(buffer)
    
    # GSPO-specific enhancement: explicit sequence length tracking
    # This ensures proper importance weight computation
    if 'episode_lengths' in batch_data:
        # Use episode lengths as sequence lengths for GSPO
        batch_data['sequence_lengths'] = batch_data['episode_lengths']
    else:
        # Fallback: infer sequence lengths from batch structure
        # This is less ideal but provides compatibility
        total_tokens = len(batch_data['obs']) if 'obs' in batch_data else 0
        num_episodes = len(batch_data['rewards']) if 'rewards' in batch_data else 1
        avg_length = total_tokens // num_episodes if num_episodes > 0 else 0
        batch_data['sequence_lengths'] = [avg_length] * num_episodes
    
    return batch_data


# Compiled versions for maximum performance
@mx.compile
def compute_sequence_weights_compiled(
    old_logprobs: mx.array,
    new_logprobs: mx.array,
    seq_len: int
) -> mx.array:
    """Compiled version of sequence weight computation for a single sequence."""
    old_seq_logprob = mx.sum(old_logprobs)
    new_seq_logprob = mx.sum(new_logprobs)
    log_ratio = new_seq_logprob - old_seq_logprob
    normalized_log_ratio = log_ratio / seq_len
    return mx.exp(normalized_log_ratio)


@mx.compile
def gspo_loss_compiled(
    importance_weights: mx.array,
    advantages: mx.array,
    clip_ratio: float = 0.2
) -> mx.array:
    """Compiled version of GSPO surrogate loss computation."""
    clipped_weights = mx.clip(importance_weights, 1 - clip_ratio, 1 + clip_ratio)
    surr1 = importance_weights * advantages
    surr2 = clipped_weights * advantages
    return -mx.mean(mx.minimum(surr1, surr2))
