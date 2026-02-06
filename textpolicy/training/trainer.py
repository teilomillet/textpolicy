# textpolicy/training/trainer.py
"""
Unified Trainer for all RL algorithms — designed for MLX and Apple Silicon.

This trainer achieves maximum efficiency through:
- Pure function composition (zero abstraction cost)
- Single training loop for all algorithms
- MLX compilation optimization
- Apple Silicon unified memory patterns
- Direct MLX-LM integration
"""

import logging
from typing import Callable, Dict, Any, Optional, Union, List, cast
import mlx.core as mx # type: ignore
import mlx.nn as nn # type: ignore
import mlx.optimizers as optim # type: ignore
from textpolicy.buffer import Buffer
from textpolicy.rollout import RolloutCoordinator
from textpolicy.utils.timing import Timer
from .metrics import TrainingMetrics


class Trainer:
    """
    Universal trainer that composes pure algorithm functions.
    
    Key design principles:
    - Algorithm-agnostic: Works with any advantage_fn + loss_fn combination
    - MLX-optimized: Direct function calls, perfect for @mx.compile
    - Memory efficient: Minimal allocations, reuses buffers
    - Composable: User picks exactly what they need
    
    Usage:
        from textpolicy.algorithms import grpo
        trainer = Trainer(
            model=mlx_model,
            advantage_fn=grpo.compute_advantages,
            loss_fn=grpo.policy_loss,
            optimizer=optimizer
        )
    """
    
    def __init__(
        self,
        model: nn.Module,
        advantage_fn: Callable,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        get_logprobs_fn: Optional[Callable] = None,
        metrics_fn: Optional[Callable] = None,
        max_grad_norm: Optional[float] = 0.5,
        compile_training: bool = True,
        buffer: Optional[Buffer] = None,
        data_selector_fn: Optional[Callable] = None,
        auto_save_lora: Optional[str] = None,
        metrics_interval: int = 10,
        advantage_transform_fn: Optional[Callable] = None,
        profile: bool = False
    ):
        """
        Initialize unified trainer with composable algorithm functions.

        Args:
            model: MLX model (typically from MLX-LM)
            advantage_fn: Pure function for computing advantages
            loss_fn: Pure function for computing policy loss
            optimizer: MLX optimizer (Adam, AdamW, etc.)
            get_logprobs_fn: Function to extract logprobs from model output
            metrics_fn: Function to compute training metrics
            max_grad_norm: Maximum gradient norm for clipping (None disables)
            compile_training: Whether to compile training step with @mx.compile
            buffer: Optional linked buffer for automatic data selection
            data_selector_fn: Algorithm-specific function to select data from buffer
            auto_save_lora: Optional path to auto-save LoRA adapters after training
            metrics_interval: Compute detailed metrics every N steps. Setting >1
                avoids a duplicate model forward pass on non-metric steps.
                Default 10 balances insight and throughput; set to 1 for
                every-step metrics when needed.
            advantage_transform_fn: Optional function to transform token-level
                advantages after expansion. Signature:
                ``(advantages: mx.array, batch_data: Dict) -> mx.array``.
                Used by HICRA to amplify planning tokens. None means no-op.
            profile: When True, insert ``mx.eval()`` barriers between training
                phases and record per-phase wall-clock times in the metrics dict
                returned by ``train()``.  Zero cost when False (single boolean
                check per phase).
        """
        self.model = model
        self.advantage_fn = advantage_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.get_logprobs_fn = get_logprobs_fn or self._default_get_logprobs
        self.metrics_fn = metrics_fn
        self.max_grad_norm = max_grad_norm
        self.metrics_interval = max(1, metrics_interval)

        self.advantage_transform_fn = advantage_transform_fn

        # Profiling: Timer is only allocated when profile=True.
        # When _timer is None every phase-timing check is a single ``if``.
        self._timer: Optional[Timer] = Timer() if profile else None

        # Buffer management
        self.buffer = buffer
        self.data_selector_fn = data_selector_fn or self._default_data_selector

        # LoRA management - detect auto-reload models
        self.auto_save_lora = auto_save_lora or self._detect_auto_reload_lora(model)
        self._has_lora = self._detect_lora_model(model)
        
        # Create compiled loss function for maximum performance
        if compile_training:
            self.loss_and_grad_fn = mx.compile(nn.value_and_grad(model, self._loss_fn))
        else:
            self.loss_and_grad_fn = nn.value_and_grad(model, self._loss_fn)
        
        # Training state
        self.metrics = TrainingMetrics()
        self._step_count = 0
    
    def _detect_lora_model(self, model: nn.Module) -> bool:
        """
        Pure function to detect if model has LoRA adapters.
        
        Args:
            model: MLX model to check
            
        Returns:
            True if model has LoRA parameters
        """
        try:
            # Try named_parameters first (for compatibility)
            if hasattr(model, 'named_parameters'):
                for name, param in model.named_parameters():
                    if 'lora_' in name.lower() and hasattr(param, 'requires_grad') and param.requires_grad:
                        return True
            
            # Fallback: check for LoRA layers in the model structure
            if hasattr(model, 'layers') or hasattr(model, 'model'):
                # This is a heuristic check for LoRA
                model_str = str(model)
                return 'lora' in model_str.lower()
                
        except Exception:
            # If inspection fails, assume no LoRA
            pass
        
        return False
    
    def _detect_auto_reload_lora(self, model: nn.Module) -> Optional[str]:
        """
        Pure function to detect if model was created with auto-reload LoRA.
        
        This is how we implement the implicit behavior - LoRA models
        created with create_lora_setup(auto_reload=True) are automatically
        detected and managed by the Trainer.
        
        Args:
            model: MLX model to check
            
        Returns:
            Path for auto-saving adapters, or None if not auto-reload model
        """
        if hasattr(model, '_is_auto_reload_lora') and model._is_auto_reload_lora:
            return getattr(model, '_auto_reload_path', None)
        return None
    
    def _save_lora_if_enabled(self):
        """
        Pure function to save LoRA adapters if auto-save is enabled.
        
        This is called automatically after each training step.
        Invisible to the user - no complex reload management needed.
        """
        if not self.auto_save_lora or not self._has_lora:
            return
            
        try:
            # Extract and save only LoRA parameters
            lora_params = {}
            for name, param in self.model.named_parameters():
                if 'lora_' in name.lower() and param.requires_grad:
                    lora_params[name] = param
            
            if lora_params:
                mx.save_safetensors(self.auto_save_lora, lora_params)
                logging.getLogger(__name__).info(
                    "✓ Auto-saved LoRA adapters to %s", self.auto_save_lora
                )
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Auto-save LoRA failed: %s", e
            )
    
    def _default_get_logprobs(self, model_output: Any, actions: mx.array) -> mx.array:
        """
        Default function to extract log probabilities from model output.
        
        This function extracts log probabilities for RL training.
        Correctness is required for policy gradient algorithms.
        
        Args:
            model_output: Raw logits from model forward pass [batch_size, seq_len, vocab_size]
            actions: Action tokens to evaluate [batch_size, seq_len] or [seq_len]
            
        Returns:
            Log probabilities of the actions [batch_size, seq_len] or [seq_len]
        """
        # Extract logits from model output
        if hasattr(model_output, 'logits'):
            logits = model_output.logits
        else:
            logits = model_output
        
        # Validate logits shape
        if logits.ndim < 2:
            raise ValueError(f"Expected logits with at least 2 dimensions, got {logits.ndim}")
        
        # Compute log probabilities with numerical stability
        # log_softmax(x) = x - logsumexp(x) is more stable than log(softmax(x))
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        
        # Extract log probabilities for specific actions
        if actions.ndim == 1:
            # Single sequence case: [seq_len]
            if log_probs.ndim == 3:
                # Remove batch dimension if present: [1, seq_len, vocab_size] -> [seq_len, vocab_size]
                log_probs = log_probs[0]
            
            # Validate sequence length alignment using MLX's size property
            # MLX arrays have .size property which is type-checker friendly
            actions_len = actions.size
            if log_probs.shape[0] != actions_len:
                raise ValueError(
                    f"Sequence length mismatch: logits have {log_probs.shape[0]} positions "
                    f"but actions have {actions_len} tokens"
                )
            
            # Extract logprobs for actions: [seq_len]
            action_indices = mx.arange(actions_len)
            action_log_probs = log_probs[action_indices, actions]
            
        elif actions.ndim == 2:
            # Batch case: [batch_size, seq_len]
            # MLX shape type annotation is incorrect (object instead of tuple), use type: ignore
            batch_size = actions.shape[0]  # type: ignore
            seq_len = actions.shape[1]  # type: ignore
            
            # Validate batch alignment
            if log_probs.shape[0] != batch_size or log_probs.shape[1] != seq_len:
                raise ValueError(
                    f"Batch shape mismatch: logits shape {log_probs.shape[:2]} "
                    f"vs actions shape {actions.shape}"
                )
            
            # Extract logprobs for actions: [batch_size, seq_len]
            batch_indices = mx.arange(batch_size)[:, None]
            seq_indices = mx.arange(seq_len)[None, :]
            action_log_probs = log_probs[batch_indices, seq_indices, actions]
            
        else:
            raise ValueError(f"Unsupported actions dimension: {actions.ndim}")
        
        # VALIDATION: Check for reasonable values
        if mx.any(mx.isnan(action_log_probs)) or mx.any(mx.isinf(action_log_probs)):
            raise ValueError("NaN or Inf values in computed logprobs")
        
        return action_log_probs
    
    def _default_data_selector(self, buffer: Buffer) -> Dict[str, mx.array]:
        """
        Default data selection strategy - use all available data.
        
        This can be overridden with algorithm-specific selectors that might:
        - Sample only recent episodes for on-policy algorithms
        - Select episodes based on reward thresholds
        - Apply importance sampling weights
        - Filter by episode length or other criteria
        
        Args:
            buffer: Buffer containing episodes
            
        Returns:
            Selected batch data for training
        """
        return self._prepare_batch_from_buffer(buffer)
    
    def _loss_fn(self, batch_data: Dict[str, mx.array]) -> mx.array:
        """
        Internal loss function for nn.value_and_grad.
        
        This function orchestrates the algorithm-specific components:
        1. Model forward pass
        2. Extract new log probabilities  
        3. Compute advantages using advantage_fn
        4. Compute loss using loss_fn
        
        Args:
            batch_data: Batch data with obs, act, logprob, rewards
            
        Returns:
            Algorithm loss (GRPO, PPO, etc. depending on functions provided)
        """
        observations = batch_data['obs']
        actions = batch_data['act']  # Actions taken during rollout
        old_logprobs = batch_data['logprob']
        rewards = batch_data['rewards']
        
        # For proper logprob extraction, we need the full context (prompt + response)
        # The model needs to see the full sequence to generate logits for all response positions
        # This matches how the model was called during rollout generation
        
        # Forward pass through model to get new logprobs
        # Use the default logprob extraction which works directly with the batch structure
        # This avoids complex prompt/response splitting and matches the old_logprobs format
        
        # The key insight: observations contain concatenated prompt+response sequences
        # actions contain the response portions that need logprob evaluation
        # old_logprobs has the exact shape we need to match
        
        # GRPO-specific logprob extraction: observations contain prompt+response,
        # actions contain only response tokens that need logprob evaluation.
        if 'episode_lengths' in batch_data:
            episode_lengths = batch_data['episode_lengths']
            prompt_lengths = batch_data.get('prompt_lengths')
            new_logprobs = self._extract_grpo_logprobs(observations, actions, old_logprobs, episode_lengths, prompt_lengths)
        else:
            # Fallback for non-GRPO data or custom pipelines
            if observations.ndim == 1:
                model_input = observations[None]  # Add batch dimension: [1, seq_len]
            else:
                model_input = observations  # Already batched: [batch_size, seq_len]

            model_output = self.model(model_input)
            new_logprobs = self.get_logprobs_fn(model_output, actions)
        
        # Compute advantages using algorithm-specific function
        advantages = self.advantage_fn(rewards)
        
        # Handle advantage expansion for sequence-level algorithms
        # Check if advantages (episode-level) need expansion to match logprobs (token-level)
        # GSPO uses sequence-level advantages; do not expand to token level
        needs_sequence_level = (
            hasattr(self.loss_fn, '__name__') and 'gspo' in self.loss_fn.__name__.lower()
        ) or (
            hasattr(self.loss_fn, '__qualname__') and 'gspo' in self.loss_fn.__qualname__.lower()  
        )
        
        if advantages.shape[0] != new_logprobs.shape[0] and not needs_sequence_level:  # type: ignore
            # Expand episode-level advantages to token-level for token-based algorithms (GRPO, PPO)
            #
            # GRPO: advantages [episodes] → [total_tokens] for token-level importance sampling
            # GSPO: advantages stay [episodes] for sequence-level importance sampling (handled above)
            num_episodes = advantages.shape[0]  # type: ignore
            total_tokens = new_logprobs.shape[0]  # type: ignore

            # Use real episode lengths when available (the standard path via train()).
            # Fall back to even distribution only when episode_lengths is missing
            # (e.g. direct _loss_fn calls from tests or custom pipelines).
            if 'episode_lengths' in batch_data:
                action_lengths = batch_data['episode_lengths']
            else:
                base_length = total_tokens // num_episodes
                remainder = total_tokens % num_episodes
                action_lengths = [base_length + (1 if i < remainder else 0) for i in range(num_episodes)]

            # Validate episode_lengths consistency before expanding.
            if len(action_lengths) != num_episodes:
                raise ValueError(
                    f"len(episode_lengths)={len(action_lengths)} does not match "
                    f"num_episodes={num_episodes}. batch_data['episode_lengths'] "
                    f"must have one entry per episode."
                )
            if sum(action_lengths) != total_tokens:
                raise ValueError(
                    f"sum(episode_lengths)={sum(action_lengths)} does not match "
                    f"total_tokens={total_tokens}. Episode lengths must sum to "
                    f"the total number of tokens in the batch."
                )

            advantages = self._expand_advantages(advantages, action_lengths)

            if getattr(self, '_debug_logging', False):
                logging.getLogger(__name__).debug(
                    "Expansion successful: final shape = %d tokens", advantages.shape[0]
                )

        # Apply optional advantage transform (e.g. HICRA planning token amplification)
        if self.advantage_transform_fn is not None:
            expected_shape = advantages.shape
            advantages = self.advantage_transform_fn(advantages, batch_data)
            if advantages.shape != expected_shape:
                raise ValueError(
                    f"advantage_transform_fn changed shape from {expected_shape} "
                    f"to {advantages.shape}. The transform must return advantages "
                    f"with the same shape as its input."
                )

        # Compute loss using algorithm-specific function
        loss = self.loss_fn(old_logprobs, new_logprobs, advantages)
        
        return loss
    
    def _extract_grpo_logprobs(
        self,
        observations: mx.array,
        actions: mx.array,
        old_logprobs: mx.array,
        episode_lengths: List[int],
        prompt_lengths: Optional[List[int]] = None,
    ) -> mx.array:
        """
        Compute per-episode logprobs under the current model.

        Three paths, in priority order:

        1. **Batched** (default for text generation): 2D obs + prompt_lengths
           → single ``compute_logprobs_batched`` call (one model forward pass).
        2. **Sequential compat**: 2D obs, no prompt_lengths → N per-episode
           ``compute_logprobs`` calls (legacy path).
        3. **Flat 1D fallback**: multi-step RL → single forward pass through
           ``get_logprobs_fn``.

        Args:
            observations: Episode observations — 2D ``[N, max_obs_len]``
                for text generation, or flat 1D for multi-step RL.
            actions: Episode actions — 2D ``[N, max_act_len]``
                for text generation, or flat 1D for multi-step RL.
            old_logprobs: Reference logprobs (flat 1D) for shape validation.
            episode_lengths: Per-episode response token counts.
            prompt_lengths: Per-episode prompt token counts. When provided
                with 2D observations, enables the batched path.

        Returns:
            Flat 1D log probabilities matching ``old_logprobs`` shape.
        """
        num_episodes = len(episode_lengths)

        if observations.ndim == 2 and actions.ndim == 2:
            # Path 1: Batched — single model forward pass for all episodes
            if prompt_lengths is not None:
                if len(prompt_lengths) != num_episodes:
                    raise ValueError(
                        f"len(prompt_lengths)={len(prompt_lengths)} does not "
                        f"match num_episodes={num_episodes} from episode_lengths."
                    )

                from textpolicy.generation.mlx_generation import compute_logprobs_batched

                return compute_logprobs_batched(
                    self.model,
                    observations,   # [N, max_obs_len]
                    actions,        # [N, max_act_len]
                    prompt_lengths,
                    episode_lengths,
                )

            # Path 2: Sequential compat — N per-episode forward passes
            from textpolicy.generation.mlx_generation import compute_logprobs

            per_episode = []
            for i in range(num_episodes):
                ep_logprobs = compute_logprobs(
                    self.model, observations[i], actions[i]
                )
                per_episode.append(ep_logprobs)
            return mx.concatenate(per_episode)

        # Path 3: Flat 1D — multi-step RL generic path.
        if observations.ndim == 1:
            model_input = observations[None]
        else:
            model_input = observations
        model_output = self.model(model_input)
        return self.get_logprobs_fn(model_output, actions)

    def _expand_advantages(self, advantages: mx.array, episode_lengths: List[int]) -> mx.array:
        """
        Expand episode-level advantages to token-level for sequence models.
        
        Avoids .item() calls and uses MLX operations to maintain device efficiency.
        
        Args:
            advantages: Episode-level advantages [num_episodes]
            episode_lengths: Length of each episode
            
        Returns:
            Token-level advantages [total_tokens]
        """
        # Use repeat operation for efficient expansion without .item() bottlenecks
        # This keeps everything on GPU and avoids synchronization overhead
        
        # For uniform episode lengths (common case), use vectorized operations
        if len(set(episode_lengths)) == 1:
            # All episodes have same length - use efficient vectorized repeat
            length = episode_lengths[0]
            return mx.repeat(advantages, length)
        else:
            # Variable lengths - use loop but with pure MLX operations
            expanded = []
            for i, length in enumerate(episode_lengths):
                # Use mx.repeat to repeat the advantage value 'length' times
                # This avoids the .item() call and keeps operations on GPU
                episode_advantage = mx.repeat(advantages[i:i+1], length)
                expanded.append(episode_advantage)
            return mx.concatenate(expanded)
    
    def train(self, rollout_data: Optional[Union[Buffer, Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Train the model on complete rollout sequences (full token generations).

        Trains on complete generated sequences rather than single environment interactions. Use either:
        1. Automatic mode: Uses linked buffer with algorithm-specific data selection
        2. Manual mode: Takes provided rollout data

        When ``profile=True`` was passed at construction, ``mx.eval()`` barriers
        are inserted between phases and wall-clock times are recorded under
        ``timing/*`` keys in the returned metrics dict.

        Args:
            rollout_data: Optional data to train on. If None, uses linked buffer
                         with algorithm-specific data selection strategy.

        Returns:
            Training metrics dictionary

        Raises:
            ValueError: If no rollout_data provided and no buffer linked
        """
        timer = self._timer  # local alias — None when profiling is off

        if timer is not None:
            # Keep timings per-step to avoid unbounded growth across long runs.
            timer.reset()
            timer.start("total")

        # ── Phase: data_selection ──────────────────────────────────────
        if timer is not None:
            timer.start("data_selection")

        if rollout_data is None:
            # Automatic mode: use linked buffer with algorithm-specific selection
            if self.buffer is None:
                raise ValueError("No rollout_data provided and no buffer linked to trainer")
            batch_data = self.data_selector_fn(self.buffer)
        elif isinstance(rollout_data, Buffer):
            # Manual mode with buffer: use provided buffer
            batch_data = self._prepare_batch_from_buffer(rollout_data)
        else:
            # Manual mode with preprocessed data
            batch_data = rollout_data

        if timer is not None:
            # Eval all array values in the batch to flush pending work from
            # data selection (obs, act, logprob, rewards may all be lazy).
            mx.eval(*[v for v in batch_data.values() if isinstance(v, mx.array)])
            timer.stop("data_selection")

        # ── Phase: loss_and_grad ───────────────────────────────────────
        if timer is not None:
            timer.start("loss_and_grad")

        loss, grads = self.loss_and_grad_fn(batch_data)

        if timer is not None:
            mx.eval(loss)
            timer.stop("loss_and_grad")

        # ── Phase: grad_clip ───────────────────────────────────────────
        if timer is not None:
            timer.start("grad_clip")

        if self.max_grad_norm is not None:
            grads = self._clip_gradients(grads, self.max_grad_norm)

        if timer is not None:
            # Force clipped gradients to materialize so their cost isn't
            # attributed to optimizer_update.
            mx.eval(grads)
            timer.stop("grad_clip")

        # ── Phase: optimizer_update ────────────────────────────────────
        if timer is not None:
            timer.start("optimizer_update")

        self.optimizer.update(self.model, grads)

        if timer is not None:
            # Force parameter materialization before stopping the timer
            mx.eval(self.model.parameters())
            timer.stop("optimizer_update")

        # ── Phase: metrics ─────────────────────────────────────────────
        if timer is not None:
            timer.start("metrics")

        metrics: Dict[str, float] = {'loss': loss.item(), 'step': self._step_count}
        if self.metrics_fn is not None and self._step_count % self.metrics_interval == 0:
            # Compute new logprobs using the same pipeline as training to ensure consistency
            # This properly handles GRPO data structure with format conversion
            #
            # NOTE: This is a second model forward pass (the first happens inside
            # loss_and_grad_fn). Set metrics_interval > 1 to amortize this cost.
            observations = batch_data['obs']
            actions = batch_data['act']

            # Use GRPO-specific extraction if episode_lengths available, otherwise fallback
            if 'episode_lengths' in batch_data:
                episode_lengths = batch_data['episode_lengths']
                prompt_lengths = batch_data.get('prompt_lengths')
                new_logprobs = self._extract_grpo_logprobs(observations, actions, batch_data['logprob'], episode_lengths, prompt_lengths)
            else:
                # Fallback: add batch dimension if needed and call model
                if observations.ndim == 1:
                    model_input = observations[None]  # Add batch dimension for 1D flat sequences
                else:
                    model_input = observations  # Already batched
                model_output = self.model(model_input)
                new_logprobs = self.get_logprobs_fn(model_output, actions)

            algorithm_metrics = self.metrics_fn(
                batch_data['logprob'],
                new_logprobs,
                self.advantage_fn(batch_data['rewards'])
            )
            metrics.update(algorithm_metrics)

        if timer is not None:
            timer.stop("metrics")

        # ── Phase: state_update ────────────────────────────────────────
        if timer is not None:
            timer.start("state_update")

        self._step_count += 1
        self.metrics.update(metrics)

        # Auto-save LoRA adapters if enabled (invisible to user)
        self._save_lora_if_enabled()

        if timer is not None:
            timer.stop("state_update")

        # ── End total ──────────────────────────────────────────────────
        if timer is not None:
            timer.stop("total")
            # Append timing data to metrics with timing/ prefix
            breakdown = timer.format_breakdown("total")
            total_stats = timer.get_stats("total")
            metrics["timing/total_s"] = total_stats["mean"]
            for phase, info in breakdown.items():
                metrics[f"timing/{phase}_s"] = info["seconds"]
                metrics[f"timing/{phase}_pct"] = info["percent"]

        return metrics
    
    def _prepare_batch_from_buffer(self, buffer: Buffer) -> Dict[str, mx.array]:
        """
        Convert buffer episodes to training batch.
        
        Args:
            buffer: Buffer containing collected episodes
            
        Returns:
            Batch dictionary for training
        """
        # Sample all episodes from buffer 
        episodes_data = buffer.sample()  # This returns concatenated transitions
        
        # We need to convert this back to episode structure for reward extraction
        # For now, let's assume we have episode boundaries in the storage
        episodes = buffer.episodes  # Access episodes directly from storage
        
        if not episodes:
            raise ValueError("Buffer is empty - no episodes to train on")

        # Extract episode rewards and lengths
        # Build reward sums lazily, then evaluate in a single sync barrier
        episode_lengths = []
        pending_sums = []

        # Collect all transitions
        all_obs = []
        all_acts = []
        all_logprobs = []

        for i, episode in enumerate(episodes):
            # Support both Episode objects (attribute access) and dicts
            rew = episode.rew if hasattr(episode, 'rew') else episode['rew']
            obs = episode.obs if hasattr(episode, 'obs') else episode['obs']
            act = episode.act if hasattr(episode, 'act') else episode['act']
            if hasattr(episode, 'logprob'):
                logprob = episode.logprob
            elif isinstance(episode, dict):
                logprob = episode.get('logprob')
            else:
                logprob = None

            if logprob is None:
                raise ValueError(
                    "Episode index "
                    f"{i} is missing logprob. Training requires logprob values "
                    "from rollout collection."
                )

            pending_sums.append(mx.sum(mx.array(rew)))
            episode_lengths.append(len(obs))

            # Collect transitions
            all_obs.append(mx.array(obs))
            all_acts.append(mx.array(act))
            all_logprobs.append(mx.array(logprob))

        # Single sync barrier for all episode rewards
        reward_stack = mx.stack(pending_sums)
        mx.eval(reward_stack)
        episode_rewards = reward_stack.tolist()

        # Concatenate all transitions
        batch_data = {
            'obs': mx.concatenate(all_obs),
            'act': mx.concatenate(all_acts),
            'logprob': mx.concatenate(all_logprobs),
            'rewards': mx.array(episode_rewards),
            'episode_lengths': episode_lengths
        }
        
        return batch_data
    
    def _clip_gradients(self, grads: Dict[str, mx.array], max_norm: float) -> Dict[str, mx.array]:
        """
        Apply gradient clipping by global norm using MLX's built-in function.
        
        This function properly handles nested parameter structures (like transformers)
        using MLX's tree utilities for robust gradient clipping.
        
        Args:
            grads: Gradient dictionary (can contain nested structures)
            max_norm: Maximum gradient norm
            
        Returns:
            Clipped gradients with same structure as input
        """
        # Use MLX's built-in gradient clipping that handles nested parameter structures
        # This replaces the manual implementation that failed with nested dicts
        clipped_grads, total_norm = optim.clip_grad_norm(grads, max_norm)
        return clipped_grads
    
    def train_epoch(
        self, 
        rollout_coordinator: RolloutCoordinator,
        num_steps: int = 1
    ) -> List[Dict[str, float]]:
        """
        Train for multiple steps using rollout coordinator.
        
        Args:
            rollout_coordinator: Coordinator for collecting rollouts
            num_steps: Number of training steps
            
        Returns:
            List of metrics from each step
        """
        all_metrics = []
        
        for step in range(num_steps):
            # Collect rollout data
            buffer = rollout_coordinator.collect()
            
            # Train on collected data
            step_metrics = self.train(buffer)
            all_metrics.append(step_metrics)
            
            # Clear buffer for next iteration
            buffer.clear()
        
        return all_metrics
    
    @property
    def step_count(self) -> int:
        """Get current training step count (number of learning rounds completed)."""
        return self._step_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get accumulated training metrics."""
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Reset training metrics."""
        self.metrics.reset()

    def reset_timer(self):
        """Reset profiling timer history. No-op when profiling is disabled."""
        if self._timer is not None:
            self._timer.reset()
    
    def link_buffer(self, buffer: Buffer, data_selector_fn: Optional[Callable] = None):
        """
        Link a buffer to the trainer for automatic data selection.
        
        Args:
            buffer: Buffer to link for automatic training
            data_selector_fn: Optional algorithm-specific data selector.
                            If None, uses current data_selector_fn.
        """
        self.buffer = buffer
        if data_selector_fn is not None:
            self.data_selector_fn = data_selector_fn
    
    def unlink_buffer(self):
        """Unlink the buffer from the trainer."""
        self.buffer = None


# No factory functions by design.
# We maintain pure modular composition for MLX optimization.
# Users compose exactly what they need:
#
# from textpolicy.algorithms import grpo
# from textpolicy.training import Trainer
# 
# trainer = Trainer(
#     model=model,
#     advantage_fn=grpo.compute_advantages,  # Pure function
#     loss_fn=grpo.policy_loss,             # Pure function
#     optimizer=optimizer
# )
#
# This gives:
# - Low abstraction overhead (direct function calls)
# - MLX compilation works on the end-to-end pipeline (@mx.compile)
# - No dispatch overhead
# - Apple Silicon–friendly performance
