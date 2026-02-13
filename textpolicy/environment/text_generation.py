# textpolicy/environment/text_generation.py
"""
Text Generation Environment for Testing MLX RL Training.

This environment provides measurable text generation tasks to validate that
models are actually learning through RL training, not just going through motions.

Key features:
- Consistent, reproducible text generation tasks  
- Before/after learning validation metrics
- Integration with MLX generation system
- Support for various text generation benchmarks
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import mlx.core as mx
import random
from dataclasses import dataclass
from .base import Environment
from .task_suites import register_task_suite, get_task_suite

# Import our generation functions
from ..generation.mlx_generation import encode, decode, generate_tokens


@dataclass
class TextGenerationTask:
    """A single text generation task with validation criteria."""
    prompt: str
    target_keywords: List[str]
    target_length_range: Tuple[int, int]  # (min_words, max_words)
    difficulty: float  # 0.0 to 1.0
    category: str
    evaluation_criteria: Dict[str, Any]


# Default task suites for registration and internal use.
def _default_basic_tasks() -> List[TextGenerationTask]:
    return [
        TextGenerationTask(
            prompt="Write a brief explanation of machine learning.",
            target_keywords=["algorithm", "data", "learn"],
            target_length_range=(20, 40),
            difficulty=0.3,
            category="length_control",
            evaluation_criteria={"keyword_weight": 0.4, "length_weight": 0.6}
        ),
        TextGenerationTask(
            prompt="Describe the benefits of renewable energy in one paragraph.",
            target_keywords=["environment", "sustainable", "clean"],
            target_length_range=(30, 50),
            difficulty=0.4,
            category="length_control",
            evaluation_criteria={"keyword_weight": 0.5, "length_weight": 0.5}
        ),
        TextGenerationTask(
            prompt="Explain how computers work.",
            target_keywords=["processor", "memory", "software", "hardware"],
            target_length_range=(25, 45),
            difficulty=0.5,
            category="keyword_inclusion",
            evaluation_criteria={"keyword_weight": 0.7, "length_weight": 0.3}
        ),
        TextGenerationTask(
            prompt="Write about the importance of education.",
            target_keywords=["knowledge", "skills", "future", "learning"],
            target_length_range=(20, 40),
            difficulty=0.4,
            category="keyword_inclusion", 
            evaluation_criteria={"keyword_weight": 0.6, "length_weight": 0.4}
        ),
        TextGenerationTask(
            prompt="Explain the process of photosynthesis step by step.",
            target_keywords=["sunlight", "carbon", "oxygen", "glucose"],
            target_length_range=(35, 60),
            difficulty=0.6,
            category="coherence",
            evaluation_criteria={"keyword_weight": 0.3, "length_weight": 0.3, "coherence_weight": 0.4}
        ),
    ]


def _default_challenging_tasks() -> List[TextGenerationTask]:
    return [
        TextGenerationTask(
            prompt="Compare and contrast neural networks and traditional algorithms.",
            target_keywords=["pattern", "weights", "training", "classification", "regression"],
            target_length_range=(50, 80),
            difficulty=0.8,
            category="comparison",
            evaluation_criteria={"keyword_weight": 0.4, "length_weight": 0.3, "coherence_weight": 0.3}
        ),
        TextGenerationTask(
            prompt="Analyze the ethical implications of artificial intelligence.",
            target_keywords=["bias", "privacy", "autonomy", "responsibility", "society"],
            target_length_range=(60, 100),
            difficulty=0.9,
            category="analysis",
            evaluation_criteria={"keyword_weight": 0.3, "length_weight": 0.2, "coherence_weight": 0.5}
        ),
    ]


# Register defaults at import time to make them discoverable via the registry.
register_task_suite("basic", _default_basic_tasks)
register_task_suite("challenging", _default_challenging_tasks)


class TextGenerationEnvironment(Environment):
    """
    Environment for testing text generation learning with MLX models.
    
    This environment provides a suite of text generation tasks that allow
    measuring model improvement through RL training. It integrates directly
    with our MLX generation system and reward functions.
    
    Key validation approach:
    1. Pre-training baseline: Measure model performance on task suite
    2. Post-training comparison: Measure same model after RL training  
    3. Learning validation: Prove statistically significant improvement
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        task_suite: str = "basic",
        num_episodes: int = 50,
        generation_params: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        """
        Initialize text generation testing environment.
        
        Args:
            model: MLX model for text generation
            tokenizer: MLX tokenizer
            task_suite: Which task suite to use ("basic", "challenging", "custom")
            num_episodes: Number of episodes per evaluation
            generation_params: Parameters for text generation
            seed: Random seed for reproducible evaluation
        """
        super().__init__()

        # Validate critical dependencies early to produce clear, actionable errors.
        # This environment integrates directly with MLX generation; both model and
        # tokenizer are required. Without these, encode/decode/generate would fail
        # later with obscure attribute errors.
        if model is None:
            raise ValueError("TextGenerationEnvironment requires a valid MLX model (got None)")
        if tokenizer is None:
            raise ValueError("TextGenerationEnvironment requires a valid tokenizer (got None)")

        self.model = model
        self.tokenizer = tokenizer
        self.generation_params = generation_params or {
            'max_tokens': 50,
            'temperature': 0.8,
            'top_p': 0.95
        }
        
        # Create task suite for evaluation
        self.tasks = self._create_task_suite(task_suite)
        self.task_suite = task_suite  # remember suite type for cloning
        self.num_episodes = num_episodes
        self.current_episode = 0
        self.current_task = None
        
        # Performance tracking for learning validation
        self.baseline_scores = []
        self.current_scores = []
        
        # Environment state
        random.seed(seed)
        self._episode_data = []
        
        # Initialization complete; environment ready for evaluation
        # (Debug prints removed for production efficiency)
    
    def _create_task_suite(self, suite_type: str) -> List[TextGenerationTask]:
        """
        Create a suite of text generation tasks for evaluation.
        
        These tasks are designed to be:
        - Measurable: Clear success criteria
        - Diverse: Cover different generation challenges
        - Reproducible: Same tasks for before/after comparison
        """
        # Prefer registry-based suites when available; fall back to defaults here.
        # First, try registry-based loader (see environment.task_suites).
        # This enables custom suites without hardcoding here.
        registered = get_task_suite(suite_type)
        if registered is not None:
            return registered

        if suite_type == "basic":
            return _default_basic_tasks()
        
        elif suite_type == "challenging":
            return _default_challenging_tasks()
        
        else:  # custom or fallback
            return [
                TextGenerationTask(
                    prompt="Tell me about your favorite topic.",
                    target_keywords=["interesting", "because", "example"],
                    target_length_range=(15, 35),
                    difficulty=0.2,
                    category="open_ended",
                    evaluation_criteria={"keyword_weight": 0.5, "length_weight": 0.5}
                )
            ]

# Register default suites in the registry to enable external access (list/get).
# Done here to avoid import cycles: the loader closures capture TextGenerationTask.
def _default_basic_tasks() -> List[TextGenerationTask]:
    return [
        TextGenerationTask(
            prompt="Write a brief explanation of machine learning.",
            target_keywords=["algorithm", "data", "learn"],
            target_length_range=(20, 40),
            difficulty=0.3,
            category="length_control",
            evaluation_criteria={"keyword_weight": 0.4, "length_weight": 0.6}
        ),
        TextGenerationTask(
            prompt="Describe the benefits of renewable energy in one paragraph.",
            target_keywords=["environment", "sustainable", "clean"],
            target_length_range=(30, 50),
            difficulty=0.4,
            category="length_control",
            evaluation_criteria={"keyword_weight": 0.5, "length_weight": 0.5}
        ),
        TextGenerationTask(
            prompt="Explain how computers work.",
            target_keywords=["processor", "memory", "software", "hardware"],
            target_length_range=(25, 45),
            difficulty=0.5,
            category="keyword_inclusion",
            evaluation_criteria={"keyword_weight": 0.7, "length_weight": 0.3}
        ),
        TextGenerationTask(
            prompt="Write about the importance of education.",
            target_keywords=["knowledge", "skills", "future", "learning"],
            target_length_range=(20, 40),
            difficulty=0.4,
            category="keyword_inclusion", 
            evaluation_criteria={"keyword_weight": 0.6, "length_weight": 0.4}
        ),
        TextGenerationTask(
            prompt="Explain the process of photosynthesis step by step.",
            target_keywords=["sunlight", "carbon", "oxygen", "glucose"],
            target_length_range=(35, 60),
            difficulty=0.6,
            category="coherence",
            evaluation_criteria={"keyword_weight": 0.3, "length_weight": 0.3, "coherence_weight": 0.4}
        ),
    ]


    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment to start a new episode.
        
        Returns:
            (observation, info): Initial prompt and episode metadata
        """
        # Select next task (cycle through tasks)
        task_index = self.current_episode % len(self.tasks)
        self.current_task = self.tasks[task_index]
        
        # Reset episode state
        self._episode_data = {
            'prompt': self.current_task.prompt,
            'task': self.current_task,
            'responses': [],
            'scores': []
        }
        
        # Return initial observation (the prompt to generate from)
        observation = encode(self.tokenizer, self.current_task.prompt)
        
        info = {
            'episode': self.current_episode,
            'task_category': self.current_task.category,
            'difficulty': self.current_task.difficulty,
            'target_keywords': self.current_task.target_keywords,
            'target_length_range': self.current_task.target_length_range
        }
        
        return observation, info
    
    def step(self, action: Any) -> Dict[str, Any]:
        """
        Take a step in the environment by generating text response.
        
        Args:
            action: Generated response tokens (MLX array)
            
        Returns:
            Step result with observation, reward, termination status, and info
        """
        if self.current_task is None:
            raise ValueError("Environment not reset - call reset() first")
        
        # Decode response from tokens
        if hasattr(action, 'tolist'):
            # Action is MLX array of tokens
            response_text = decode(self.tokenizer, action)
        else:
            # Action might already be text
            response_text = str(action)
        
        # Compute reward using our reward system
        reward_score = self._evaluate_response(
            prompt=self.current_task.prompt,
            response=response_text,
            task=self.current_task
        )
        
        # Store episode data for analysis
        self._episode_data['responses'].append(response_text)
        self._episode_data['scores'].append(reward_score)
        
        # Episode terminates after each generation (single-turn tasks)
        terminated = True
        truncated = False
        
        # Prepare next observation (empty since episode ended)
        next_observation = mx.array([])
        
        info = {
            'response': response_text,
            'reward_score': reward_score,
            'task_category': self.current_task.category,
            'target_keywords_found': [kw for kw in self.current_task.target_keywords 
                                    if kw.lower() in response_text.lower()],
            'response_length': len(response_text.split()),
            'target_length_range': self.current_task.target_length_range
        }
        
        # Move to next episode
        self.current_episode += 1
        
        return {
            'observation': next_observation,
            'reward': reward_score,
            'terminated': terminated,
            'truncated': truncated,
            'info': info
        }
    
    def _evaluate_response(self, prompt: str, response: str, task: TextGenerationTask) -> float:
        """
        Evaluate response quality using task-specific criteria.
        
        This function integrates with our reward system to provide
        consistent, measurable evaluation of text generation quality.
        """
        criteria = task.evaluation_criteria
        total_score = 0.0
        
        # Length-based scoring
        if 'length_weight' in criteria:
            word_count = len(response.split())
            min_len, max_len = task.target_length_range
            target_len = (min_len + max_len) / 2
            
            # Score based on proximity to target length
            if min_len <= word_count <= max_len:
                length_score = 1.0
            else:
                # Penalty for being outside range
                distance = min(abs(word_count - min_len), abs(word_count - max_len))
                length_score = max(0.0, 1.0 - distance / target_len)
            
            total_score += criteria['length_weight'] * length_score
        
        # Keyword inclusion scoring
        if 'keyword_weight' in criteria:
            keywords_found = sum(1 for kw in task.target_keywords 
                               if kw.lower() in response.lower())
            keyword_score = keywords_found / len(task.target_keywords)
            total_score += criteria['keyword_weight'] * keyword_score
        
        # Coherence scoring (simple heuristic)
        if 'coherence_weight' in criteria:
            # Use our existing coherence evaluation
            coherence_score = self._simple_coherence_score(response)
            total_score += criteria['coherence_weight'] * coherence_score
        
        return total_score
    
    def _simple_coherence_score(self, text: str) -> float:
        """Simple coherence scoring based on structure indicators."""
        if not text.strip():
            return 0.0
        
        # Basic coherence indicators
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.5  # Single sentence is moderately coherent
        
        # Look for logical connectors
        connectors = ['therefore', 'however', 'moreover', 'furthermore', 'because', 'since']
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        
        # Coherence score based on structure
        connector_score = min(1.0, connector_count / 2.0)  # 2+ connectors is good
        sentence_score = min(1.0, len(sentences) / 3.0)    # 3+ sentences is good
        
        return (connector_score + sentence_score) / 2.0
    
    def evaluate_model(self, mode: str = "current") -> Dict[str, float]:
        """
        Evaluate model performance on the full task suite.
        
        This function runs through all tasks and computes aggregate
        performance metrics to measure learning progress.
        
        Args:
            mode: "baseline" (store baseline) or "current" (compare to baseline)
            
        Returns:
            Performance metrics dictionary
        """
        print(f"Running {mode} evaluation on {self.num_episodes} episodes...")
        
        all_scores = []
        category_scores = {}
        
        # Reset episode counter for evaluation
        original_episode = self.current_episode
        self.current_episode = 0
        
        try:
            for episode in range(self.num_episodes):
                # Reset environment
                observation, info = self.reset()
                
                # Generate response using current model
                response_tokens, generation_info = generate_tokens(
                    model=self.model,
                    tokenizer=self.tokenizer, 
                    prompt_tokens=observation,
                    **self.generation_params
                )
                
                # Take step to get reward
                step_result = self.step(response_tokens)
                score = step_result['reward']
                category = step_result['info']['task_category']
                
                all_scores.append(score)
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)
        
        finally:
            # Restore episode counter
            self.current_episode = original_episode
        
        # Compute aggregate metrics
        mean_score = float(mx.mean(mx.array(all_scores)))
        std_score = float(mx.std(mx.array(all_scores)))
        
        metrics = {
            'mean_score': mean_score,
            'std_score': std_score,
            'num_episodes': self.num_episodes,
            'category_breakdown': {
                cat: float(mx.mean(mx.array(scores))) 
                for cat, scores in category_scores.items()
            }
        }
        
        # Store results based on mode
        if mode == "baseline":
            self.baseline_scores = all_scores
            print(f"✓ Baseline evaluation complete: {mean_score:.3f} ± {std_score:.3f}")
        else:
            self.current_scores = all_scores
            
            # Compute learning improvement if we have baseline
            if self.baseline_scores:
                baseline_mean = float(mx.mean(mx.array(self.baseline_scores)))
                improvement = mean_score - baseline_mean
                improvement_pct = (improvement / baseline_mean) * 100 if baseline_mean > 0 else 0
                
                metrics['baseline_score'] = baseline_mean
                metrics['improvement'] = improvement
                metrics['improvement_percent'] = improvement_pct
                
                print(f"Current evaluation complete: {mean_score:.3f} ± {std_score:.3f}")
                print(f"  Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
                
                # Statistical significance test (simple)
                if improvement > 2 * std_score:  # Rough 2-sigma test
                    print("  LEARNING DETECTED: Statistically significant improvement!")
                else:
                    print("  Learning uncertain: Improvement not statistically significant")
        
        return metrics
    
    @property
    def observation_space(self) -> Any:
        """Observation space is tokenized text (variable length)."""
        return "TokenizedText"  # Placeholder - MLX doesn't need gym spaces
    
    @property
    def action_space(self) -> Any:
        """Action space is generated text tokens (variable length)."""
        return "GeneratedTokens"  # Placeholder - MLX doesn't need gym spaces
    
    def clone(self) -> 'TextGenerationEnvironment':
        """Create a multiprocessing clone with the same configuration.

        This returns a new environment instance that references the same
        model/tokenizer objects. On some systems, MLX models are not picklable;
        for process spawning, prefer passing an environment factory (env_fn)
        so model/tokenizer can be constructed in each process. See rollout.coordinator.
        """
        # Delegate to the same constructor with preserved parameters
        return TextGenerationEnvironment(
            model=self.model,
            tokenizer=self.tokenizer,
            task_suite=self.task_suite,
            num_episodes=self.num_episodes,
            generation_params=self.generation_params,
            seed=random.randint(0, 10000)
        )


def create_text_generation_test_env(
    model: Any,
    tokenizer: Any,
    task_suite: str = "basic",
    num_episodes: int = 50,
    **kwargs
) -> TextGenerationEnvironment:
    """
    Factory function to create a text generation testing environment.
    
    This is the main entry point for creating environments to test
    whether RL training actually improves model performance.
    
    Args:
        model: MLX model for text generation
        tokenizer: MLX tokenizer  
        task_suite: Which task suite to use for evaluation
        num_episodes: Number of episodes per evaluation
        **kwargs: Additional environment parameters
        
    Returns:
        Configured TextGenerationEnvironment ready for testing
    """
    return TextGenerationEnvironment(
        model=model,
        tokenizer=tokenizer,
        task_suite=task_suite,
        num_episodes=num_episodes,
        **kwargs
    )


def validate_learning_progress(
    env: TextGenerationEnvironment,
    pre_training_metrics: Dict[str, float],
    post_training_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Pure function to validate that learning actually occurred.
    
    This function provides statistical analysis to prove that
    RL training resulted in measurable improvement.
    
    Args:
        env: The environment used for testing
        pre_training_metrics: Metrics before training
        post_training_metrics: Metrics after training
        
    Returns:
        Learning validation report
    """
    improvement = post_training_metrics['mean_score'] - pre_training_metrics['mean_score']
    improvement_pct = (improvement / pre_training_metrics['mean_score']) * 100
    
    # Simple statistical significance test
    pre_std = pre_training_metrics['std_score']
    post_std = post_training_metrics['std_score']
    pooled_std = (pre_std + post_std) / 2
    
    significance_threshold = 2 * pooled_std  # Rough 2-sigma test
    is_significant = abs(improvement) > significance_threshold
    
    validation_report = {
        'learning_detected': improvement > 0 and is_significant,
        'improvement_score': improvement,
        'improvement_percent': improvement_pct,
        'statistical_significance': is_significant,
        'significance_threshold': significance_threshold,
        'pre_training_score': pre_training_metrics['mean_score'],
        'post_training_score': post_training_metrics['mean_score'],
        'recommendation': (
            "LEARNING CONFIRMED: Model shows statistically significant improvement"
            if improvement > 0 and is_significant
            else "LEARNING UNCERTAIN: No significant improvement detected"
        )
    }
    
    return validation_report


class TextGenerationEnv(Environment):
    """
    Simple text generation environment for RL training.
    
    This is a lightweight wrapper around TextGenerationEnvironment that provides
    the simple interface expected by training examples. It's designed for:
    - Simple prompt-based training tasks
    - External reward function integration
    - Basic RL training workflows
    
    For comprehensive testing and validation, use TextGenerationEnvironment instead.
    """
    
    def __init__(
        self,
        prompts: List[str],
        reward_fn: Callable[[str, str, dict], Any],
        max_tokens: int = 25,
        seed: int = 42,
        tokenizer: Any = None,
        examples: Optional[List[dict]] = None,
        group_size: int = 1,
    ):
        """
        Initialize simple text generation environment.

        Args:
            prompts: List of prompts to cycle through
            reward_fn: Function that computes reward from (prompt, completion, example).
                May return:
                - scalar reward
                - dict {"reward": float, "is_correct": bool}
                - tuple/list (reward, is_correct)
            max_tokens: Maximum tokens to generate per response
            seed: Random seed for reproducible behavior
            tokenizer: Tokenizer for converting prompts to tokens (required for MLX compatibility)
            examples: Optional list of example dicts to pass to reward function. If provided,
                      must have same length as prompts. examples[i] is passed when prompts[i] is used.
            group_size: Number of consecutive episodes that share the same prompt.
                Set to episodes_per_step for GRPO so that group-relative advantages
                compare completions of the *same* problem. Default 1 preserves the
                legacy round-robin behaviour.
        """
        super().__init__()

        if tokenizer is None:
            raise ValueError("tokenizer is required for TextGenerationEnv to work with MLX rollout system")

        if examples is not None and len(examples) != len(prompts):
            raise ValueError(f"examples length ({len(examples)}) must match prompts length ({len(prompts)})")

        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")

        self.prompts = prompts
        self.examples = examples if examples is not None else [{} for _ in prompts]
        self.reward_fn = reward_fn
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.current_episode = 0
        self.current_prompt = None

        # Environment state
        random.seed(seed)

        # Debug prints removed for production efficiency
    
    def reset(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset environment to start a new episode.
        
        Returns:
            (observation, info): Current prompt tokens and episode metadata
        """
        # Cycle through prompts, repeating each prompt for `group_size`
        # consecutive episodes.  With group_size=episodes_per_step every
        # episode in a training batch shares the same prompt — required for
        # correct GRPO group-relative advantage normalisation and enabling
        # shared KV-cache prefill (Opt 2).
        prompt_index = (self.current_episode // self.group_size) % len(self.prompts)
        self.current_prompt = self.prompts[prompt_index]
        
        # Tokenize prompt for MLX compatibility
        # Import encode function from mlx_generation to avoid circular imports
        from ..generation.mlx_generation import encode
        observation = encode(self.tokenizer, self.current_prompt)
        
        info = {
            'episode': self.current_episode,
            'prompt_index': prompt_index,
            'max_tokens': self.max_tokens,
            'prompt_text': self.current_prompt  # Keep original text for reward computation
        }
        
        return observation, info
    
    def step(self, action: Any) -> Dict[str, Any]:
        """
        Take a step in the environment by evaluating generated text.
        
        Args:
            action: Generated text response (string or token array)
            
        Returns:
            Dictionary with keys: observation, reward, terminated, truncated, info.
            This matches the Environment base class contract. The rollout runner
            normalizes both dict and tuple returns, so returning a dict here keeps
            interfaces consistent and compatible with rollouts.
        """
        if self.current_prompt is None:
            raise ValueError("Environment not reset - call reset() first")
        
        # Handle different action types - properly decode token arrays to text
        if hasattr(action, 'tolist'):
            # Action is MLX array of tokens - decode to text using tokenizer
            try:
                from ..generation.mlx_generation import decode
                response_text = decode(self.tokenizer, action)
            except Exception as e:
                print(f"WARNING: Failed to decode MLX action array: {e}")
                # Fallback: try to handle as raw tokens
                try:
                    response_text = self.tokenizer.decode(action.tolist())
                except Exception as e2:
                    print(f"WARNING: Fallback decode also failed: {e2}")
                    response_text = "Generated response (decode failed)"
        elif isinstance(action, list) and len(action) > 0 and isinstance(action[0], (int, float)):
            # Action is a Python list of token IDs - decode to text
            try:
                response_text = self.tokenizer.decode(action)
            except Exception as e:
                print(f"WARNING: Failed to decode token list: {e}")
                response_text = "Generated response (decode failed)"
        else:
            # Action is already text or something else
            response_text = str(action)
        
        # Detect if response was truncated by max_tokens limit
        # This happens when the generation hits the token limit before naturally ending
        response_tokens = len(response_text.split()) if response_text else 0
        truncated = response_tokens >= (self.max_tokens * 0.95)  # Consider 95% of limit as likely truncated
        
        # Episode terminates after each generation (single-turn tasks)
        terminated = True
        
        # Compute reward using provided reward function.
        # Supports either:
        # - scalar reward
        # - dict with {"reward": float, "is_correct": bool}
        # - tuple/list of (reward, is_correct)
        # Pass tokenizer for EOS token detection and truncation detection.
        prompt_index = (self.current_episode // self.group_size) % len(self.prompts)
        reward_output = self.reward_fn(
            prompt=self.current_prompt,
            completion=response_text,
            example=self.examples[prompt_index],
            tokenizer=self.tokenizer,  # Pass tokenizer for EOS detection
            truncated=truncated        # Pass truncation flag from environment
        )
        reward, is_correct = self._parse_reward_output(reward_output)
        
        # Prepare next observation (empty MLX array since episode ended)
        next_observation = mx.array([])
        
        info = {
            'response': response_text,
            'reward': reward,
            'prompt': self.current_prompt,
            'episode': self.current_episode
        }
        if is_correct is not None:
            info['is_correct'] = bool(is_correct)
        
        # Move to next episode
        self.current_episode += 1
        
        # Return unified dict format per Environment contract.
        # Runner code now normalizes both dict and tuple step results, so
        # this remains fully compatible with rollout collection while aligning
        # with our base interface and other adapters (GymAdapter, VectorizedEnvironment).
        return {
            'observation': next_observation,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
        }

    @staticmethod
    def _parse_reward_output(reward_output: Any) -> Tuple[float, Optional[bool]]:
        """Normalize reward function outputs into (reward, is_correct)."""
        reward: float
        is_correct: Optional[bool] = None

        if isinstance(reward_output, dict):
            if 'reward' not in reward_output:
                raise ValueError(
                    "reward_fn dict output must include a 'reward' key."
                )
            reward = float(reward_output['reward'])
            if 'is_correct' in reward_output:
                raw = reward_output['is_correct']
                if hasattr(raw, "item"):
                    raw = raw.item()
                is_correct = bool(raw)
            return reward, is_correct

        if isinstance(reward_output, (tuple, list)) and len(reward_output) == 2:
            reward = float(reward_output[0])
            raw = reward_output[1]
            if raw is not None:
                if hasattr(raw, "item"):
                    raw = raw.item()
                is_correct = bool(raw)
            return reward, is_correct

        return float(reward_output), None
    
    @property
    def observation_space(self) -> Any:
        """Observation space is text prompts (variable length)."""
        return "TextPrompt"  # Placeholder - MLX doesn't need gym spaces
    
    @property
    def action_space(self) -> Any:
        """Action space is generated text responses (variable length)."""
        return "GeneratedText"  # Placeholder - MLX doesn't need gym spaces
    
    def clone(self) -> 'TextGenerationEnv':
        """Create a clone for multiprocessing."""
        return TextGenerationEnv(
            prompts=self.prompts.copy(),
            reward_fn=self.reward_fn,
            max_tokens=self.max_tokens,
            tokenizer=self.tokenizer,  # Tokenizer is required for MLX compatibility
            seed=random.randint(0, 10000),  # New seed for variety
            examples=self.examples.copy(),
            group_size=self.group_size,
        )
