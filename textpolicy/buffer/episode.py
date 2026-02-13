# textpolicy/buffer/episode.py
"""
Single episode trajectory management.

The Episode class stores transitions as Python lists during rollout,
then converts to MLX arrays only at sampling time. This aims to be optimal
for Apple Silicon's unified memory architecture.
"""

from typing import Optional, Any, Dict
import mlx.core as mx # type: ignore


class Episode:
    """
    Represents a single complete episode trajectory.

    Stores transitions as Python lists during rollout, then converts to MLX arrays
    only at sampling time. This aims to be optimal for Apple Silicon's unified memory.

    All optional fields (e.g. `logprob`, `value`) must be provided for **all steps**
    or **none** — mixing will raise an error. This ensures tensor shape consistency.

    Example:
        ep = Episode()
        ep.append(obs=1, act=0, rew=1, next_obs=2, done=False, logprob=0.1, value=1.5)
        ep.append(obs=2, act=1, rew=2, next_obs=3, done=True, logprob=0.2, value=2.5)

        batch = ep.to_tensor_dict()  # Returns dict of MLX arrays
    """
    
    def __init__(self):
        """Initialize empty episode with required fields."""
        # Required fields - always present
        self.obs: list[Any] = []
        self.act: list[Any] = []
        self.rew: list[Any] = []
        self.next_obs: list[Any] = []
        self.done: list[bool] = []
        self.timeout: list[bool] = []

        # Optional fields - all-or-nothing consistency
        self.logprob: Optional[list[Any]] = None
        self.value: Optional[list[Any]] = None
        self.entropy: Optional[list[Any]] = None
        self.is_correct: Optional[list[Any]] = None

    def append(
        self, 
        obs, 
        act, 
        rew, 
        next_obs, 
        done, 
        timeout=False, 
        logprob=None, 
        value=None, 
        entropy=None,
        is_correct=None,
    ):
        """
        Append a single environment transition to the episode.

        Args:
            obs: Observation from environment
            act: Action taken
            rew: Reward received
            next_obs: Next observation
            done: Boolean indicating episode termination
            timeout: Boolean indicating truncation (e.g. time limit)
            logprob: Log probability of action (optional, but must be all-or-nothing)
            value: Estimated state value (optional, but must be all-or-nothing)
            entropy: Action entropy (optional, but must be all-or-nothing)
            is_correct: Verifier correctness signal (optional, but must be
                all-or-nothing)

        Raises:
            ValueError: If optional fields are inconsistent (some provided, some missing)

        Example:
            episode.append(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
        """
        # Store required fields
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.next_obs.append(next_obs)
        self.done.append(done)
        self.timeout.append(timeout)

        # Handle logprob: must be all-or-nothing
        if logprob is not None:
            if self.logprob is None:
                self.logprob = []
            self.logprob.append(logprob)
        else:
            if self.logprob is not None:
                raise ValueError(
                    "This episode includes logprob, but one step is missing it. "
                    "Either provide logprob for all steps or none."
                )

        # Handle value: must be all-or-nothing
        if value is not None:
            if self.value is None:
                self.value = []
            self.value.append(value)
        else:
            if self.value is not None:
                raise ValueError(
                    "This episode includes value, but one step is missing it. "
                    "Either provide value for all steps or none."
                )

        # Handle entropy: must be all-or-nothing
        if entropy is not None:
            if self.entropy is None:
                self.entropy = []
            self.entropy.append(entropy)
        else:
            if self.entropy is not None:
                raise ValueError(
                    "This episode includes entropy, but one step is missing it. "
                    "Either provide entropy for all steps or none."
                )

        # Handle is_correct: must be all-or-nothing
        if is_correct is not None:
            if self.is_correct is None:
                self.is_correct = []
            self.is_correct.append(is_correct)
        else:
            if self.is_correct is not None:
                raise ValueError(
                    "This episode includes is_correct, but one step is missing it. "
                    "Either provide is_correct for all steps or none."
                )

    def __len__(self) -> int:
        """Return the number of steps in this episode."""
        return len(self.obs)

    def to_tensor_dict(self) -> Dict[str, mx.array]:
        """
        Convert all stored data to MLX arrays for training.
        Performed once at sample time for efficiency on Apple Silicon and MLX.

        Returns:
            Dict of MLX arrays with keys:
            - 'obs': (T, *obs_shape) - observations
            - 'act': (T, *act_shape) - actions  
            - 'rew': (T,) - rewards
            - 'next_obs': (T, *obs_shape) - next observations
            - 'done': (T,) - termination flags
            - 'timeout': (T,) - truncation flags
            - 'logprob': (T,) - log probabilities (if provided)
            - 'value': (T,) - value estimates (if provided)
            - 'entropy': (T,) - action entropy (if provided)
            - 'is_correct': (T,) - verifier correctness flags (if provided)

        Notes:
            This runs once at sample time and uses batched array conversion.
        """
        # Batched array conversion for improved memory efficiency
        # Convert to numpy first, then a single MLX array
        import numpy as np
        
        # Convert required fields to MLX arrays - BATCHED APPROACH
        try:
            # Try numpy-based batched conversion first (most efficient)
            obs_np = np.array(self.obs)
            next_obs_np = np.array(self.next_obs)
            act_np = np.array(self.act)
            
            result = {
                'obs': mx.array(obs_np),              # Single batched conversion
                'act': mx.array(act_np),              # Single batched conversion  
                'rew': mx.array(self.rew),            # Already efficient for scalars
                'next_obs': mx.array(next_obs_np),    # Single batched conversion
                'done': mx.array(self.done),          # Already efficient for booleans
                'timeout': mx.array(self.timeout),    # Already efficient for booleans
            }
        except (ValueError, TypeError):
            # Batch conversion fallback with pre-allocation
            # (for heterogeneous data types or complex structures)
            try:
                # Try batch conversion first (faster for homogeneous data)
                import numpy as np
                result = {
                    'obs': mx.array(np.array(self.obs)),
                    'act': mx.array(np.array(self.act)),
                    'rew': mx.array(self.rew),
                    'next_obs': mx.array(np.array(self.next_obs)),
                    'done': mx.array(self.done),
                    'timeout': mx.array(self.timeout),
                }
            except:
                # Fallback for heterogeneous data - try stacking first
                try:
                    result = {
                        'obs': mx.stack([mx.array(o) for o in self.obs]),
                        'act': mx.stack([mx.array(a) for a in self.act]),
                        'rew': mx.array(self.rew),
                        'next_obs': mx.stack([mx.array(o) for o in self.next_obs]),
                        'done': mx.array(self.done),
                        'timeout': mx.array(self.timeout),
                    }
                except:
                    # Final fallback for truly heterogeneous shapes - return as list of arrays
                    # This handles cases where observations have completely different shapes
                    result = {
                        'obs': [mx.array(o) for o in self.obs],
                        'act': [mx.array(a) for a in self.act] if not all(isinstance(a, (int, float)) for a in self.act) else mx.array(self.act),
                        'rew': mx.array(self.rew),
                        'next_obs': [mx.array(o) for o in self.next_obs],
                        'done': mx.array(self.done),
                        'timeout': mx.array(self.timeout),
                    }

        # Add optional fields if present - handle variable-length sequences properly
        if self.logprob is not None:
            # Handle variable-length logprob sequences (common in text generation)
            # Each transition may have different response lengths, so we flatten them
            try:
                # Try direct conversion first (for uniform lengths)
                result['logprob'] = mx.array(self.logprob)
            except ValueError as e:
                if "non-uniform length" in str(e):
                    # Handle variable-length sequences by flattening
                    # This preserves all logprob data while making it MLX-compatible
                    flattened_logprobs = []
                    for logprob_item in self.logprob:
                        if hasattr(logprob_item, 'tolist'):  # MLX array
                            flattened_logprobs.extend(logprob_item.tolist())
                        elif isinstance(logprob_item, list):  # Python list
                            flattened_logprobs.extend(logprob_item)
                        else:  # Single value
                            flattened_logprobs.append(float(logprob_item))
                    result['logprob'] = mx.array(flattened_logprobs) if flattened_logprobs else mx.array([])
                else:
                    # Re-raise other ValueError types
                    raise
        
        if self.value is not None:
            # Apply same variable-length handling to value if needed
            try:
                result['value'] = mx.array(self.value)
            except ValueError as e:
                if "non-uniform length" in str(e):
                    flattened_values = []
                    for value_item in self.value:
                        if hasattr(value_item, 'tolist'):  # MLX array
                            flattened_values.extend(value_item.tolist())
                        elif isinstance(value_item, list):  # Python list
                            flattened_values.extend(value_item)
                        else:  # Single value
                            flattened_values.append(float(value_item))
                    result['value'] = mx.array(flattened_values) if flattened_values else mx.array([])
                else:
                    raise
        
        if self.entropy is not None:
            # Apply same variable-length handling to entropy if needed
            try:
                result['entropy'] = mx.array(self.entropy)
            except ValueError as e:
                if "non-uniform length" in str(e):
                    flattened_entropy = []
                    for entropy_item in self.entropy:
                        if hasattr(entropy_item, 'tolist'):  # MLX array
                            flattened_entropy.extend(entropy_item.tolist())
                        elif isinstance(entropy_item, list):  # Python list
                            flattened_entropy.extend(entropy_item)
                        else:  # Single value
                            flattened_entropy.append(float(entropy_item))
                    result['entropy'] = mx.array(flattened_entropy) if flattened_entropy else mx.array([])
                else:
                    raise

        if self.is_correct is not None:
            result['is_correct'] = mx.array(self.is_correct)

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert episode to dictionary for serialization (multiprocessing).

        Used for inter-process communication where MLX arrays can't be shared.
        This preserves all data as Python-native types for queue transmission.

        Returns:
            Dictionary representation with all Python-native types.
            This is the inverse of creating an episode from a dict.

        Example:
            # In worker process
            ep_dict = episode.to_dict()
            queue.put(ep_dict)

            # In trainer process  
            buffer.add_episode_from_dict(ep_dict)
        """
        # Always include required fields
        result = {
            'obs': self.obs,
            'act': self.act,
            'rew': self.rew,
            'next_obs': self.next_obs,
            'done': self.done,
            'timeout': self.timeout,
        }

        # Add optional fields if present
        if self.logprob is not None:
            result['logprob'] = self.logprob
        if self.value is not None:
            result['value'] = self.value
        if self.entropy is not None:
            result['entropy'] = self.entropy
        if self.is_correct is not None:
            result['is_correct'] = self.is_correct

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """
        Create Episode from dictionary representation (for deserialization).
        
        This is the inverse of to_dict() - reconstructs an Episode from
        serialized dictionary data, typically used after inter-process
        communication where Episode objects are transmitted as dicts.
        
        Args:
            data: Dictionary containing episode data with Python-native types
            
        Returns:
            New Episode instance with data from the dictionary
            
        Example:
            # Reconstruct episode from serialized data
            episode = Episode.from_dict(ep_dict)
        """
        episode = cls()
        
        # Reconstruct episode by appending each step
        length = len(data['obs'])
        for i in range(length):
            step_data = {
                'obs': data['obs'][i],
                'act': data['act'][i], 
                'rew': data['rew'][i],
                'next_obs': data['next_obs'][i],
                'done': data['done'][i],
                'timeout': data['timeout'][i] if i < len(data['timeout']) else False
            }
            
            # Add optional fields if present in the data
            if 'logprob' in data and i < len(data['logprob']):
                step_data['logprob'] = data['logprob'][i]
            if 'value' in data and i < len(data['value']):
                step_data['value'] = data['value'][i]
            if 'entropy' in data and i < len(data['entropy']):
                step_data['entropy'] = data['entropy'][i]
            if 'is_correct' in data and i < len(data['is_correct']):
                step_data['is_correct'] = data['is_correct'][i]
                
            episode.append(**step_data)
            
        return episode

    def validate_consistency(self):
        """
        Validate internal consistency of episode data.
        
        Checks:
        - All required fields have same length
        - Optional fields have correct length if present
        - Episode has at least one step
        
        Raises:
            ValueError: If episode data is inconsistent
        """
        if len(self) == 0:
            raise ValueError("Episode is empty")
            
        # Check required fields have consistent length
        required_lengths = [
            len(self.obs), len(self.act), len(self.rew),
            len(self.next_obs), len(self.done), len(self.timeout)
        ]
        
        if not all(length == required_lengths[0] for length in required_lengths):
            raise ValueError(f"Inconsistent required field lengths: {required_lengths}")
        
        # Check optional fields have correct length if present
        episode_length = len(self.obs)
        
        if self.logprob is not None and len(self.logprob) != episode_length:
            raise ValueError(f"logprob length {len(self.logprob)} != episode length {episode_length}")
            
        if self.value is not None and len(self.value) != episode_length:
            raise ValueError(f"value length {len(self.value)} != episode length {episode_length}")
            
        if self.entropy is not None and len(self.entropy) != episode_length:
            raise ValueError(f"entropy length {len(self.entropy)} != episode length {episode_length}") 

        if self.is_correct is not None and len(self.is_correct) != episode_length:
            raise ValueError(
                f"is_correct length {len(self.is_correct)} != episode length {episode_length}"
            )
