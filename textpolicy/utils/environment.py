"""
Environment analysis and profiling utilities for MLX-RL.

This module provides tools to analyze environment characteristics and determine
optimal parallelization strategies based on empirical performance testing.
"""

import time
import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from .timing import Timer
from .debug import debug_print


class EnvironmentType(Enum):
    """Classification of environment computational complexity."""
    ULTRA_LIGHT = "ultra_light"      # <1ms per step (CartPole, MountainCar)
    LIGHT = "light"                  # 1-5ms per step  
    MODERATE = "moderate"            # 5-20ms per step
    HEAVY = "heavy"                  # 20-100ms per step
    ULTRA_HEAVY = "ultra_heavy"      # >100ms per step


@dataclass
class EnvironmentProfile:
    """Environment performance characteristics."""
    name: str
    avg_step_time_ms: float
    steps_per_second: float
    avg_episode_length: float
    environment_type: EnvironmentType
    vectorization_recommended: bool
    recommended_num_envs: Optional[int]
    reasoning: str


class EnvironmentAnalyzer:
    """
    Analyzes environment performance characteristics to guide optimization decisions.
    
    Based on empirical testing from MLX-RL vectorization analysis:
    - CartPole-v1: ~250k steps/sec (ultra_light)
    - MountainCar-v0: ~225k steps/sec (ultra_light) 
    - Acrobot-v1: ~42k steps/sec (light)
    """
    
    # Performance thresholds based on empirical analysis
    STEP_TIME_THRESHOLDS = {
        EnvironmentType.ULTRA_LIGHT: 0.005,  # <5ms per step
        EnvironmentType.LIGHT: 0.02,         # 5-20ms per step
        EnvironmentType.MODERATE: 0.1,       # 20-100ms per step
        EnvironmentType.HEAVY: 0.5,          # 100-500ms per step
        EnvironmentType.ULTRA_HEAVY: float('inf')  # >500ms per step
    }
    
    def __init__(self, test_steps: int = 500, warmup_steps: int = 50):
        """
        Initialize environment analyzer.
        
        Args:
            test_steps: Number of steps to use for performance testing
            warmup_steps: Number of warmup steps (excluded from timing)
        """
        self.test_steps = test_steps
        self.warmup_steps = warmup_steps
        self.timer = Timer()
        
    def analyze_environment(self, env_name: str, **env_kwargs) -> EnvironmentProfile:
        """
        Analyze an environment's performance characteristics.
        
        Args:
            env_name: Environment name (e.g., "CartPole-v1")
            **env_kwargs: Additional environment creation arguments
            
        Returns:
            EnvironmentProfile with performance analysis and recommendations
        """
        debug_print(f"Analyzing environment: {env_name}", "environment")
        
        try:
            # Import here to avoid circular dependencies
            from textpolicy.environment import GymAdapter
            
            env = GymAdapter(env_name, **env_kwargs)
            
            # Warmup
            obs, info = env.reset()
            for _ in range(self.warmup_steps):
                action = self._sample_action(env)
                step_result = env.step(action)
                if step_result["terminated"] or step_result["truncated"]:
                    obs, info = env.reset()
            
            # Performance measurement
            step_times = []
            episode_lengths = []
            current_episode_length = 0
            
            obs, info = env.reset()
            
            start_time = time.perf_counter()
            for i in range(self.test_steps):
                action = self._sample_action(env)
                
                step_start = time.perf_counter()
                step_result = env.step(action)
                step_end = time.perf_counter()
                
                step_times.append(step_end - step_start)
                current_episode_length += 1
                
                if step_result["terminated"] or step_result["truncated"]:
                    episode_lengths.append(current_episode_length)
                    current_episode_length = 0
                    obs, info = env.reset()
            
            end_time = time.perf_counter()
            
            env.close()
            
            # Calculate metrics
            total_time = end_time - start_time
            avg_step_time = np.mean(step_times)
            steps_per_second = self.test_steps / total_time
            avg_episode_length = np.mean(episode_lengths) if episode_lengths else current_episode_length
            
            # Classify environment type
            env_type = self._classify_environment_type(avg_step_time)
            
            # Generate recommendations
            vectorization_recommended, recommended_num_envs, reasoning = self._generate_recommendations(
                env_type, avg_step_time, steps_per_second
            )
            
            return EnvironmentProfile(
                name=env_name,
                avg_step_time_ms=float(avg_step_time * 1000),
                steps_per_second=float(steps_per_second),
                avg_episode_length=float(avg_episode_length),
                environment_type=env_type,
                vectorization_recommended=vectorization_recommended,
                recommended_num_envs=recommended_num_envs,
                reasoning=reasoning
            )
            
        except Exception as e:
            debug_print(f"Error analyzing environment {env_name}: {e}", "environment")
            # Return minimal profile on error
            return EnvironmentProfile(
                name=env_name,
                avg_step_time_ms=0.0,
                steps_per_second=0.0,
                avg_episode_length=0.0,
                environment_type=EnvironmentType.MODERATE,
                vectorization_recommended=False,
                recommended_num_envs=None,
                reasoning=f"Analysis failed: {e}"
            )
    
    def _sample_action(self, env) -> Union[int, float, np.ndarray]:
        """Sample a random action from the environment's action space."""
        try:
            # Use gymnasium's action space sampling
            return env.action_space.sample()
        except Exception:
            # Fallback for discrete spaces
            if hasattr(env.action_space, 'n'):
                return np.random.randint(0, env.action_space.n)
            else:
                # Assume continuous space
                return np.random.random()
    
    def _classify_environment_type(self, avg_step_time: float) -> EnvironmentType:
        """Classify environment based on average step time."""
        for env_type, threshold in self.STEP_TIME_THRESHOLDS.items():
            if avg_step_time <= threshold:
                return env_type
        return EnvironmentType.ULTRA_HEAVY
    
    def _generate_recommendations(self, env_type: EnvironmentType, avg_step_time: float, 
                                steps_per_second: float) -> Tuple[bool, Optional[int], str]:
        """
        Generate vectorization recommendations based on environment analysis.
        
        Returns:
            Tuple of (should_vectorize, recommended_num_envs, reasoning)
        """
        if env_type == EnvironmentType.ULTRA_LIGHT:
            return False, None, (
                f"Ultra-lightweight environment ({avg_step_time*1000:.2f}ms/step, "
                f"{steps_per_second:.0f} steps/sec). Process overhead will dominate. "
                "Use single environment for optimal performance."
            )
        
        elif env_type == EnvironmentType.LIGHT:
            return False, None, (
                f"Lightweight environment ({avg_step_time*1000:.2f}ms/step). "
                "May benefit from vectorization with complex environments only. "
                "Test with 2-4 environments if training is I/O bound."
            )
        
        elif env_type == EnvironmentType.MODERATE:
            return True, 4, (
                f"Moderate complexity environment ({avg_step_time*1000:.2f}ms/step). "
                "Good candidate for vectorization. Start with 4 environments."
            )
        
        elif env_type == EnvironmentType.HEAVY:
            return True, 8, (
                f"Heavy computation environment ({avg_step_time*1000:.2f}ms/step). "
                "Excellent vectorization candidate. Recommended 8 environments."
            )
        
        else:  # ULTRA_HEAVY
            return True, 16, (
                f"Ultra-heavy environment ({avg_step_time*1000:.2f}ms/step). "
                "Excellent vectorization candidate. Consider 16+ environments."
            )


def analyze_environment(env_name: str, **env_kwargs) -> EnvironmentProfile:
    """
    Convenience function to analyze an environment.
    
    Args:
        env_name: Environment name
        **env_kwargs: Environment creation arguments
        
    Returns:
        EnvironmentProfile with analysis results
    """
    analyzer = EnvironmentAnalyzer()
    return analyzer.analyze_environment(env_name, **env_kwargs)


def should_vectorize(env_name: str, **env_kwargs) -> Tuple[bool, str]:
    """
    Quick recommendation on whether to vectorize an environment.
    
    Args:
        env_name: Environment name
        **env_kwargs: Environment creation arguments
        
    Returns:
        Tuple of (should_vectorize, reasoning)
    """
    profile = analyze_environment(env_name, **env_kwargs)
    return profile.vectorization_recommended, profile.reasoning


def print_environment_analysis(profile: EnvironmentProfile, detailed: bool = True):
    """
    Print environment analysis results in a readable format.
    
    Args:
        profile: EnvironmentProfile to display
        detailed: Whether to show detailed statistics
    """
    print(f"\nEnvironment Analysis: {profile.name}")
    print("=" * 50)
    
    if detailed:
        print("Performance Metrics:")
        print(f"  Average step time: {profile.avg_step_time_ms:.3f} ms")
        print(f"  Steps per second: {profile.steps_per_second:.0f}")
        print(f"  Average episode length: {profile.avg_episode_length:.1f}")
        print(f"  Environment type: {profile.environment_type.value}")
        print()
    
    print("Vectorization Recommendation:")
    if profile.vectorization_recommended:
        print("  Use vectorized environments")
        if profile.recommended_num_envs:
            print(f"  Recommended: {profile.recommended_num_envs} environments")
    else:
        print("  Use single environment")
    
    print("\nReasoning:")
    print(f"  {profile.reasoning}")


class EnvironmentBenchmark:
    """
    Benchmark multiple environments to compare their characteristics.
    
    Useful for comparing similar environments or testing modifications.
    """
    
    def __init__(self, analyzer: Optional[EnvironmentAnalyzer] = None):
        """Initialize with optional custom analyzer."""
        self.analyzer = analyzer or EnvironmentAnalyzer()
        self.profiles: List[EnvironmentProfile] = []
    
    def add_environment(self, env_name: str, **env_kwargs) -> EnvironmentProfile:
        """
        Add an environment to the benchmark.
        
        Args:
            env_name: Environment name
            **env_kwargs: Environment creation arguments
            
        Returns:
            EnvironmentProfile for the added environment
        """
        profile = self.analyzer.analyze_environment(env_name, **env_kwargs)
        self.profiles.append(profile)
        return profile
    
    def print_comparison(self):
        """Print comparison of all benchmarked environments."""
        if not self.profiles:
            print("No environments benchmarked yet.")
            return
        
        print(f"\nEnvironment Comparison ({len(self.profiles)} environments)")
        print("=" * 80)
        
        # Header
        print(f"{'Environment':<20} {'Step Time (ms)':<15} {'Steps/sec':<12} {'Type':<12} {'Vectorize':<10}")
        print("-" * 80)
        
        # Sort by steps per second (descending)
        sorted_profiles = sorted(self.profiles, key=lambda p: p.steps_per_second, reverse=True)
        
        for profile in sorted_profiles:
            vectorize_str = "Yes" if profile.vectorization_recommended else "No"
            print(f"{profile.name:<20} {profile.avg_step_time_ms:<15.3f} "
                  f"{profile.steps_per_second:<12.0f} {profile.environment_type.value:<12} {vectorize_str:<10}")
        
        print()
        
        # Summary statistics
        ultra_light = sum(1 for p in self.profiles if p.environment_type == EnvironmentType.ULTRA_LIGHT)
        light = sum(1 for p in self.profiles if p.environment_type == EnvironmentType.LIGHT)
        moderate_plus = len(self.profiles) - ultra_light - light
        
        print("Summary:")
        print(f"  Ultra-light environments: {ultra_light} (single process recommended)")
        print(f"  Light environments: {light} (test vectorization carefully)")
        print(f"  Moderate+ environments: {moderate_plus} (vectorization recommended)")