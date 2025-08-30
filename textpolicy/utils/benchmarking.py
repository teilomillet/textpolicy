"""
Performance benchmarking utilities for MLX-RL.

This module provides tools for comprehensive performance analysis, comparing
different configurations, and generating detailed performance reports.
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

try:
    import mlx.core as mx  # type: ignore
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


from .timing import Timer, benchmark_function
from .debug import debug_print
from .environment import EnvironmentAnalyzer, EnvironmentProfile


class BenchmarkType(Enum):
    """Types of benchmarks that can be performed."""
    ENVIRONMENT = "environment"
    TRAINING = "training"
    POLICY = "policy"
    VECTORIZATION = "vectorization"
    MEMORY = "memory"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    benchmark_type: BenchmarkType
    duration_seconds: float
    iterations: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def avg_time_per_iteration(self) -> float:
        """Average time per iteration in seconds."""
        return self.duration_seconds / max(1, self.iterations)
    
    @property
    def iterations_per_second(self) -> float:
        """Number of iterations per second."""
        return self.iterations / max(0.001, self.duration_seconds)


@dataclass
class ComparisonResult:
    """Results from comparing multiple benchmarks."""
    baseline: BenchmarkResult
    comparisons: List[BenchmarkResult]
    
    def get_speedup(self, result: BenchmarkResult) -> float:
        """Calculate speedup compared to baseline."""
        if self.baseline.avg_time_per_iteration == 0:
            return 1.0
        return self.baseline.avg_time_per_iteration / result.avg_time_per_iteration
    
    def get_throughput_ratio(self, result: BenchmarkResult) -> float:
        """Calculate throughput ratio compared to baseline."""
        if self.baseline.iterations_per_second == 0:
            return 1.0
        return result.iterations_per_second / self.baseline.iterations_per_second


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking system for MLX-RL.
    
    Features:
    - Environment performance comparison
    - Vectorization vs single environment benchmarks
    - Training pipeline performance analysis
    - Statistical significance testing
    - Detailed reporting and visualization
    """
    
    def __init__(self, warmup_iterations: int = 10, min_duration: float = 1.0):
        """
        Initialize performance benchmark.
        
        Args:
            warmup_iterations: Number of warmup iterations before timing
            min_duration: Minimum benchmark duration in seconds
        """
        self.warmup_iterations = warmup_iterations
        self.min_duration = min_duration
        self.timer = Timer()
        self.results: List[BenchmarkResult] = []
        
    def benchmark_environment_speed(self, env_name: str, iterations: int = 1000, **env_kwargs) -> BenchmarkResult:
        """
        Benchmark environment step performance.
        
        Args:
            env_name: Environment name
            iterations: Number of steps to benchmark
            **env_kwargs: Environment creation arguments
            
        Returns:
            BenchmarkResult with environment performance data
        """
        debug_print(f"Benchmarking environment speed: {env_name}", "benchmarking")
        
        try:
            from textpolicy.environment import GymAdapter
            
            env = GymAdapter(env_name, **env_kwargs)
            
            # Warmup
            obs, info = env.reset()
            for _ in range(self.warmup_iterations):
                action = env.action_space.sample()
                step_result = env.step(action)
                if step_result["terminated"] or step_result["truncated"]:
                    obs, info = env.reset()
            
            # Benchmark
            obs, info = env.reset()
            episode_lengths = []
            current_episode_length = 0
            
            start_time = time.perf_counter()
            for i in range(iterations):
                action = env.action_space.sample()
                step_result = env.step(action)
                current_episode_length += 1
                
                if step_result["terminated"] or step_result["truncated"]:
                    episode_lengths.append(current_episode_length)
                    current_episode_length = 0
                    obs, info = env.reset()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            env.close()
            
            # Calculate metrics
            avg_episode_length = statistics.mean(episode_lengths) if episode_lengths else current_episode_length
            steps_per_second = iterations / duration
            
            result = BenchmarkResult(
                name=f"{env_name}_speed",
                benchmark_type=BenchmarkType.ENVIRONMENT,
                duration_seconds=duration,
                iterations=iterations,
                metrics={
                    "steps_per_second": steps_per_second,
                    "avg_step_time_ms": (duration / iterations) * 1000,
                    "avg_episode_length": avg_episode_length,
                    "episodes_completed": len(episode_lengths)
                },
                metadata={"env_name": env_name, "env_kwargs": env_kwargs}
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            debug_print(f"Error benchmarking environment {env_name}: {e}", "benchmarking")
            raise
    
    def benchmark_vectorization_comparison(self, env_name: str, num_envs: int = 4, 
                                         iterations: int = 1000, **env_kwargs) -> ComparisonResult:
        """
        Compare single vs vectorized environment performance.
        
        Args:
            env_name: Environment name
            num_envs: Number of vectorized environments
            iterations: Number of steps per environment
            **env_kwargs: Environment creation arguments
            
        Returns:
            ComparisonResult with single vs vectorized performance
        """
        debug_print(f"Benchmarking vectorization: {env_name} (1 vs {num_envs} envs)", "benchmarking")
        
        # Benchmark single environment
        single_result = self.benchmark_environment_speed(env_name, iterations, **env_kwargs)
        single_result.name = f"{env_name}_single"
        
        # Benchmark vectorized environments
        try:
            from textpolicy.environment import make_vectorized_env
            
            vectorized_env = make_vectorized_env(env_name, num_envs=num_envs, env_kwargs=env_kwargs)
            
            # Warmup
            obs, infos = vectorized_env.reset()
            for _ in range(self.warmup_iterations):
                # Sample actions for each environment and create batch
                action_samples = [vectorized_env.action_space.sample() for _ in range(num_envs)]
                actions = mx.array(action_samples)
                step_results = vectorized_env.step(actions)
                # Check if any environments are done
                terminated = step_results["terminated"] 
                truncated = step_results["truncated"]
                if mx.any(terminated | truncated):
                    obs, infos = vectorized_env.reset()
            
            # Benchmark
            obs, infos = vectorized_env.reset()
            start_time = time.perf_counter()
            
            for i in range(iterations):
                # Sample actions for each environment and create batch
                action_samples = [vectorized_env.action_space.sample() for _ in range(num_envs)]
                actions = mx.array(action_samples)
                step_results = vectorized_env.step(actions)
                # Check if any environments are done
                terminated = step_results["terminated"]
                truncated = step_results["truncated"] 
                if mx.any(terminated | truncated):
                    obs, infos = vectorized_env.reset()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            vectorized_env.close()
            
            # Total steps = iterations * num_envs
            total_steps = iterations * num_envs
            steps_per_second = total_steps / duration
            
            vectorized_result = BenchmarkResult(
                name=f"{env_name}_vectorized_{num_envs}",
                benchmark_type=BenchmarkType.VECTORIZATION,
                duration_seconds=duration,
                iterations=total_steps,
                metrics={
                    "steps_per_second": steps_per_second,
                    "avg_step_time_ms": (duration / total_steps) * 1000,
                    "num_envs": num_envs,
                    "parallelization_efficiency": steps_per_second / (single_result.metrics["steps_per_second"] * num_envs)
                },
                metadata={"env_name": env_name, "num_envs": num_envs, "env_kwargs": env_kwargs}
            )
            
            self.results.append(vectorized_result)
            
            return ComparisonResult(baseline=single_result, comparisons=[vectorized_result])
            
        except Exception as e:
            debug_print(f"Error benchmarking vectorized environment: {e}", "benchmarking")
            # Return comparison with just single environment
            return ComparisonResult(baseline=single_result, comparisons=[])
    
    def benchmark_function_performance(self, name: str, func: Callable, *args, 
                                     iterations: int = 100, **kwargs) -> BenchmarkResult:
        """
        Benchmark arbitrary function performance.
        
        Args:
            name: Benchmark name
            func: Function to benchmark
            *args: Function positional arguments
            iterations: Number of iterations
            **kwargs: Function keyword arguments
            
        Returns:
            BenchmarkResult with function performance data
        """
        stats = benchmark_function(func, *args, iterations=iterations, 
                                 warmup=self.warmup_iterations, **kwargs)
        
        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.TRAINING,
            duration_seconds=stats["total"],
            iterations=iterations,
            metrics={
                "avg_time_ms": stats["mean"] * 1000,
                "min_time_ms": stats["min"] * 1000,
                "max_time_ms": stats["max"] * 1000,
                "std_time_ms": 0.0  # Would need to calculate from raw data
            }
        )
        
        self.results.append(result)
        return result
    
    def compare_environments(self, env_names: List[str], iterations: int = 1000, 
                           **shared_env_kwargs) -> List[BenchmarkResult]:
        """
        Compare performance across multiple environments.
        
        Args:
            env_names: List of environment names to compare
            iterations: Number of steps per environment
            **shared_env_kwargs: Shared environment creation arguments
            
        Returns:
            List of BenchmarkResults sorted by performance
        """
        debug_print(f"Comparing {len(env_names)} environments", "benchmarking")
        
        results = []
        for env_name in env_names:
            try:
                result = self.benchmark_environment_speed(env_name, iterations, **shared_env_kwargs)
                results.append(result)
            except Exception as e:
                debug_print(f"Failed to benchmark {env_name}: {e}", "benchmarking")
                continue
        
        # Sort by steps per second (descending)
        results.sort(key=lambda r: r.metrics.get("steps_per_second", 0), reverse=True)
        return results
    
    def get_results(self, benchmark_type: Optional[BenchmarkType] = None) -> List[BenchmarkResult]:
        """
        Get benchmark results, optionally filtered by type.
        
        Args:
            benchmark_type: Optional filter by benchmark type
            
        Returns:
            List of BenchmarkResults
        """
        if benchmark_type is None:
            return self.results.copy()
        return [r for r in self.results if r.benchmark_type == benchmark_type]
    
    def clear_results(self):
        """Clear all benchmark results."""
        self.results.clear()
    
    def print_results(self, benchmark_type: Optional[BenchmarkType] = None, detailed: bool = True):
        """
        Print benchmark results in a readable format.
        
        Args:
            benchmark_type: Optional filter by benchmark type
            detailed: Whether to show detailed metrics
        """
        results = self.get_results(benchmark_type)
        
        if not results:
            print("No benchmark results available.")
            return
        
        print(f"\nBenchmark Results ({len(results)} tests)")
        print("=" * 70)
        
        if detailed:
            for result in results:
                print(f"\n{result.name}")
                print(f"  Type: {result.benchmark_type.value}")
                print(f"  Duration: {result.duration_seconds:.3f}s")
                print(f"  Iterations: {result.iterations}")
                print(f"  Avg time/iteration: {result.avg_time_per_iteration*1000:.3f}ms")
                print(f"  Iterations/second: {result.iterations_per_second:.0f}")
                
                if result.metrics:
                    print("  Metrics:")
                    for key, value in result.metrics.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.3f}")
                        else:
                            print(f"    {key}: {value}")
        else:
            # Compact table format
            print(f"{'Name':<25} {'Type':<12} {'Time/iter (ms)':<15} {'Iter/sec':<12}")
            print("-" * 70)
            for result in results:
                print(f"{result.name:<25} {result.benchmark_type.value:<12} "
                      f"{result.avg_time_per_iteration*1000:<15.3f} {result.iterations_per_second:<12.0f}")
    
    def print_comparison(self, comparison: ComparisonResult):
        """
        Print detailed comparison results.
        
        Args:
            comparison: ComparisonResult to display
        """
        print("\nPerformance Comparison")
        print("=" * 50)
        
        baseline = comparison.baseline
        print(f"Baseline: {baseline.name}")
        print(f"  {baseline.iterations_per_second:.0f} iterations/sec")
        print(f"  {baseline.avg_time_per_iteration*1000:.3f} ms/iteration")
        
        print("\nComparisons:")
        for result in comparison.comparisons:
            speedup = comparison.get_speedup(result)
            throughput_ratio = comparison.get_throughput_ratio(result)
            
            print(f"  {result.name}:")
            print(f"    {result.iterations_per_second:.0f} iterations/sec ({throughput_ratio:.2f}x throughput)")
            print(f"    {result.avg_time_per_iteration*1000:.3f} ms/iteration ({speedup:.2f}x speedup)")
            
            if speedup > 1.1:
                print(f"    {speedup:.1f}x faster than baseline")
            elif speedup < 0.9:
                print(f"    {1/speedup:.1f}x slower than baseline")
            else:
                print("    Similar performance to baseline")


def quick_environment_benchmark(env_name: str, iterations: int = 1000, **env_kwargs) -> EnvironmentProfile:
    """
    Quick benchmark and analysis of a single environment.
    
    Args:
        env_name: Environment name
        iterations: Number of steps to benchmark
        **env_kwargs: Environment creation arguments
        
    Returns:
        EnvironmentProfile with analysis and recommendations
    """
    # Use EnvironmentAnalyzer for detailed analysis
    analyzer = EnvironmentAnalyzer(test_steps=iterations)
    return analyzer.analyze_environment(env_name, **env_kwargs)


def quick_vectorization_test(env_name: str, num_envs_list: List[int] = [1, 2, 4, 8], 
                           iterations: int = 500, **env_kwargs) -> Dict[int, BenchmarkResult]:
    """
    Quick test of vectorization performance across different environment counts.
    
    Args:
        env_name: Environment name
        num_envs_list: List of environment counts to test
        iterations: Number of steps per test
        **env_kwargs: Environment creation arguments
        
    Returns:
        Dictionary mapping num_envs to BenchmarkResult
    """
    benchmark = PerformanceBenchmark()
    results = {}
    
    for num_envs in num_envs_list:
        if num_envs == 1:
            result = benchmark.benchmark_environment_speed(env_name, iterations, **env_kwargs)
        else:
            comparison = benchmark.benchmark_vectorization_comparison(
                env_name, num_envs=num_envs, iterations=iterations//num_envs, **env_kwargs
            )
            result = comparison.comparisons[0] if comparison.comparisons else None
        
        if result:
            results[num_envs] = result
    
    return results


@contextmanager
def benchmark_context(name: str, benchmark: Optional[PerformanceBenchmark] = None):
    """
    Context manager for benchmarking code blocks.
    
    Args:
        name: Benchmark name
        benchmark: PerformanceBenchmark instance (creates new if None)
        
    Example:
        with benchmark_context("training_step") as bench:
            loss = trainer.update(batch)
    """
    bench = benchmark or PerformanceBenchmark()
    start_time = time.perf_counter()
    
    try:
        yield bench
    finally:
        duration = time.perf_counter() - start_time
        result = BenchmarkResult(
            name=name,
            benchmark_type=BenchmarkType.TRAINING,
            duration_seconds=duration,
            iterations=1,
            metrics={"duration_ms": duration * 1000}
        )
        bench.results.append(result)