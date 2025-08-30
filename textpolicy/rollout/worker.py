# textpolicy/rollout/worker.py
"""
Worker process management and queue communication.
"""

import multiprocessing as mp
import queue
from typing import Callable, Any, Optional
from .runner import RolloutRunner
from .base import DEFAULT_WORKER_TIMEOUT, DEFAULT_MAX_STEPS
from textpolicy.buffer import Buffer


class RolloutWorker:
    """
    Manages a rollout runner in a separate process.
    
    Handles:
    - Process lifecycle (start, stop, cleanup)
    - Queue-based communication with trainer
    - Data serialization for multiprocessing
    - Async rollout collection
    """
    
    def __init__(
        self,
        env_fn: Callable[[], Any],
        policy_fn: Callable[[], Any],
        strategy: Any,
        max_steps: int = DEFAULT_MAX_STEPS,
        send_queue: Optional[mp.Queue] = None,
    ):
        """
        Initialize worker for separate process execution.

        Args:
            env_fn: Function that returns a fresh environment
            policy_fn: Function that returns a fresh policy (for multiprocessing compatibility)
            strategy: RolloutStrategy instance (e.g., PPOStrategy)
            max_steps: Maximum steps per rollout collection
            send_queue: Queue to send data back to trainer
        """
        self.env_fn = env_fn
        self.policy_fn = policy_fn
        self.strategy = strategy
        self.max_steps = max_steps

        # Communication queues
        self.send_queue: mp.Queue = send_queue if send_queue is not None else mp.Queue()
        self.control_queue: mp.Queue = mp.Queue()  # Commands: "collect", "exit"

        # Process handle and status
        self.process: Optional[mp.Process] = None
        self.is_closed = False  # Track cleanup status

    def run(self):
        """
        Target function for the worker process.
        
        Creates fresh environment and policy instances, then runs collection loop.
        Communicates via queues to avoid shared memory issues.
        """
        # Create fresh instances in worker process (avoid pickle issues)
        env = self.env_fn()
        policy = self.policy_fn()
        runner = RolloutRunner(env, policy, self.strategy, self.max_steps)

        while True:
            try:
                # Wait for commands from trainer
                msg = self.control_queue.get(timeout=DEFAULT_WORKER_TIMEOUT)
                
                if msg == "collect":
                    # Run rollout collection
                    buffer = runner.collect()
                    
                    # Serialize buffer to pure Python types for queue transmission
                    episodes_data = [ep.to_dict() for ep in buffer.episodes]
                    self.send_queue.put(episodes_data)
                    
                elif msg == "exit":
                    break
                    
            except queue.Empty:
                # Timeout - continue listening for commands
                continue

    def start(self):
        """Launch the worker process."""
        if self.process is not None:
            raise RuntimeError("Worker process already started")
            
        self.process = mp.Process(target=self.run, daemon=True)
        self.process.start()

    def collect_async(self):
        """Request a rollout without blocking."""
        if self.process is None:
            raise RuntimeError("Worker process not started")
            
        self.control_queue.put("collect")

    def has_data(self) -> bool:
        """Check if rollout result is ready."""
        return not self.send_queue.empty()

    def get_buffer(self) -> Buffer:
        """
        Retrieve the collected buffer.
        
        Call only after has_data() returns True.
        
        Returns:
            Buffer instance with collected episodes
        """
        episodes_data = self.send_queue.get()
        
        # Reconstruct buffer from serialized data
        buffer = Buffer(max_episodes=len(episodes_data))
        for ep_dict in episodes_data:
            buffer.add_episode_from_dict(ep_dict)
            
        return buffer

    def close(self):
        """Shut down the worker process gracefully."""
        if self.process is None or self.is_closed:
            return
            
        try:
            # Step 1: Signal shutdown to worker process
            self.control_queue.put("exit")
            
            # Step 2: Wait for graceful shutdown
            self.process.join(timeout=3)
            
            # Step 3: Force termination if needed
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            
            # Step 4: Properly drain and close queues to prevent file descriptor errors
            # This prevents the background feeder threads from trying to close already-closed FDs
            self._cleanup_queues()
                
        except Exception:
            # Ignore cleanup errors during testing - they're just noise
            pass
        finally:
            # Step 5: Mark as closed
            self.is_closed = True
            self.process = None

    def _cleanup_queues(self):
        """
        Properly cleanup multiprocessing queues to prevent file descriptor errors.
        
        The issue: Multiprocessing queues have background "feeder" threads that can
        try to close file descriptors after the main process has already closed them.
        
        Solution: Explicitly drain and close queues before process termination.
        """
        try:
            # Cancel the queue's join thread to prevent hanging on exit
            # This tells the queue not to wait for the feeder thread
            if hasattr(self.send_queue, 'cancel_join_thread'):
                self.send_queue.cancel_join_thread()
            if hasattr(self.control_queue, 'cancel_join_thread'):
                self.control_queue.cancel_join_thread()
                
            # Drain any remaining items from queues to prevent deadlock
            # Empty queues close more cleanly
            try:
                while not self.send_queue.empty():
                    self.send_queue.get_nowait()
            except queue.Empty:
                pass  # Queue might be closed already
                
            try:
                while not self.control_queue.empty():
                    self.control_queue.get_nowait()
            except queue.Empty:
                pass  # Queue might be closed already
                
            # Now close the queues properly
            if hasattr(self.send_queue, 'close'):
                self.send_queue.close()
            if hasattr(self.control_queue, 'close'):
                self.control_queue.close()
                
        except Exception:
            # During testing, ignore queue cleanup errors
            # They're artifacts of rapid process termination
            pass 