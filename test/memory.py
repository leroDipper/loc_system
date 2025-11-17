import psutil
import os

class MemoryMonitor:
    """Simple memory usage tracker."""
    
    @staticmethod
    def get_memory_mb():
        """Get current process memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def print_memory(label=""):
        """Print current memory usage with label."""
        mem_mb = MemoryMonitor.get_memory_mb()
        print(f"Memory [{label}]: {mem_mb:.2f} MB")