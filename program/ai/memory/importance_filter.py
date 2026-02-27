"""
Importance filter module
Filters memories based on importance scores with adaptive thresholds
"""
from typing import Optional


class ImportanceFilter:
    """Importance filter with adaptive threshold"""
    
    def __init__(self, base_threshold: float = 0.5, max_memories: int = 1000):
        """
        Initialize importance filter
        
        Args:
            base_threshold: Base importance threshold (0.0-1.0)
            max_memories: Maximum number of memories (for adaptive threshold calculation)
        """
        self.base_threshold = base_threshold
        self.max_memories = max_memories
    
    def get_adaptive_threshold(self, current_memory_count: int) -> float:
        """
        Get adaptive threshold based on current memory count
        
        Adaptive formula: threshold = base_threshold * (1 - min(memory_count / max_memories, 0.5))
        - When memory_count is low, threshold is close to base_threshold
        - When memory_count approaches max_memories, threshold decreases (but never below base_threshold * 0.5)
        
        Args:
            current_memory_count: Current number of memories in storage
        
        Returns:
            Adaptive threshold value
        """
        # Calculate factor: 0.0 when empty, up to 0.5 when at max
        factor = min(current_memory_count / self.max_memories, 0.5)
        
        # Adaptive threshold: decreases as memory count increases
        adaptive_threshold = self.base_threshold * (1 - factor)
        
        # Ensure threshold never goes below base_threshold * 0.5
        min_threshold = self.base_threshold * 0.5
        return max(adaptive_threshold, min_threshold)
    
    def should_store(self, importance_score: float, current_memory_count: int) -> bool:
        """
        Check if a memory should be stored based on importance
        
        Args:
            importance_score: Importance score of the memory (0.0-1.0)
            current_memory_count: Current number of memories
        
        Returns:
            True if memory should be stored, False otherwise
        """
        threshold = self.get_adaptive_threshold(current_memory_count)
        return importance_score >= threshold
    
    def set_base_threshold(self, threshold: float):
        """
        Update base threshold
        
        Args:
            threshold: New base threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.base_threshold = threshold
        else:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
    
    def set_max_memories(self, max_memories: int):
        """
        Update max memories
        
        Args:
            max_memories: New maximum number of memories
        """
        if max_memories > 0:
            self.max_memories = max_memories
        else:
            raise ValueError(f"Max memories must be positive, got {max_memories}")
