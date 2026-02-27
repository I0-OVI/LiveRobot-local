"""
Memory System for Rubus
Provides Replay (session-based) and RAG (long-term) memory systems
"""

# New dual memory system
from .memory_coordinator import MemoryCoordinator
from .replay_memory import ReplayMemory
from .rag_memory import RAGMemory
from .memory_summarizer import MemorySummarizer
from .importance_filter import ImportanceFilter
from .importance_calculator import ImportanceCalculator
from .memory_merger import MemoryMerger
from .rag_trigger import RAGTrigger

# Legacy modules (kept for backward compatibility)
from .memory_manager import MemoryManager
from .memory_vector_store import MemoryVectorStore
from .memory_embedder import MemoryEmbedder
from .memory_retriever import MemoryRetriever
from .rag_integration import RAGIntegration

__all__ = [
    # New dual memory system
    'MemoryCoordinator',
    'ReplayMemory',
    'RAGMemory',
    'MemorySummarizer',
    'ImportanceFilter',
    'ImportanceCalculator',
    'MemoryMerger',
    'RAGTrigger',
    # Legacy modules
    'MemoryManager',
    'MemoryVectorStore',
    'MemoryEmbedder',
    'MemoryRetriever',
    'RAGIntegration',
]
