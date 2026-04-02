"""
Memory manager module for RAG memory system
Manages the complete lifecycle of memories
"""
import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class MemoryManager:
    """Main memory manager for RAG system"""
    
    def __init__(
        self,
        persist_directory: str = "./memory_db",
        collection_name: str = "conversation_memories",
        embedder_model: Optional[str] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.7,
        use_time_weight: bool = False,
        time_decay_days: int = 30
    ):
        """
        Initialize memory manager
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
            embedder_model: Model name for embedder (None for default)
            top_k: Number of top memories to retrieve
            similarity_threshold: Minimum similarity score
            use_time_weight: Whether to weight memories by recency
            time_decay_days: Days for time decay
        """
        from .memory_vector_store import MemoryVectorStore
        from .memory_embedder import MemoryEmbedder
        from .memory_retriever import MemoryRetriever
        
        # Initialize components
        self.vector_store = MemoryVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        self.embedder = MemoryEmbedder(model_name=embedder_model)
        
        self.retriever = MemoryRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            use_time_weight=use_time_weight,
            time_decay_days=time_decay_days
        )
        
        self.persist_directory = persist_directory
    
    def save_conversation(
        self,
        user_input: str,
        assistant_response: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Save a conversation to memory
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization
        
        Returns:
            Memory ID
        """
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Create embedding for the conversation
        # Combine user input and response for better retrieval
        combined_text = f"{user_input} {assistant_response}"
        embedding = self.embedder.embed_text(combined_text)
        
        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "tags": tags or [],
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        
        # Save to vector store
        success = self.vector_store.add_memory(
            memory_id=memory_id,
            user_input=user_input,
            assistant_response=assistant_response,
            embedding=embedding,
            metadata=metadata
        )
        
        if success:
            print(f"[Memory] Saved conversation to memory (ID: {memory_id[:8]}...)")
            return memory_id
        else:
            raise RuntimeError("Failed to save conversation to memory")
    
    def get_relevant_memories(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Get relevant memories for a query
        
        Args:
            query_text: Query text
            top_k: Number of results (overrides default)
            similarity_threshold: Minimum similarity (overrides default)
        
        Returns:
            List of relevant memories
        """
        return self.retriever.retrieve(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    
    def cleanup_old_memories(
        self,
        days: Optional[int] = None,
        max_memories: Optional[int] = None,
        min_importance: float = 0.0
    ) -> int:
        """
        Clean up old or low-importance memories
        
        Args:
            days: Keep only memories from the last N days
            max_memories: Keep only the top N most important memories
            min_importance: Minimum importance score to keep
        
        Returns:
            Number of memories deleted
        """
        all_memories = self.vector_store.get_all_memories()
        deleted_count = 0
        now = datetime.now()
        
        for memory in all_memories:
            metadata = memory.get("metadata", {})
            timestamp_str = metadata.get("timestamp")
            importance = metadata.get("importance", 0.0)
            memory_id = memory.get("id")
            
            should_delete = False
            
            # Check age
            if days is not None and timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if (now - timestamp).days > days:
                        should_delete = True
                except Exception:
                    pass
            
            # Check importance
            if importance < min_importance:
                should_delete = True
            
            if should_delete and memory_id:
                if self.vector_store.delete_memory(memory_id):
                    deleted_count += 1
        
        # If max_memories is set, keep only the most important ones
        if max_memories is not None:
            remaining = self.vector_store.get_all_memories()
            if len(remaining) > max_memories:
                # Sort by importance and timestamp
                remaining.sort(
                    key=lambda m: (
                        m.get("metadata", {}).get("importance", 0.0),
                        m.get("metadata", {}).get("timestamp", "")
                    ),
                    reverse=True
                )
                
                # Delete the least important ones
                to_delete = remaining[max_memories:]
                for memory in to_delete:
                    memory_id = memory.get("id")
                    if memory_id and self.vector_store.delete_memory(memory_id):
                        deleted_count += 1
        
        if deleted_count > 0:
            print(f"[Memory] Cleaned up {deleted_count} memories")
        
        return deleted_count
    
    def get_memory_stats(self) -> Dict:
        """
        Get statistics about stored memories
        
        Returns:
            Dictionary with memory statistics
        """
        all_memories = self.vector_store.get_all_memories()
        total_count = len(all_memories)
        
        # Calculate statistics
        importances = [
            m.get("metadata", {}).get("importance", 0.0)
            for m in all_memories
        ]
        
        timestamps = []
        for m in all_memories:
            ts_str = m.get("metadata", {}).get("timestamp")
            if ts_str:
                try:
                    timestamps.append(datetime.fromisoformat(ts_str))
                except Exception:
                    pass
        
        stats = {
            "total_memories": total_count,
            "average_importance": sum(importances) / len(importances) if importances else 0.0,
            "oldest_memory": min(timestamps).isoformat() if timestamps else None,
            "newest_memory": max(timestamps).isoformat() if timestamps else None,
        }
        
        return stats
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory
        
        Args:
            memory_id: ID of the memory to delete
        
        Returns:
            True if successful
        """
        return self.vector_store.delete_memory(memory_id)
    
    def clear_all_memories(self) -> bool:
        """Clear all memories"""
        return self.vector_store.clear()
    
    def persist(self):
        """Persist the memory database"""
        self.vector_store.persist()
    
    def load(self):
        """Load the memory database"""
        self.vector_store.load()
