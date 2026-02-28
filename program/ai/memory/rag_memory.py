"""
RAG memory module (refactored)
Long-term memory storage with importance filtering and duplicate merging
"""
import uuid
from typing import List, Dict, Optional
from datetime import datetime

from .memory_vector_store import MemoryVectorStore
from .memory_embedder import MemoryEmbedder
from .memory_retriever import MemoryRetriever
from .importance_filter import ImportanceFilter
from .memory_merger import MemoryMerger
from .query_canonicalizer import canonicalize


class RAGMemory:
    """RAG memory system with importance filtering and duplicate merging"""
    
    def __init__(
        self,
        persist_directory: str = "./memory_db",
        collection_name: str = "conversation_memories",
        embedder_model: Optional[str] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.7,
        use_time_weight: bool = False,
        time_decay_days: int = 30,
        importance_base_threshold: float = 0.5,
        importance_max_memories: int = 1000,
        merge_similarity_threshold: float = 0.95,
        allow_no_memories: bool = True,
        user_name: Optional[str] = None
    ):
        """
        Initialize RAG memory system
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
            embedder_model: Model name for embedder (None for default)
            top_k: Number of top memories to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            use_time_weight: Whether to weight memories by recency
            time_decay_days: Days for time decay
            importance_base_threshold: Base importance threshold
            importance_max_memories: Maximum memories for adaptive threshold
            merge_similarity_threshold: Similarity threshold for merging duplicates
            allow_no_memories: Allow returning empty results if no valid memories
            user_name: User name from setup.txt for canonicalization (e.g. Carambola -> 用户 in retrieval)
        """
        # Initialize core components
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
            time_decay_days=time_decay_days,
            user_name=user_name
        )
        
        # Initialize importance filter
        self.importance_filter = ImportanceFilter(
            base_threshold=importance_base_threshold,
            max_memories=importance_max_memories
        )
        
        # Initialize memory merger
        self.memory_merger = MemoryMerger(
            similarity_threshold=merge_similarity_threshold
        )
        
        self.allow_no_memories = allow_no_memories
        self.persist_directory = persist_directory
        self.user_name = user_name
    
    def add_memory_with_importance_check(
        self,
        user_input: str,
        assistant_response: str,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
        auto_merge: bool = True
    ) -> Optional[str]:
        """
        Add memory with importance filtering
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization
            auto_merge: Whether to automatically merge duplicates after adding
        
        Returns:
            Memory ID if stored, None if filtered out
        """
        # Check importance threshold
        current_count = self.vector_store.count()
        if not self.importance_filter.should_store(importance, current_count):
            print(f"[RAG] Memory filtered out by importance (score: {importance:.2f}, threshold: {self.importance_filter.get_adaptive_threshold(current_count):.2f})")
            return None
        
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Create embedding (canonicalize for consistent retrieval across 你/我/用户 and user_name)
        combined_text = f"{user_input} {assistant_response}"
        canonical_text = canonicalize(combined_text, user_name=self.user_name)
        embedding = self.embedder.embed_text(canonical_text)
        
        # Prepare metadata (ChromaDB rejects empty list values, so omit tags when empty)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "user_input": user_input,
            "assistant_response": assistant_response
        }
        if tags and len(tags) > 0:
            metadata["tags"] = tags
        
        # Save to vector store
        success = self.vector_store.add_memory(
            memory_id=memory_id,
            user_input=user_input,
            assistant_response=assistant_response,
            embedding=embedding,
            metadata=metadata
        )
        
        if not success:
            raise RuntimeError("Failed to save memory to vector store")
        
        print(f"[RAG] Saved memory (ID: {memory_id[:8]}..., importance: {importance:.2f})")
        
        # Auto-merge duplicates if enabled
        if auto_merge:
            try:
                self.merge_similar_memories()
            except Exception as e:
                print(f"[RAG] Warning: Auto-merge failed: {e}")
        
        return memory_id
    
    def merge_similar_memories(self) -> int:
        """
        Find and merge similar memories
        
        Returns:
            Number of memories merged
        """
        # Get all memories
        all_memories = self.vector_store.get_all_memories()
        
        if len(all_memories) < 2:
            return 0
        
        # Convert to format expected by merger
        # Note: ChromaDB doesn't expose embeddings directly, so we re-embed documents
        # This is acceptable since merge is not called frequently
        memories_for_merging = []
        for memory in all_memories:
            memory_id = memory.get("id")
            metadata = memory.get("metadata", {})
            document = memory.get("document", "")
            
            # Re-embed the document for clustering
            # This is necessary because ChromaDB doesn't expose stored embeddings via get()
            if document:
                embedding = self.embedder.embed_text(document)
                memories_for_merging.append({
                    "id": memory_id,
                    "embedding": embedding,
                    "metadata": metadata,
                    "document": document
                })
        
        if len(memories_for_merging) < 2:
            return 0
        
        # Find clusters
        clusters = self.memory_merger.find_duplicate_clusters(memories_for_merging)
        
        if not clusters:
            return 0
        
        merged_count = 0
        
        # Process each cluster
        for cluster_indices in clusters:
            if len(cluster_indices) < 2:
                continue
            
            cluster_memories = [memories_for_merging[i] for i in cluster_indices]
            
            # Merge cluster
            merged_memory = self.memory_merger.merge_cluster(cluster_memories)
            
            # Delete original memories
            for memory in cluster_memories:
                self.vector_store.delete_memory(memory["id"])
                merged_count += 1
            
            # Add merged memory
            merged_id = str(uuid.uuid4())
            merged_metadata = merged_memory["metadata"]
            
            self.vector_store.add_memory(
                memory_id=merged_id,
                user_input=merged_metadata.get("user_input", ""),
                assistant_response=merged_metadata.get("assistant_response", ""),
                embedding=merged_memory["embedding"],
                metadata=merged_metadata
            )
        
        if merged_count > 0:
            print(f"[RAG] Merged {merged_count} duplicate memories into {len(clusters)} merged memories")
        
        return merged_count
    
    def get_relevant_memories(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Get relevant memories for a query (allows no valid memories)
        
        Args:
            query_text: Query text
            top_k: Number of results (overrides default)
            similarity_threshold: Minimum similarity (overrides default)
        
        Returns:
            List of relevant memories (may be empty if allow_no_memories=True)
        """
        memories = self.retriever.retrieve(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # If no memories found and allow_no_memories is True, return empty list
        if not memories and self.allow_no_memories:
            return []
        
        return memories
    
    def save_summary(self, summary_text: str, importance: float = 0.8) -> Optional[str]:
        """
        Save a summary as a memory
        
        Args:
            summary_text: Summary text
            importance: Importance score for summary (default: 0.8, higher than regular memories)
        
        Returns:
            Memory ID if stored, None if filtered out
        """
        # Summaries are stored as special memories
        # Format: "Summary: [summary_text]"
        user_input = "Summary"
        assistant_response = summary_text
        
        return self.add_memory_with_importance_check(
            user_input=user_input,
            assistant_response=assistant_response,
            importance=importance,
            tags=["summary"],
            auto_merge=False  # Don't auto-merge summaries
        )
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about stored memories"""
        all_memories = self.vector_store.get_all_memories()
        total_count = len(all_memories)
        
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
            "current_threshold": self.importance_filter.get_adaptive_threshold(total_count)
        }
        
        return stats
    
    def persist(self):
        """Persist the memory database"""
        self.vector_store.persist()
    
    def load(self):
        """Load the memory database"""
        self.vector_store.load()
    
    def clear_all_memories(self) -> bool:
        """Clear all memories"""
        return self.vector_store.clear()
    
    def count(self) -> int:
        """Get total number of memories"""
        return self.vector_store.count()
