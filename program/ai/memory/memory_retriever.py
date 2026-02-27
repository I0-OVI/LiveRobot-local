"""
Retrieval module for RAG memory system
Handles memory retrieval with various strategies
"""
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math


class MemoryRetriever:
    """Memory retrieval module with various retrieval strategies"""
    
    def __init__(
        self,
        vector_store,
        embedder,
        top_k: int = 3,
        similarity_threshold: float = 0.7,
        use_time_weight: bool = False,
        time_decay_days: int = 30
    ):
        """
        Initialize memory retriever
        
        Args:
            vector_store: MemoryVectorStore instance
            embedder: MemoryEmbedder instance
            top_k: Number of top memories to retrieve
            similarity_threshold: Minimum similarity score (0.0-1.0)
            use_time_weight: Whether to weight memories by recency
            time_decay_days: Days for time decay (recent memories weighted higher)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_time_weight = use_time_weight
        self.time_decay_days = time_decay_days
    
    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant memories for a query
        
        Args:
            query_text: Query text to search for
            top_k: Number of results (overrides instance default)
            similarity_threshold: Minimum similarity (overrides instance default)
        
        Returns:
            List of relevant memories
        """
        if not query_text or not query_text.strip():
            return []
        
        # Use provided parameters or instance defaults
        top_k = top_k if top_k is not None else self.top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        
        # Embed the query
        query_embedding = self.embedder.embed_text(query_text)
        
        # Search in vector store
        memories = self.vector_store.search_similar(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more results for filtering/weighting
            similarity_threshold=0.0  # Don't filter here, we'll do it after weighting
        )

        # Recompute similarity with cosine directly from text embeddings.
        # This avoids metric mismatch when Chroma distance space/config differs.
        if memories:
            for memory in memories:
                document = memory.get("document", "")
                if document:
                    doc_embedding = self.embedder.embed_text(document)
                    memory["similarity"] = self._cosine_similarity(query_embedding, doc_embedding)
        
        # Apply time weighting if enabled
        if self.use_time_weight and memories:
            memories = self._apply_time_weighting(memories)
        
        # Filter by threshold and limit to top_k
        filtered_memories = [
            mem for mem in memories
            if mem.get("similarity", 0.0) >= threshold
        ]
        
        # Sort by similarity (descending) and take top_k
        filtered_memories.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return filtered_memories[:top_k]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity and clamp to [0, 1] for thresholding."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot = 0.0
        norm_a_sq = 0.0
        norm_b_sq = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a_sq += a * a
            norm_b_sq += b * b

        denom = math.sqrt(norm_a_sq) * math.sqrt(norm_b_sq)
        if denom <= 1e-12:
            return 0.0

        cos = dot / denom
        if cos < 0.0:
            return 0.0
        if cos > 1.0:
            return 1.0
        return cos
    
    def retrieve_with_score(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve memories with similarity scores
        
        Args:
            query_text: Query text to search for
            top_k: Number of results
            similarity_threshold: Minimum similarity
        
        Returns:
            List of tuples (memory_dict, similarity_score)
        """
        memories = self.retrieve(query_text, top_k, similarity_threshold)
        return [(mem, mem.get("similarity", 0.0)) for mem in memories]
    
    def _apply_time_weighting(self, memories: List[Dict]) -> List[Dict]:
        """
        Apply time-based weighting to memories (recent memories weighted higher)
        
        Args:
            memories: List of memory dicts
        
        Returns:
            List of memories with adjusted similarity scores
        """
        now = datetime.now()
        
        for memory in memories:
            metadata = memory.get("metadata", {})
            timestamp_str = metadata.get("timestamp")
            
            if timestamp_str:
                try:
                    # Parse timestamp
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str)
                    else:
                        timestamp = timestamp_str
                    
                    # Calculate days since memory was created
                    days_ago = (now - timestamp).days
                    
                    # Calculate time weight (exponential decay)
                    # Memories within time_decay_days get full weight
                    # Older memories get reduced weight
                    if days_ago <= self.time_decay_days:
                        time_weight = 1.0
                    else:
                        # Exponential decay beyond time_decay_days
                        decay_factor = 0.5  # 50% weight after time_decay_days
                        time_weight = decay_factor ** ((days_ago - self.time_decay_days) / self.time_decay_days)
                    
                    # Adjust similarity score
                    original_similarity = memory.get("similarity", 0.0)
                    memory["similarity"] = original_similarity * time_weight
                    memory["time_weight"] = time_weight
                except Exception as e:
                    # If timestamp parsing fails, use original similarity
                    print(f"[Memory] Error applying time weight: {e}")
                    memory["time_weight"] = 1.0
        
        return memories
    
    def set_retrieval_params(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        use_time_weight: Optional[bool] = None,
        time_decay_days: Optional[int] = None
    ):
        """
        Update retrieval parameters
        
        Args:
            top_k: Number of top memories to retrieve
            similarity_threshold: Minimum similarity score
            use_time_weight: Whether to use time weighting
            time_decay_days: Days for time decay
        """
        if top_k is not None:
            self.top_k = top_k
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        if use_time_weight is not None:
            self.use_time_weight = use_time_weight
        if time_decay_days is not None:
            self.time_decay_days = time_decay_days
