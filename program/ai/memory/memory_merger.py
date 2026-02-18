"""
Memory merger module
Merges near-duplicate memories using clustering strategy
"""
import numpy as np
from typing import List, Dict, Optional

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not installed. Clustering-based merging will be unavailable.")


class MemoryMerger:
    """Memory merger for near-duplicate detection and merging"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize memory merger
        
        Args:
            similarity_threshold: Similarity threshold for considering memories as duplicates (0.0-1.0)
                                 Higher values = stricter (only very similar memories are merged)
        """
        self.similarity_threshold = similarity_threshold
        self.eps = 1.0 - similarity_threshold  # DBSCAN eps parameter
    
    def find_duplicate_clusters(self, memories: List[Dict]) -> List[List[int]]:
        """
        Find clusters of duplicate memories using DBSCAN
        
        Args:
            memories: List of memory dictionaries, each with 'embedding' key
        
        Returns:
            List of clusters, where each cluster is a list of memory indices
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: simple distance-based clustering
            return self._simple_clustering(memories)
        
        if len(memories) < 2:
            return []
        
        # Extract embeddings
        embeddings = np.array([m['embedding'] for m in memories])
        
        # Normalize embeddings for better clustering
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        try:
            # Use DBSCAN to find clusters
            # eps: maximum distance between samples in the same cluster
            # min_samples: minimum number of samples in a cluster
            clustering = DBSCAN(eps=self.eps, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(normalized_embeddings)
            
            # Group indices by cluster label
            clusters = {}
            for idx, label in enumerate(cluster_labels):
                if label != -1:  # -1 means noise (not in any cluster)
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(idx)
            
            return list(clusters.values())
        except Exception as e:
            print(f"[MemoryMerger] Error in DBSCAN clustering: {e}")
            # Fallback to simple clustering
            return self._simple_clustering(memories)
    
    def _simple_clustering(self, memories: List[Dict]) -> List[List[int]]:
        """
        Simple distance-based clustering (fallback when sklearn is not available)
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            List of clusters
        """
        if len(memories) < 2:
            return []
        
        embeddings = np.array([m['embedding'] for m in memories])
        clusters = []
        used = set()
        
        for i in range(len(memories)):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(memories)):
                if j in used:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity >= self.similarity_threshold:
                    cluster.append(j)
                    used.add(j)
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def merge_cluster(self, cluster_memories: List[Dict]) -> Dict:
        """
        Merge a cluster of similar memories into a single memory
        
        Args:
            cluster_memories: List of memory dictionaries in the cluster
        
        Returns:
            Merged memory dictionary
        """
        if not cluster_memories:
            raise ValueError("Cannot merge empty cluster")
        
        if len(cluster_memories) == 1:
            return cluster_memories[0].copy()
        
        # Calculate embedding centroid (weighted average)
        embeddings = np.array([m['embedding'] for m in cluster_memories])
        
        # Weight by importance if available
        importances = np.array([
            m.get('metadata', {}).get('importance', 1.0) for m in cluster_memories
        ])
        weights = importances / importances.sum() if importances.sum() > 0 else np.ones(len(cluster_memories)) / len(cluster_memories)
        
        # Weighted average
        centroid = np.average(embeddings, axis=0, weights=weights)
        
        # Merge metadata
        merged_metadata = {
            'user_input': self._merge_texts([m.get('metadata', {}).get('user_input', '') for m in cluster_memories]),
            'assistant_response': self._merge_texts([m.get('metadata', {}).get('assistant_response', '') for m in cluster_memories]),
            'timestamp': max([m.get('metadata', {}).get('timestamp', '') for m in cluster_memories]),  # Latest timestamp
            'importance': max([m.get('metadata', {}).get('importance', 1.0) for m in cluster_memories]),  # Highest importance
            'tags': list(set([
                tag for m in cluster_memories
                for tag in m.get('metadata', {}).get('tags', [])
            ])),
            'merged_from': len(cluster_memories),  # Number of memories merged
            'is_merged': True
        }
        
        return {
            'embedding': centroid.tolist(),
            'metadata': merged_metadata,
            'document': f"{merged_metadata['user_input']} {merged_metadata['assistant_response']}"
        }
    
    def _merge_texts(self, texts: List[str]) -> str:
        """
        Merge multiple texts, removing duplicates and preserving meaning
        
        Args:
            texts: List of text strings
        
        Returns:
            Merged text
        """
        if not texts:
            return ""
        
        # Remove empty texts
        texts = [t.strip() for t in texts if t.strip()]
        
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # For now, use the longest text (most complete)
        # In the future, could use LLM to merge texts intelligently
        return max(texts, key=len)
    
    def merge_similar_memories(self, memories: List[Dict]) -> List[Dict]:
        """
        Find and merge all similar memories in a list
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            List of memories with duplicates merged
        """
        if len(memories) < 2:
            return memories.copy()
        
        # Find clusters
        clusters = self.find_duplicate_clusters(memories)
        
        if not clusters:
            return memories.copy()
        
        # Merge clusters
        merged_memories = []
        merged_indices = set()
        
        for cluster_indices in clusters:
            cluster_memories = [memories[i] for i in cluster_indices]
            merged_memory = self.merge_cluster(cluster_memories)
            merged_memories.append(merged_memory)
            merged_indices.update(cluster_indices)
        
        # Add non-clustered memories
        for i, memory in enumerate(memories):
            if i not in merged_indices:
                merged_memories.append(memory.copy())
        
        return merged_memories
    
    def set_similarity_threshold(self, threshold: float):
        """
        Update similarity threshold
        
        Args:
            threshold: New similarity threshold (0.0-1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            self.eps = 1.0 - threshold
        else:
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0, got {threshold}")
