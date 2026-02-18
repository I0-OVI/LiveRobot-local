"""
Text embedding module for RAG memory system
Uses sentence-transformers for text vectorization
"""
from typing import List, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Please install: pip install sentence-transformers")


class MemoryEmbedder:
    """Text embedding module using sentence-transformers"""
    
    # Default model for Chinese-English support
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    # Alternative Chinese-optimized model
    CHINESE_MODEL = "BAAI/bge-small-zh-v1.5"
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize text embedder
        
        Args:
            model_name: Name of the sentence-transformers model
                       If None, uses DEFAULT_MODEL
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        
        print(f"[Memory] Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[Memory] ✓ Embedding model loaded successfully")
        except Exception as e:
            print(f"[Memory] ⚠ Failed to load {self.model_name}, trying fallback...")
            # Fallback to default model
            if self.model_name != self.DEFAULT_MODEL:
                self.model_name = self.DEFAULT_MODEL
                self.model = SentenceTransformer(self.model_name, device=self.device)
                print(f"[Memory] ✓ Fallback model loaded: {self.DEFAULT_MODEL}")
            else:
                raise RuntimeError(f"Failed to load embedding model: {e}")
        
        # Cache for embeddings (optional, can be enabled for performance)
        self._cache: dict = {}
        self._use_cache = False
    
    def embed_text(self, text: str, use_cache: bool = False) -> List[float]:
        """
        Convert text to embedding vector
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings
        
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector if text is empty
            return [0.0] * self.get_embedding_dim()
        
        # Check cache if enabled
        if use_cache and self._use_cache and text in self._cache:
            return self._cache[text]
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            embedding_list = embedding.tolist()
            
            # Cache if enabled
            if use_cache and self._use_cache:
                self._cache[text] = embedding_list
            
            return embedding_list
        except Exception as e:
            print(f"[Memory] Error embedding text: {e}")
            # Return zero vector on error
            return [0.0] * self.get_embedding_dim()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (batch processing)
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            return [[0.0] * self.get_embedding_dim()] * len(texts)
        
        try:
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            # Pad with zero vectors for empty texts
            result = []
            text_idx = 0
            for text in texts:
                if text and text.strip():
                    result.append(embeddings_list[text_idx])
                    text_idx += 1
                else:
                    result.append([0.0] * self.get_embedding_dim())
            
            return result
        except Exception as e:
            print(f"[Memory] Error embedding batch: {e}")
            # Return zero vectors on error
            return [[0.0] * self.get_embedding_dim()] * len(texts)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embedding vectors
        
        Returns:
            Dimension of the embedding vector
        """
        # Get dimension by encoding a dummy text
        try:
            dummy_embedding = self.model.encode("test", convert_to_numpy=True)
            return dummy_embedding.shape[0]
        except Exception:
            # Default dimensions for common models
            if "MiniLM" in self.model_name:
                return 384
            elif "bge-small" in self.model_name:
                return 512
            else:
                return 384  # Default fallback
    
    def enable_cache(self, max_cache_size: int = 1000):
        """
        Enable embedding cache
        
        Args:
            max_cache_size: Maximum number of cached embeddings
        """
        self._use_cache = True
        self._max_cache_size = max_cache_size
    
    def disable_cache(self):
        """Disable embedding cache"""
        self._use_cache = False
        self._cache.clear()
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._cache.clear()
