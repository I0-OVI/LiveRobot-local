"""
Vector storage module for RAG memory system
Uses ChromaDB for vector storage and retrieval
"""
import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: chromadb not installed. Please install: pip install chromadb")


class MemoryVectorStore:
    """Vector storage for conversation memories using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./memory_db", collection_name: str = "conversation_memories"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the ChromaDB collection
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"[Memory] Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Conversation memories for RAG system"}
            )
            print(f"[Memory] Created new collection: {collection_name}")
    
    def add_memory(
        self,
        memory_id: str,
        user_input: str,
        assistant_response: str,
        embedding: List[float],
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a memory to the vector store
        
        Args:
            memory_id: Unique identifier for the memory
            user_input: User's input text
            assistant_response: Assistant's response text
            embedding: Vector embedding of the text
            metadata: Additional metadata (timestamp, importance, etc.)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Combine user input and response for better retrieval
            combined_text = f"{user_input} {assistant_response}"
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "user_input": user_input,
                "assistant_response": assistant_response,
                "memory_id": memory_id,
                "timestamp": metadata.get("timestamp", datetime.now().isoformat())
            })
            # ChromaDB rejects empty list in metadata; drop any key with value []
            metadata = {k: v for k, v in metadata.items() if not (isinstance(v, list) and len(v) == 0)}
            
            # Add to collection
            self.collection.add(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[metadata]
            )
            
            return True
        except Exception as e:
            print(f"[Memory] Error adding memory: {e}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar memories
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of similar memories with metadata
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            memories = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i, memory_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    # ChromaDB defaults to L2; for normalized vectors: cos_sim = 1 - (L2^2)/2
                    dist_sq = min(distance * distance, 4.0)
                    similarity = max(0.0, 1.0 - dist_sq / 2.0)
                    
                    if similarity >= similarity_threshold:
                        memory = {
                            "id": memory_id,
                            "similarity": similarity,
                            "distance": distance,
                            "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                            "document": results['documents'][0][i] if results['documents'] else ""
                        }
                        memories.append(memory)
            
            return memories
        except Exception as e:
            print(f"[Memory] Error searching memories: {e}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a memory by ID
        
        Args:
            memory_id: ID of the memory to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"[Memory] Error deleting memory: {e}")
            return False
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """
        Get a memory by ID
        
        Args:
            memory_id: ID of the memory
        
        Returns:
            Memory dict or None if not found
        """
        try:
            results = self.collection.get(ids=[memory_id])
            if results['ids'] and len(results['ids']) > 0:
                return {
                    "id": results['ids'][0],
                    "metadata": results['metadatas'][0] if results['metadatas'] else {},
                    "document": results['documents'][0] if results['documents'] else ""
                }
            return None
        except Exception as e:
            print(f"[Memory] Error getting memory: {e}")
            return None
    
    def get_all_memories(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all memories
        
        Args:
            limit: Maximum number of memories to return
        
        Returns:
            List of all memories
        """
        try:
            results = self.collection.get()
            memories = []
            
            if results['ids']:
                for i, memory_id in enumerate(results['ids']):
                    memory = {
                        "id": memory_id,
                        "metadata": results['metadatas'][i] if results['metadatas'] else {},
                        "document": results['documents'][i] if results['documents'] else ""
                    }
                    memories.append(memory)
                    
                    if limit and len(memories) >= limit:
                        break
            
            return memories
        except Exception as e:
            print(f"[Memory] Error getting all memories: {e}")
            return []
    
    def count(self) -> int:
        """Get total number of memories"""
        try:
            return self.collection.count()
        except Exception as e:
            print(f"[Memory] Error counting memories: {e}")
            return 0
    
    def clear(self) -> bool:
        """Clear all memories"""
        try:
            # Delete collection and recreate
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Conversation memories for RAG system"}
            )
            return True
        except Exception as e:
            print(f"[Memory] Error clearing memories: {e}")
            return False
    
    def persist(self):
        """Persist the database (ChromaDB handles this automatically, but we can force it)"""
        # ChromaDB with PersistentClient automatically persists
        # This method is here for API consistency
        pass
    
    def load(self):
        """Load the database (ChromaDB handles this automatically)"""
        # ChromaDB with PersistentClient automatically loads
        # This method is here for API consistency
        pass
