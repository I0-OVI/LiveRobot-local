"""
RAG integration module
Integrates RAG memory system with text generation
"""
from typing import List, Dict, Optional


class RAGIntegration:
    """RAG integration for prompt enhancement"""
    
    def __init__(self, memory_manager):
        """
        Initialize RAG integration
        
        Args:
            memory_manager: MemoryManager instance
        """
        self.memory_manager = memory_manager
        self.enabled = True
    
    def enhance_prompt(
        self,
        user_input: str,
        system_prompt: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> str:
        """
        Enhance prompt with relevant memories
        
        Args:
            user_input: Current user input
            system_prompt: Original system prompt
            top_k: Number of memories to include
            similarity_threshold: Minimum similarity threshold
        
        Returns:
            Enhanced prompt with memories
        """
        if not self.enabled:
            return system_prompt
        
        # Retrieve relevant memories
        memories = self.memory_manager.get_relevant_memories(
            query_text=user_input,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        if not memories:
            # No relevant memories found, return original prompt
            return system_prompt
        
        # Format memories for prompt
        memory_text = self.format_memories_for_prompt(memories)
        
        # Combine with original system prompt
        enhanced_prompt = f"""{system_prompt}

相关记忆：
{memory_text}
"""
        return enhanced_prompt
    
    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        Format memories for inclusion in prompt
        
        Args:
            memories: List of memory dictionaries
        
        Returns:
            Formatted memory text
        """
        if not memories:
            return ""
        
        memory_lines = []
        for i, memory in enumerate(memories, 1):
            metadata = memory.get("metadata", {})
            user_input = metadata.get("user_input", "")
            assistant_response = metadata.get("assistant_response", "")
            similarity = memory.get("similarity", 0.0)
            
            # Format: "1. [用户输入] -> [助手回复]"
            memory_line = f"{i}. {user_input} -> {assistant_response}"
            memory_lines.append(memory_line)
        
        return "\n".join(memory_lines)
    
    def should_use_rag(self) -> bool:
        """
        Check if RAG should be used
        
        Returns:
            True if RAG is enabled and should be used
        """
        return self.enabled
    
    def enable(self):
        """Enable RAG"""
        self.enabled = True
    
    def disable(self):
        """Disable RAG"""
        self.enabled = False
    
    def get_memory_count(self) -> int:
        """Get total number of stored memories"""
        return self.memory_manager.vector_store.count()
