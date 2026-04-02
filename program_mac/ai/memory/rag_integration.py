"""
RAG integration module
Enhance a system prompt with retrieved memories (no trigger gate — always retrieves).
Supports MemoryCoordinator (recommended) or legacy MemoryManager.
"""
from typing import Any, List, Dict, Optional


class RAGIntegration:
    """RAG integration for prompt enhancement"""

    def __init__(self, memory_backend: Any):
        """
        Args:
            memory_backend: MemoryCoordinator (recommended) or MemoryManager (legacy).
        """
        self.memory_backend = memory_backend
        self.enabled = True

    def enhance_prompt(
        self,
        user_input: str,
        system_prompt: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> str:
        if not self.enabled:
            return system_prompt

        if hasattr(self.memory_backend, "get_rag_memories"):
            memories = self.memory_backend.get_rag_memories(
                user_input,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        else:
            memories = self.memory_backend.get_relevant_memories(
                query_text=user_input,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

        if not memories:
            return system_prompt

        memory_text = self.format_memories_for_prompt(memories)
        return f"""{system_prompt}

相关记忆：
{memory_text}
"""

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        if not memories:
            return ""

        memory_lines = []
        for i, memory in enumerate(memories, 1):
            metadata = memory.get("metadata", {}) or {}
            u = metadata.get("user_input", "")
            a = metadata.get("assistant_response", "")
            memory_lines.append(f"{i}. {u} -> {a}")

        return "\n".join(memory_lines)

    def should_use_rag(self) -> bool:
        return self.enabled

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def get_memory_count(self) -> int:
        if hasattr(self.memory_backend, "rag"):
            return int(self.memory_backend.rag.count())
        return int(self.memory_backend.vector_store.count())
