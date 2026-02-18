"""
Memory coordinator module
Coordinates Replay and RAG memory systems
"""
import threading
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .replay_memory import ReplayMemory
from .rag_memory import RAGMemory
from .memory_summarizer import MemorySummarizer
from .importance_calculator import ImportanceCalculator
from .rag_trigger import RAGTrigger


class MemoryCoordinator:
    """Coordinates Replay and RAG memory systems"""
    
    def __init__(
        self,
        replay_token_budget: int = 2000,
        replay_persist_sessions: bool = True,
        replay_persist_path: str = "./replay_db",
        rag_persist_directory: str = "./memory_db",
        rag_collection_name: str = "conversation_memories",
        rag_embedder_model: Optional[str] = None,
        rag_top_k: int = 3,
        rag_similarity_threshold: float = 0.7,
        rag_summary_interval: int = 10,
        rag_importance_base_threshold: float = 0.5,
        rag_importance_max_memories: int = 1000,
        rag_merge_similarity_threshold: float = 0.95,
        rag_allow_no_memories: bool = True,
        rag_use_llm_trigger: bool = False,
        text_generator=None
    ):
        """
        Initialize memory coordinator
        
        Args:
            replay_token_budget: Token budget for Replay system
            replay_persist_sessions: Whether to persist Replay sessions
            replay_persist_path: Path for Replay persistence
            rag_persist_directory: Directory for RAG persistence
            rag_collection_name: ChromaDB collection name
            rag_embedder_model: Embedder model name
            rag_top_k: Top-k for RAG retrieval
            rag_similarity_threshold: Similarity threshold for RAG
            rag_summary_interval: Turns between summaries
            rag_importance_base_threshold: Base importance threshold
            rag_importance_max_memories: Max memories for adaptive threshold
            rag_merge_similarity_threshold: Similarity threshold for merging
            rag_allow_no_memories: Allow no valid memories in RAG
            rag_use_llm_trigger: If True, use LLM to decide when to use RAG (can block ~2–5s before first token). If False, only keyword-based trigger for smoother voice.
            text_generator: Text generator for summary generation
        """
        # Initialize Replay system
        self.replay = ReplayMemory(
            token_budget=replay_token_budget,
            persist_sessions=replay_persist_sessions,
            persist_path=replay_persist_path
        )
        
        # Initialize RAG system
        self.rag = RAGMemory(
            persist_directory=rag_persist_directory,
            collection_name=rag_collection_name,
            embedder_model=rag_embedder_model,
            top_k=rag_top_k,
            similarity_threshold=rag_similarity_threshold,
            importance_base_threshold=rag_importance_base_threshold,
            importance_max_memories=rag_importance_max_memories,
            merge_similarity_threshold=rag_merge_similarity_threshold,
            allow_no_memories=rag_allow_no_memories
        )
        
        # Initialize summarizer
        self.summarizer = MemorySummarizer(
            summary_interval=rag_summary_interval,
            text_generator=text_generator
        )
        
        # Initialize importance calculator
        self.importance_calculator = ImportanceCalculator(text_generator=text_generator)
        
        # Initialize RAG trigger (use_llm_judgment=False by default to avoid blocking stream start and voice stutter)
        self.rag_trigger = RAGTrigger(text_generator=text_generator, use_llm_judgment=rag_use_llm_trigger)
        
        self.text_generator = text_generator
    
    def get_context_for_generation(
        self,
        user_input: str,
        replay_token_budget: Optional[int] = None,
        rag_top_k: Optional[int] = None,
        force_use_rag: bool = False
    ) -> Tuple[List[Tuple[str, str]], List[Dict], Optional[str], bool]:
        """
        Get context for text generation, combining Replay and RAG
        
        Args:
            user_input: Current user input
            replay_token_budget: Token budget for Replay (uses default if None)
            rag_top_k: Top-k for RAG (uses default if None)
            force_use_rag: Force RAG retrieval regardless of trigger (default: False)
        
        Returns:
            Tuple of (replay_history, rag_memories, enhanced_prompt, rag_used)
            - replay_history: List of (user_input, assistant_response) tuples
            - rag_memories: List of relevant RAG memories (empty if not triggered)
            - enhanced_prompt: Enhanced prompt with memories (None if no memories)
            - rag_used: Whether RAG was actually used
        """
        # Always get Replay history (short-term, time-ordered, controlled by token budget)
        replay_history = self.replay.get_replay_history(token_budget=replay_token_budget)
        
        # Conditionally get RAG memories (long-term, relevance-ordered)
        rag_memories = []
        rag_used = False
        should_use_rag = force_use_rag
        
        if not should_use_rag:
            # Check if RAG should be triggered
            should_use_rag, trigger_reason = self.rag_trigger.should_use_rag(user_input)
            if should_use_rag:
                print(f"[MemoryCoordinator] RAG triggered: {trigger_reason}")
        
        if should_use_rag:
            # Retrieve RAG memories
            rag_memories = self.rag.get_relevant_memories(
                query_text=user_input,
                top_k=rag_top_k
            )
            rag_used = True
            if rag_memories:
                print(f"[MemoryCoordinator] Retrieved {len(rag_memories)} RAG memories")
            else:
                print(f"[MemoryCoordinator] No relevant RAG memories found")
        else:
            print(f"[MemoryCoordinator] RAG not triggered, using Replay only")
        
        # Build enhanced prompt if RAG memories exist
        enhanced_prompt = None
        if rag_memories:
            # Format memories for prompt
            memory_lines = []
            for i, memory in enumerate(rag_memories, 1):
                metadata = memory.get("metadata", {})
                mem_user = metadata.get("user_input", "")
                mem_assistant = metadata.get("assistant_response", "")
                memory_lines.append(f"{i}. {mem_user} -> {mem_assistant}")
            
            memory_text = "\n".join(memory_lines)
            enhanced_prompt = f"\n相关记忆：\n{memory_text}\n"
        
        return replay_history, rag_memories, enhanced_prompt, rag_used
    
    def save_conversation(
        self,
        user_input: str,
        assistant_response: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        use_llm_evaluation: bool = True,
        async_save: bool = True
    ):
        """
        Save conversation to both Replay and RAG systems
        
        Args:
            user_input: User's input
            assistant_response: Assistant's response
            importance: Importance score for RAG (0.0-1.0). If None, will be calculated using LLM
            tags: Optional tags for RAG
            use_llm_evaluation: Whether to use LLM to calculate importance if not provided
            async_save: If True, save Replay immediately and process RAG in background thread
        """
        # Always save to Replay immediately (short-term memory, fast operation)
        self.replay.add_turn(user_input, assistant_response)
        
        if async_save:
            # Save RAG in background thread to avoid blocking
            def save_rag_async():
                try:
                    self._save_to_rag(
                        user_input=user_input,
                        assistant_response=assistant_response,
                        importance=importance,
                        tags=tags,
                        use_llm_evaluation=use_llm_evaluation
                    )
                except Exception as e:
                    print(f"[MemoryCoordinator] Error in async RAG save: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Start background thread for RAG saving
            rag_thread = threading.Thread(target=save_rag_async, daemon=True, name="rag_save_thread")
            rag_thread.start()
            print(f"[MemoryCoordinator] Started async RAG save thread (ID: {rag_thread.ident})")
        else:
            # Synchronous save (for backward compatibility)
            self._save_to_rag(
                user_input=user_input,
                assistant_response=assistant_response,
                importance=importance,
                tags=tags,
                use_llm_evaluation=use_llm_evaluation
            )
    
    def _save_to_rag(
        self,
        user_input: str,
        assistant_response: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        use_llm_evaluation: bool = True
    ):
        """
        Internal method to save conversation to RAG system
        This method is called in background thread for async saves
        
        Args:
            user_input: User's input
            assistant_response: Assistant's response
            importance: Importance score for RAG (0.0-1.0). If None, will be calculated using LLM
            tags: Optional tags for RAG
            use_llm_evaluation: Whether to use LLM to calculate importance if not provided
        """
        # Calculate importance if not provided
        calculated_importance = importance
        importance_details = {}
        
        if calculated_importance is None and use_llm_evaluation and self.importance_calculator:
            try:
                print(f"[MemoryCoordinator] Calculating importance in background thread...")
                calculated_importance, importance_details = self.importance_calculator.calculate_importance(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    fallback_importance=1.0
                )
                print(f"[MemoryCoordinator] Calculated importance: {calculated_importance:.2f}")
                if importance_details:
                    print(f"[MemoryCoordinator] Details: {importance_details}")
            except Exception as e:
                print(f"[MemoryCoordinator] Error calculating importance: {e}")
                calculated_importance = 1.0  # Fallback
        
        if calculated_importance is None:
            calculated_importance = 1.0  # Final fallback
        
        # Save to RAG with importance check (long-term memory)
        # This will automatically filter by importance and merge duplicates
        rag_memory_id = self.rag.add_memory_with_importance_check(
            user_input=user_input,
            assistant_response=assistant_response,
            importance=calculated_importance,
            tags=tags,
            auto_merge=True
        )
        
        # Check if summary should be generated
        if self.summarizer.should_generate_summary():
            self._generate_and_save_summary()
    
    def _generate_and_save_summary(self):
        """Generate summary from recent Replay turns and save to RAG"""
        try:
            # Get recent turns for summary (last N turns, where N = summary_interval)
            recent_turns = self.replay.get_replay_history_with_metadata(
                token_budget=self.replay.token_budget  # Get all recent turns
            )
            
            # Limit to summary_interval number of turns
            interval = self.summarizer.summary_interval
            recent_turns = recent_turns[-interval:] if len(recent_turns) > interval else recent_turns
            
            if not recent_turns:
                return
            
            # Generate summary
            summary = self.summarizer.generate_summary_from_turns(recent_turns)
            
            if summary:
                # Save summary to RAG with higher importance
                self.rag.save_summary(summary, importance=0.8)
                print(f"[MemoryCoordinator] Generated and saved summary: {summary[:50]}...")
        except Exception as e:
            print(f"[MemoryCoordinator] Error generating summary: {e}")
            import traceback
            traceback.print_exc()
    
    def should_use_rag(self) -> bool:
        """
        Check if RAG should be used
        
        Returns:
            True if RAG has memories or allow_no_memories is True
        """
        if self.rag.allow_no_memories:
            return True
        
        return self.rag.count() > 0
    
    def get_replay_history(self, token_budget: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get Replay history
        
        Args:
            token_budget: Token budget (uses default if None)
        
        Returns:
            List of (user_input, assistant_response) tuples
        """
        return self.replay.get_replay_history(token_budget=token_budget)
    
    def get_rag_memories(self, query_text: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Get RAG memories for a query
        
        Args:
            query_text: Query text
            top_k: Top-k (uses default if None)
        
        Returns:
            List of relevant memories
        """
        return self.rag.get_relevant_memories(query_text=query_text, top_k=top_k)
    
    def clear_replay_session(self):
        """Clear current Replay session"""
        self.replay.clear_session()
        self.summarizer.reset_turn_count()
    
    def get_stats(self) -> Dict:
        """Get statistics from both systems"""
        replay_stats = {
            "session_id": self.replay.get_session_id(),
            "turn_count": self.replay.get_turn_count(),
            "total_tokens": self.replay.get_total_tokens()
        }
        
        rag_stats = self.rag.get_memory_stats()
        
        return {
            "replay": replay_stats,
            "rag": rag_stats
        }
    
    def persist(self):
        """Persist both systems"""
        self.replay._save_session()
        self.rag.persist()
    
    def set_text_generator(self, text_generator):
        """Set text generator for summary generation and importance calculation"""
        self.text_generator = text_generator
        self.summarizer.set_text_generator(text_generator)
        self.importance_calculator.set_text_generator(text_generator)
        self.rag_trigger.set_text_generator(text_generator)