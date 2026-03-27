"""
Memory coordinator module
Coordinates Replay and RAG memory systems
"""
import queue
import threading
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .replay_memory import ReplayMemory
from .rag_memory import RAGMemory
from .memory_summarizer import MemorySummarizer
from .importance_calculator import ImportanceCalculator
from .rag_trigger import RAGTrigger
from .rag_save_evaluator import RAGSaveEvaluator


class MemoryCoordinator:
    """Coordinates Replay and RAG memory systems"""
    
    def __init__(
        self,
        replay_token_budget: int = 2000,
        replay_persist_sessions: bool = True,
        replay_persist_path: str = "./replay_db",
        replay_max_turns: int = 50,
        rag_persist_directory: str = "./memory_db",
        rag_collection_name: str = "conversation_memories",
        rag_embedder_model: Optional[str] = None,
        rag_top_k: int = 3,
        rag_similarity_threshold: float = 0.7,
        rag_summary_interval: int = 10,
        rag_importance_base_threshold: float = 0.5,
        rag_importance_max_memories: int = 1000,
        rag_merge_similarity_threshold: float = 0.95,
        rag_auto_merge_every: int = 50,
        rag_allow_no_memories: bool = True,
        rag_use_llm_trigger: bool = False,
        rag_llm_trigger_timeout_sec: Optional[float] = None,
        rag_always_retrieve: bool = False,
        rag_use_time_weight: bool = False,
        rag_time_decay_days: int = 30,
        rag_user_name: Optional[str] = None,
        text_generator=None,
        rag_use_llm_long_term_eval: bool = False,
        rag_save_llm_timeout_sec: Optional[float] = None,
        rag_use_save_worker: bool = True,
        replay_conversation_summary_enabled: bool = True,
        replay_summary_max_output_tokens: int = 150,
        replay_summary_llm_max_new_tokens: int = 256,
        also_persist_summary_to_rag: bool = False,
    ):
        """
        Initialize memory coordinator
        
        Args:
            replay_token_budget: Token budget for Replay system
            replay_persist_sessions: Whether to persist Replay sessions
            replay_persist_path: Path for Replay persistence
            replay_max_turns: Max turns to retain; oldest dropped when exceeded (sliding window)
            rag_persist_directory: Directory for RAG persistence
            rag_collection_name: ChromaDB collection name
            rag_embedder_model: Embedder model name
            rag_top_k: Top-k for RAG retrieval
            rag_similarity_threshold: 语义相似度阈值，达到此值(默认0.7)的记忆才会被注入 prompt
            rag_summary_interval: Turns between summaries
            rag_importance_base_threshold: Base importance threshold
            rag_importance_max_memories: Max memories for adaptive threshold
            rag_merge_similarity_threshold: Similarity threshold for merging
            rag_auto_merge_every: Auto-merge after every N successful RAG saves (0 disables)
            rag_allow_no_memories: Allow no valid memories in RAG
            rag_use_llm_trigger: If True, use LLM to decide when to use RAG when keywords miss (adds latency before first token unless capped).
            rag_llm_trigger_timeout_sec: Max seconds to wait for that LLM call (None = no limit). <= 0 disables Layer2 LLM (keywords only).
            rag_always_retrieve: If True and the vector store is non-empty, always run retrieval; still only injects when similarity ≥ threshold.
            rag_use_time_weight: If True, boost recent memories in retrieval (see MemoryRetriever).
            rag_time_decay_days: Recency half-life style window for time weighting.
            rag_user_name: User name from setup.txt for RAG canonicalization (e.g. Carambola). If None, RAG uses default canonicalization only.
            text_generator: Text generator for summary generation and optional LLM RAG trigger
            rag_use_llm_long_term_eval: If True, one LLM JSON call decides store_long_term + importance (async path; uses inference queue).
            rag_save_llm_timeout_sec: Max seconds for save-side LLM (combined eval or importance-only). None = no limit.
                If <= 0 while long_term_eval is True, combined eval is skipped (legacy importance LLM only, if enabled).
            rag_use_save_worker: If True, enqueue RAG saves to a single daemon worker instead of spawning a thread per turn.
            replay_conversation_summary_enabled: If True, periodically merge oldest replay turns into a rolling summary (shortens Qwen context).
            replay_summary_max_output_tokens: Hard cap on stored summary length (after LLM).
            replay_summary_llm_max_new_tokens: Generation budget for the summarization chat() call.
            also_persist_summary_to_rag: If True, also write each new rolling summary to long-term RAG (same text).
        """
        # Initialize Replay system
        self.replay = ReplayMemory(
            token_budget=replay_token_budget,
            persist_sessions=replay_persist_sessions,
            persist_path=replay_persist_path,
            max_turns=replay_max_turns
        )
        
        # Initialize RAG system
        self.rag = RAGMemory(
            persist_directory=rag_persist_directory,
            collection_name=rag_collection_name,
            embedder_model=rag_embedder_model,
            top_k=rag_top_k,
            similarity_threshold=rag_similarity_threshold,
            use_time_weight=rag_use_time_weight,
            time_decay_days=rag_time_decay_days,
            importance_base_threshold=rag_importance_base_threshold,
            importance_max_memories=rag_importance_max_memories,
            merge_similarity_threshold=rag_merge_similarity_threshold,
            allow_no_memories=rag_allow_no_memories,
            user_name=rag_user_name
        )
        
        # Initialize summarizer (rolling replay compaction)
        self.summarizer = MemorySummarizer(
            summary_interval=rag_summary_interval,
            text_generator=text_generator,
            max_output_tokens=replay_summary_max_output_tokens,
            llm_max_new_tokens=replay_summary_llm_max_new_tokens,
        )
        
        # Initialize importance calculator
        self.importance_calculator = ImportanceCalculator(text_generator=text_generator)

        self.rag_save_evaluator = RAGSaveEvaluator(text_generator=text_generator)
        
        # Initialize RAG trigger (use_llm_judgment=False by default to avoid blocking stream start and voice stutter)
        self.rag_trigger = RAGTrigger(
            text_generator=text_generator,
            use_llm_judgment=rag_use_llm_trigger,
            llm_judgment_timeout_sec=rag_llm_trigger_timeout_sec,
        )
        
        # Periodic merge controls (to keep per-write latency low while still deduplicating)
        self.rag_auto_merge_every = max(0, int(rag_auto_merge_every))
        self._rag_save_count_since_merge = 0
        self._rag_merge_lock = threading.Lock()
        self.rag_similarity_threshold = rag_similarity_threshold  # 语义相似度>=0.7时使用该记忆
        self.rag_always_retrieve = bool(rag_always_retrieve)

        self.text_generator = text_generator
        self.rag_use_llm_long_term_eval = bool(rag_use_llm_long_term_eval)
        self.rag_save_llm_timeout_sec = rag_save_llm_timeout_sec
        self.rag_use_save_worker = bool(rag_use_save_worker)
        self.replay_conversation_summary_enabled = bool(replay_conversation_summary_enabled)
        self.also_persist_summary_to_rag = bool(also_persist_summary_to_rag)

        self._save_job_queue: queue.Queue = queue.Queue()
        self._save_worker_thread: Optional[threading.Thread] = None
        if self.rag_use_save_worker:
            self._start_save_worker()

    def _start_save_worker(self) -> None:
        if self._save_worker_thread is not None and self._save_worker_thread.is_alive():
            return

        def loop():
            while True:
                job = self._save_job_queue.get()
                try:
                    self._save_to_rag(
                        user_input=job["user_input"],
                        assistant_response=job["assistant_response"],
                        importance=job.get("importance"),
                        tags=job.get("tags"),
                        use_llm_evaluation=bool(job.get("use_llm_evaluation", True)),
                    )
                except Exception as e:
                    print(f"[MemoryCoordinator] Save worker error: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    self._save_job_queue.task_done()

        t = threading.Thread(
            target=loop, daemon=True, name="rag_save_worker"
        )
        self._save_worker_thread = t
        t.start()
        print("[MemoryCoordinator] RAG save worker thread started (single-queue)")

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

        if not should_use_rag and self.rag_always_retrieve and self.rag.count() > 0:
            should_use_rag = True
            print("[MemoryCoordinator] RAG triggered: always_retrieve (store non-empty)")

        if not should_use_rag:
            # Check if RAG should be triggered
            should_use_rag, trigger_reason = self.rag_trigger.should_use_rag(user_input)
            if should_use_rag:
                print(f"[MemoryCoordinator] RAG triggered: {trigger_reason}")
        
        if should_use_rag:
            # Retrieve RAG memories (only those with semantic similarity >= threshold, default 0.7)
            rag_memories = self.rag.get_relevant_memories(
                query_text=user_input,
                top_k=rag_top_k,
                similarity_threshold=self.rag_similarity_threshold
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
            # Format memories for prompt - explicit instruction so model uses them
            memory_lines = []
            for i, memory in enumerate(rag_memories, 1):
                metadata = memory.get("metadata", {})
                mem_user = metadata.get("user_input", "")
                mem_assistant = metadata.get("assistant_response", "")
                memory_lines.append(f"{i}. {mem_user} -> {mem_assistant}")
            
            memory_text = "\n".join(memory_lines)
            enhanced_prompt = (
                "\n\n【重要：请根据以下相关记忆回答用户问题，若记忆中有答案则直接使用】\n"
                "相关记忆：\n"
                f"{memory_text}\n"
            )
        
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

        try:
            self._maybe_compact_replay_with_summary()
        except Exception as e:
            print(f"[MemoryCoordinator] Replay rolling summary skipped: {e}")
            import traceback
            traceback.print_exc()
        
        if async_save:
            job = {
                "user_input": user_input,
                "assistant_response": assistant_response,
                "importance": importance,
                "tags": tags,
                "use_llm_evaluation": use_llm_evaluation,
            }
            if self.rag_use_save_worker:
                self._save_job_queue.put(job)
                qsize = self._save_job_queue.qsize()
                if qsize > 3:
                    print(
                        f"[MemoryCoordinator] RAG save queue depth={qsize} "
                        "(waits for single worker + Qwen FIFO)"
                    )
            else:
                def save_rag_async():
                    try:
                        self._save_to_rag(
                            user_input=user_input,
                            assistant_response=assistant_response,
                            importance=importance,
                            tags=tags,
                            use_llm_evaluation=use_llm_evaluation,
                        )
                    except Exception as e:
                        print(f"[MemoryCoordinator] Error in async RAG save: {e}")
                        import traceback
                        traceback.print_exc()

                rag_thread = threading.Thread(
                    target=save_rag_async, daemon=True, name="rag_save_thread"
                )
                rag_thread.start()
                print(
                    f"[MemoryCoordinator] Started async RAG save thread (ID: {rag_thread.ident})"
                )
        else:
            # Synchronous save (for backward compatibility)
            self._save_to_rag(
                user_input=user_input,
                assistant_response=assistant_response,
                importance=importance,
                tags=tags,
                use_llm_evaluation=use_llm_evaluation
            )
    
    def _is_non_answer_response(self, assistant_response: str) -> bool:
        """
        Detect 'I don't know' type responses that should not be stored in RAG.
        E.g. "我不知道xxx喜欢什么", "我不太了解" - storing these adds noise, not knowledge.
        """
        if not assistant_response or len(assistant_response.strip()) < 6:
            return False
        text = assistant_response.strip()
        # Non-answer phrases: response starting with or dominated by these
        non_answer_phrases = (
            "我不知道", "我不太知道", "不了解", "不太了解",
            "没法", "无法回答", "不太清楚", "不清楚",
            "I don't know", "I don't have", "I'm not sure",
        )
        for phrase in non_answer_phrases:
            if text.startswith(phrase):
                return True
            # Also catch "xxx，我不知道..." (phrase in first 40 chars)
            if phrase in text[:40]:
                return True
        return False

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
        # Skip non-answer responses (e.g. "我不知道xxx喜欢什么")
        if self._is_non_answer_response(assistant_response):
            print(f"[RAG] Skipped saving non-answer response (e.g. 不知道/不了解)")
            return

        calculated_importance = importance
        importance_details: Dict = {}
        merged_tags = list(tags) if tags else []

        # Explicit importance (e.g. tool calls): skip all save-side LLM gates
        if calculated_importance is not None:
            pass
        elif (
            self.rag_use_llm_long_term_eval
            and use_llm_evaluation
            and self.rag_save_evaluator
            and self.text_generator
        ):
            # <= 0 disables combined JSON eval → fall through to importance-only LLM
            use_combined = (
                self.rag_save_llm_timeout_sec is None
                or self.rag_save_llm_timeout_sec > 0
            )
            if use_combined:
                tout = (
                    self.rag_save_llm_timeout_sec
                    if self.rag_save_llm_timeout_sec is not None
                    and self.rag_save_llm_timeout_sec > 0
                    else None
                )
                print(
                    "[MemoryCoordinator] Long-term save eval (single LLM JSON) "
                    f"timeout_sec={tout}"
                )
                eval_result = self.rag_save_evaluator.evaluate(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    timeout_sec=tout,
                    fallback_importance=0.7,
                )
                print(
                    f"[MemoryCoordinator] Save eval: store={eval_result.store_long_term} "
                    f"imp={eval_result.importance:.2f} source={eval_result.source} "
                    f"reason={eval_result.reason[:80]!r}"
                )
                if not eval_result.store_long_term:
                    print("[MemoryCoordinator] Skipped long-term RAG (LLM store_long_term=false)")
                    return
                calculated_importance = eval_result.importance
                for t in eval_result.tags:
                    if t and t not in merged_tags:
                        merged_tags.append(t)
            elif use_llm_evaluation and self.importance_calculator:
                tout = (
                    self.rag_save_llm_timeout_sec
                    if self.rag_save_llm_timeout_sec is not None
                    and self.rag_save_llm_timeout_sec > 0
                    else None
                )
                try:
                    print(f"[MemoryCoordinator] Calculating importance (legacy LLM)...")
                    calculated_importance, importance_details = (
                        self.importance_calculator.calculate_importance(
                            user_input=user_input,
                            assistant_response=assistant_response,
                            fallback_importance=1.0,
                            timeout_sec=tout,
                        )
                    )
                    print(
                        f"[MemoryCoordinator] Calculated importance: {calculated_importance:.2f}"
                    )
                    if importance_details:
                        print(f"[MemoryCoordinator] Details: {importance_details}")
                except Exception as e:
                    print(f"[MemoryCoordinator] Error calculating importance: {e}")
                    calculated_importance = 1.0
        elif use_llm_evaluation and self.importance_calculator:
            tout = (
                self.rag_save_llm_timeout_sec
                if self.rag_save_llm_timeout_sec is not None
                and self.rag_save_llm_timeout_sec > 0
                else None
            )
            try:
                print(f"[MemoryCoordinator] Calculating importance in background...")
                calculated_importance, importance_details = (
                    self.importance_calculator.calculate_importance(
                        user_input=user_input,
                        assistant_response=assistant_response,
                        fallback_importance=1.0,
                        timeout_sec=tout,
                    )
                )
                print(
                    f"[MemoryCoordinator] Calculated importance: {calculated_importance:.2f}"
                )
                if importance_details:
                    print(f"[MemoryCoordinator] Details: {importance_details}")
            except Exception as e:
                print(f"[MemoryCoordinator] Error calculating importance: {e}")
                calculated_importance = 1.0

        if calculated_importance is None:
            calculated_importance = 1.0
        
        # Save to RAG with importance check (long-term memory)
        # Avoid per-write merge to keep single-write latency low.
        rag_memory_id = self.rag.add_memory_with_importance_check(
            user_input=user_input,
            assistant_response=assistant_response,
            importance=calculated_importance,
            tags=merged_tags if merged_tags else tags,
            auto_merge=False
        )

        # Run merge periodically instead of every write.
        if rag_memory_id and self.rag_auto_merge_every > 0:
            should_merge_now = False
            with self._rag_merge_lock:
                self._rag_save_count_since_merge += 1
                if self._rag_save_count_since_merge >= self.rag_auto_merge_every:
                    self._rag_save_count_since_merge = 0
                    should_merge_now = True

            if should_merge_now:
                try:
                    merged_count = self.rag.merge_similar_memories()
                    print(
                        f"[MemoryCoordinator] Periodic merge done "
                        f"(interval={self.rag_auto_merge_every}, merged={merged_count})"
                    )
                except Exception as e:
                    print(f"[MemoryCoordinator] Periodic merge failed: {e}")

    def _maybe_compact_replay_with_summary(self) -> None:
        """
        When replay has more than summary_interval turns, summarize the oldest N turns,
        remove them from replay, and store a rolling conversation_summary (prepended in get_replay_history).
        Runs synchronously after each add_turn so the next user message sees compressed context.
        """
        if not self.replay_conversation_summary_enabled:
            return
        n = self.summarizer.summary_interval
        if self.replay.get_turn_count() <= n:
            return
        if not self.text_generator or getattr(self.text_generator, "model", None) is None:
            return

        batch = list(self.replay.turns[:n])
        prev = self.replay.get_conversation_summary()
        new_summary = self.summarizer.rolling_merge(prev, batch)
        if not new_summary:
            return

        self.replay.pop_turns_from_start(n)
        self.replay.set_conversation_summary(new_summary)
        print(
            f"[MemoryCoordinator] Rolling replay summary updated "
            f"(removed {n} turns, summary_len={len(new_summary)})"
        )
        if self.also_persist_summary_to_rag:
            try:
                self.rag.save_summary(new_summary, importance=0.8)
            except Exception as e:
                print(f"[MemoryCoordinator] RAG save_summary failed: {e}")
    
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
    
    def get_rag_memories(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Get RAG memories for a query (no trigger gate; use for tools / RAGIntegration).
        """
        return self.rag.get_relevant_memories(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def cleanup_old_memories(
        self,
        days: Optional[int] = None,
        max_memories: Optional[int] = None,
        min_importance: float = 0.0,
    ) -> int:
        """Delegate to RAG long-term store (same semantics as legacy MemoryManager)."""
        return self.rag.cleanup_old_memories(
            days=days, max_memories=max_memories, min_importance=min_importance
        )

    def clear_all_rag_memories(self) -> bool:
        """Remove all vectors in the RAG collection."""
        return self.rag.clear_all_memories()
    
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
        if self.rag_use_save_worker:
            try:
                self._save_job_queue.join()
            except Exception:
                pass
        self.replay._save()
        self.rag.persist()
    
    def set_text_generator(self, text_generator):
        """Set text generator for summary generation, importance calculation, and optional LLM RAG trigger"""
        self.text_generator = text_generator
        self.summarizer.set_text_generator(text_generator)
        self.importance_calculator.set_text_generator(text_generator)
        self.rag_save_evaluator.set_text_generator(text_generator)
        self.rag_trigger.set_text_generator(text_generator)