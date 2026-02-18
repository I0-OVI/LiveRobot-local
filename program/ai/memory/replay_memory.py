"""
Replay memory module
Manages session-based time-series conversation replay with token budget limits
"""
import os
import json
import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Token counting will be approximate.")


class ReplayMemory:
    """Replay memory for session-based conversation history"""
    
    def __init__(self, token_budget: int = 2000, persist_sessions: bool = True, persist_path: str = "./replay_db"):
        """
        Initialize replay memory
        
        Args:
            token_budget: Maximum tokens for replay history (default: 2000)
            persist_sessions: Whether to persist sessions across restarts
            persist_path: Path to persist session data
        """
        self.token_budget = token_budget
        self.persist_sessions = persist_sessions
        self.persist_path = persist_path
        
        # Current session
        self.session_id = self._generate_session_id()
        self.turns: List[Dict] = []
        self.total_tokens = 0
        
        # Token encoder (for Qwen model)
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                # Use Qwen's tokenizer
                self.encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4/Qwen compatible
            except Exception as e:
                print(f"[Replay] Warning: Failed to load tiktoken encoder: {e}")
                self.encoder = None
        
        # Load persisted sessions if enabled
        if self.persist_sessions:
            os.makedirs(persist_path, exist_ok=True)
            self._load_sessions()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Input text
        
        Returns:
            Estimated token count
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: approximate token count (1 token ≈ 4 characters for Chinese, 0.75 for English)
        # Use a conservative estimate
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 0.5 + other_chars * 0.75)
    
    def add_turn(self, user_input: str, assistant_response: str) -> int:
        """
        Add a conversation turn
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        
        Returns:
            Turn ID
        """
        turn_id = len(self.turns) + 1
        
        # Estimate token count for this turn
        turn_text = f"用户: {user_input}\n助手: {assistant_response}"
        token_count = self._count_tokens(turn_text)
        
        turn = {
            "turn_id": turn_id,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "token_count": token_count
        }
        
        self.turns.append(turn)
        self.total_tokens += token_count
        
        # Persist if enabled
        if self.persist_sessions:
            self._save_session()
        
        return turn_id
    
    def get_replay_history(self, token_budget: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get replay history within token budget
        
        Args:
            token_budget: Token budget (uses instance default if None)
        
        Returns:
            List of (user_input, assistant_response) tuples, ordered from oldest to newest
        """
        budget = token_budget if token_budget is not None else self.token_budget
        
        if not self.turns:
            return []
        
        # Select turns from newest to oldest until budget is reached
        selected_turns = []
        used_tokens = 0
        
        for turn in reversed(self.turns):
            if used_tokens + turn["token_count"] <= budget:
                selected_turns.insert(0, turn)  # Insert at beginning to maintain order
                used_tokens += turn["token_count"]
            else:
                break
        
        # Return as (user_input, assistant_response) tuples
        return [(turn["user_input"], turn["assistant_response"]) for turn in selected_turns]
    
    def get_replay_history_with_metadata(self, token_budget: Optional[int] = None) -> List[Dict]:
        """
        Get replay history with metadata
        
        Args:
            token_budget: Token budget
        
        Returns:
            List of turn dictionaries with metadata
        """
        budget = token_budget if token_budget is not None else self.token_budget
        
        if not self.turns:
            return []
        
        selected_turns = []
        used_tokens = 0
        
        for turn in reversed(self.turns):
            if used_tokens + turn["token_count"] <= budget:
                selected_turns.insert(0, turn)
                used_tokens += turn["token_count"]
            else:
                break
        
        return selected_turns
    
    def get_session_id(self) -> str:
        """Get current session ID"""
        return self.session_id
    
    def clear_session(self):
        """Clear current session"""
        self.turns = []
        self.total_tokens = 0
        self.session_id = self._generate_session_id()
        
        if self.persist_sessions:
            self._save_session()
    
    def get_turn_count(self) -> int:
        """Get number of turns in current session"""
        return len(self.turns)
    
    def get_total_tokens(self) -> int:
        """Get total tokens in current session"""
        return self.total_tokens
    
    def _save_session(self):
        """Save current session to disk"""
        if not self.persist_sessions:
            return
        
        try:
            os.makedirs(self.persist_path, exist_ok=True)  # Ensure directory exists
            session_file = os.path.join(self.persist_path, f"{self.session_id}.json")
            session_data = {
                "session_id": self.session_id,
                "turns": self.turns,
                "total_tokens": self.total_tokens,
                "created_at": datetime.now().isoformat()
            }
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            if len(self.turns) == 1:  # First turn of session - log path once
                print(f"[Replay] Session saved to: {os.path.abspath(session_file)}")
        except Exception as e:
            print(f"[Replay] Error saving session to {self.persist_path}: {e}")
    
    def _load_sessions(self):
        """Load persisted sessions (for future use)"""
        if not os.path.exists(self.persist_path):
            return
        
        try:
            # For now, we only load the most recent session if needed
            # This can be extended to support loading specific sessions
            session_files = [
                f for f in os.listdir(self.persist_path)
                if f.endswith('.json') and f.startswith('session_')
            ]
            
            if session_files:
                # Sort by modification time, get most recent
                session_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.persist_path, f)), reverse=True)
                # Note: We don't auto-load sessions, but the infrastructure is here
        except Exception as e:
            print(f"[Replay] Error loading sessions: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a specific session
        
        Args:
            session_id: Session ID to load
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.persist_sessions:
            return False
        
        try:
            session_file = os.path.join(self.persist_path, f"{session_id}.json")
            if not os.path.exists(session_file):
                return False
            
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.session_id = session_data.get("session_id", self.session_id)
            self.turns = session_data.get("turns", [])
            self.total_tokens = session_data.get("total_tokens", 0)
            
            return True
        except Exception as e:
            print(f"[Replay] Error loading session {session_id}: {e}")
            return False
