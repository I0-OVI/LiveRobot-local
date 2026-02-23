"""
Replay memory module
Single JSON file with sliding window: keeps only the most recent N turns
"""
import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Token counting will be approximate.")

REPLAY_FILENAME = "replay.json"


class ReplayMemory:
    """
    Replay memory: single JSON file, sliding window of max N turns.
    When exceeding max_turns, oldest turns are dropped (FIFO).
    """
    
    def __init__(
        self,
        token_budget: int = 2000,
        persist_sessions: bool = True,
        persist_path: str = "./replay_db",
        max_turns: int = 50
    ):
        """
        Initialize replay memory
        
        Args:
            token_budget: Maximum tokens for replay history (default: 2000)
            persist_sessions: Whether to persist to disk
            persist_path: Directory for replay.json
            max_turns: Maximum number of turns to retain; oldest dropped when exceeded (default: 50)
        """
        self.token_budget = token_budget
        self.persist_sessions = persist_sessions
        self.persist_path = persist_path
        self.max_turns = max_turns
        
        self.session_id = "replay"
        self.turns: List[Dict] = []
        self.total_tokens = 0
        
        self.encoder = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                print(f"[Replay] Warning: Failed to load tiktoken encoder: {e}")
                self.encoder = None
        
        if self.persist_sessions:
            os.makedirs(persist_path, exist_ok=True)
            self._load()
    
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
        Add a conversation turn. If exceeds max_turns, drop oldest (sliding window).
        
        Args:
            user_input: User's input text
            assistant_response: Assistant's response text
        
        Returns:
            Turn ID
        """
        turn_text = f"用户: {user_input}\n助手: {assistant_response}"
        token_count = self._count_tokens(turn_text)
        
        turn = {
            "turn_id": len(self.turns) + 1,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "timestamp": datetime.now().isoformat(),
            "token_count": token_count
        }
        
        self.turns.append(turn)
        self.total_tokens += token_count
        
        # Sliding window: remove oldest when exceeding max_turns
        while len(self.turns) > self.max_turns:
            removed = self.turns.pop(0)
            self.total_tokens -= removed.get("token_count", 0)
            # Re-number turn_ids
            for i, t in enumerate(self.turns, 1):
                t["turn_id"] = i
        
        if self.persist_sessions:
            self._save()
        
        return turn["turn_id"]
    
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
        """Clear all turns"""
        self.turns = []
        self.total_tokens = 0
        
        if self.persist_sessions:
            self._save()
    
    def get_turn_count(self) -> int:
        """Get number of turns in current session"""
        return len(self.turns)
    
    def get_total_tokens(self) -> int:
        """Get total tokens in current session"""
        return self.total_tokens
    
    def _save(self):
        """Save to single replay.json"""
        if not self.persist_sessions:
            return
        
        try:
            os.makedirs(self.persist_path, exist_ok=True)
            file_path = os.path.join(self.persist_path, REPLAY_FILENAME)
            data = {
                "turns": self.turns,
                "total_tokens": self.total_tokens,
                "max_turns": self.max_turns,
                "updated_at": datetime.now().isoformat()
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Replay] Error saving to {self.persist_path}: {e}")
    
    def _load(self):
        """Load from replay.json if exists"""
        file_path = os.path.join(self.persist_path, REPLAY_FILENAME)
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.turns = data.get("turns", [])
            self.total_tokens = data.get("total_tokens", 0)
            if len(self.turns) > self.max_turns:
                # Truncate if loaded more than current max_turns
                excess = len(self.turns) - self.max_turns
                self.turns = self.turns[excess:]
                self.total_tokens = sum(t.get("token_count", 0) for t in self.turns)
                for i, t in enumerate(self.turns, 1):
                    t["turn_id"] = i
                self._save()
            if self.turns:
                print(f"[Replay] Loaded {len(self.turns)} turns from {file_path}")
        except Exception as e:
            print(f"[Replay] Error loading {file_path}: {e}")
