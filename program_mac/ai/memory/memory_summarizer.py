"""
Memory summarizer: rolling conversation summary for Replay compaction (Qwen via chat queue).
"""
from typing import List, Dict, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# User-specified instructions (English); rolling merge adds structured sections below.
SUMMARY_INSTRUCTIONS = """Summarize the following conversation.
Keep only information that may be useful for future interactions.
Remove greetings and irrelevant details.
Do not add information that was not explicitly stated.
Limit the summary to 150 tokens."""


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not (text or "").strip():
        return (text or "").strip()
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            ids = enc.encode(text)
            if len(ids) <= max_tokens:
                return text.strip()
            return enc.decode(ids[:max_tokens]).strip()
        except Exception:
            pass
    # Fallback: ~4 chars per token
    cap = max_tokens * 4
    s = text.strip()
    return s[:cap] if len(s) > cap else s


class MemorySummarizer:
    """Builds rolling summaries for Replay compaction using the shared text generator."""

    def __init__(
        self,
        summary_interval: int = 10,
        text_generator=None,
        max_output_tokens: int = 150,
        llm_max_new_tokens: int = 256,
    ):
        self.summary_interval = max(1, int(summary_interval))
        self.text_generator = text_generator
        self.max_output_tokens = max(1, int(max_output_tokens))
        self.llm_max_new_tokens = max(32, int(llm_max_new_tokens))

    def set_text_generator(self, text_generator):
        self.text_generator = text_generator

    def set_summary_interval(self, interval: int):
        if interval > 0:
            self.summary_interval = interval
        else:
            raise ValueError(f"Summary interval must be positive, got {interval}")

    def rolling_merge(
        self,
        previous_summary: str,
        turns: List[Dict],
    ) -> Optional[str]:
        """
        Merge previous_summary with transcript from turn dicts (user_input / assistant_response).
        Returns trimmed summary text or None on failure.
        """
        if not self.text_generator or not turns:
            return None
        if getattr(self.text_generator, "model", None) is None:
            return None

        turns_text = "\n\n".join(
            f"用户: {t.get('user_input', '')}\n助手: {t.get('assistant_response', '')}"
            for t in turns
        )
        prev = (previous_summary or "").strip()
        prompt = (
            f"{SUMMARY_INSTRUCTIONS}\n\n"
            f"Previous summary (may be empty):\n{prev if prev else '(none)'}\n\n"
            f"New conversation lines to merge:\n{turns_text}\n\n"
            "Summary:"
        )

        try:
            if hasattr(self.text_generator, "chat"):
                response, _ = self.text_generator.chat(
                    prompt,
                    history=[],
                    enhanced_prompt=None,
                    max_new_tokens=self.llm_max_new_tokens,
                )
            else:
                return None
        except Exception as e:
            print(f"[MemorySummarizer] rolling_merge chat failed: {e}")
            return None

        out = (response or "").strip()
        if not out:
            return None
        return _truncate_to_tokens(out, self.max_output_tokens)

    def reset_turn_count(self):
        """Legacy no-op (rolling compaction uses Replay turn counts)."""
        pass

    # --- Legacy API used by older scripts / tests ---
    def should_generate_summary(self) -> bool:
        return False

    def generate_summary(self, recent_memories: List[Dict], max_length: int = 100) -> Optional[str]:
        if not recent_memories:
            return None
        turns = [
            {
                "user_input": m.get("metadata", {}).get("user_input", ""),
                "assistant_response": m.get("metadata", {}).get("assistant_response", ""),
            }
            for m in recent_memories
        ]
        return self.rolling_merge("", turns)

    def generate_summary_from_turns(self, turns: List[Dict], max_length: int = 100) -> Optional[str]:
        del max_length  # unused; output capped by max_output_tokens
        return self.rolling_merge("", turns)
