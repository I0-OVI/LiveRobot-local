"""
Memory summarizer module
Generates summaries of conversations every N turns using LLM
"""
from typing import List, Dict, Optional
from datetime import datetime


class MemorySummarizer:
    """Memory summarizer for periodic summary generation"""
    
    def __init__(self, summary_interval: int = 10, text_generator=None):
        """
        Initialize memory summarizer
        
        Args:
            summary_interval: Number of turns between summaries (default: 10)
            text_generator: QwenTextGenerator instance for generating summaries
        """
        self.summary_interval = summary_interval
        self.text_generator = text_generator
        self.turn_count = 0
        self.last_summary_turn = 0
    
    def should_generate_summary(self) -> bool:
        """
        Check if a summary should be generated
        
        Returns:
            True if summary should be generated
        """
        self.turn_count += 1
        return (self.turn_count - self.last_summary_turn) >= self.summary_interval
    
    def generate_summary(self, recent_memories: List[Dict], max_length: int = 100) -> Optional[str]:
        """
        Generate a summary of recent memories using LLM
        
        Args:
            recent_memories: List of recent memory dictionaries
            max_length: Maximum length of summary in characters
        
        Returns:
            Generated summary text, or None if generation fails
        """
        if not self.text_generator:
            print("[MemorySummarizer] Warning: No text generator available for summary generation")
            return None
        
        if not recent_memories:
            return None
        
        try:
            # Build prompt with recent memories
            memories_text = "\n\n".join([
                f"用户: {m.get('metadata', {}).get('user_input', '')}\n助手: {m.get('metadata', {}).get('assistant_response', '')}"
                for m in recent_memories
            ])
            
            # Create summary prompt
            prompt = f"""请总结以下对话的核心内容，生成一个简洁的summary（不超过{max_length}字）：

{memories_text}

请用简洁的语言总结这段对话的主要内容和要点。Summary:"""
            
            # Generate summary using text generator
            # Use generate_simple for non-streaming generation
            if hasattr(self.text_generator, 'generate_simple'):
                summary = self.text_generator.generate_simple(prompt)
            elif hasattr(self.text_generator, 'generate_text'):
                summary = self.text_generator.generate_text(prompt, max_new_tokens=50)
            else:
                print("[MemorySummarizer] Warning: Text generator does not support summary generation")
                return None
            
            # Clean and truncate summary
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            self.last_summary_turn = self.turn_count
            print(f"[MemorySummarizer] Generated summary: {summary[:50]}...")
            
            return summary
            
        except Exception as e:
            print(f"[MemorySummarizer] Error generating summary: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_summary_from_turns(self, turns: List[Dict], max_length: int = 100) -> Optional[str]:
        """
        Generate summary from conversation turns (for Replay memory)
        
        Args:
            turns: List of turn dictionaries with user_input and assistant_response
            max_length: Maximum length of summary
        
        Returns:
            Generated summary text
        """
        if not self.text_generator:
            return None
        
        if not turns:
            return None
        
        try:
            # Build prompt
            turns_text = "\n\n".join([
                f"用户: {turn.get('user_input', '')}\n助手: {turn.get('assistant_response', '')}"
                for turn in turns
            ])
            
            prompt = f"""请总结以下对话的核心内容，生成一个简洁的summary（不超过{max_length}字）：

{turns_text}

请用简洁的语言总结这段对话的主要内容和要点。Summary:"""
            
            # Generate summary
            if hasattr(self.text_generator, 'generate_simple'):
                summary = self.text_generator.generate_simple(prompt)
            elif hasattr(self.text_generator, 'generate_text'):
                summary = self.text_generator.generate_text(prompt, max_new_tokens=50)
            else:
                return None
            
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
            
        except Exception as e:
            print(f"[MemorySummarizer] Error generating summary from turns: {e}")
            return None
    
    def reset_turn_count(self):
        """Reset turn count (useful for new sessions)"""
        self.turn_count = 0
        self.last_summary_turn = 0
    
    def set_summary_interval(self, interval: int):
        """
        Update summary interval
        
        Args:
            interval: New summary interval (must be positive)
        """
        if interval > 0:
            self.summary_interval = interval
        else:
            raise ValueError(f"Summary interval must be positive, got {interval}")
    
    def set_text_generator(self, text_generator):
        """
        Set text generator for summary generation
        
        Args:
            text_generator: QwenTextGenerator instance
        """
        self.text_generator = text_generator
