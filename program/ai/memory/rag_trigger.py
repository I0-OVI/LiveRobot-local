"""
RAG trigger module
Determines when to use RAG based on user input patterns
"""
import re
from typing import Optional, Tuple


class RAGTrigger:
    """
    Determines when to trigger RAG retrieval
    
    Two-layer judgment:
    - Layer 1: Keyword detection - if specific memory keywords appear, use RAG directly
    - Layer 2: LLM judgment - only when Layer 1 does not trigger, use LLM to judge
    """
    
    # Layer 1: First-layer keywords (specific memory-related phrases)
    # If any of these appear, use RAG directly without LLM judgment
    LAYER1_KEYWORDS = [
        # Chinese: temporal/history
        "之前", "刚才", "我们说过", "以前", "之前的", "原来那个",
        # Chinese: user preference / self info (often stored in RAG)
        "喜欢什么", "偏好", "爱好", "叫什么", "住在哪", "母语",
        "喜欢看", "喜欢读", "喜欢听",
        # English
        "my preference", "we decided", "what do I like", "my name"
    ]
    
    def __init__(self, text_generator=None, use_llm_judgment: bool = True):
        """
        Initialize RAG trigger
        
        Args:
            text_generator: Text generator for LLM-based judgment (optional)
            use_llm_judgment: Whether to use LLM for judgment (default: True)
        """
        self.text_generator = text_generator
        self.use_llm_judgment = use_llm_judgment
    
    def should_use_rag(self, user_input: str) -> Tuple[bool, str]:
        """
        Determine if RAG should be used for this query
        
        Two-layer judgment:
        - Layer 1: If specific keywords appear, use RAG directly
        - Layer 2: If no keywords, use LLM to judge (supplementary)
        
        Args:
            user_input: User's input text
        
        Returns:
            Tuple of (should_use_rag, reason)
            - should_use_rag: True if RAG should be used
            - reason: Reason for the decision
        """
        if not user_input or len(user_input.strip()) == 0:
            return False, "Empty input"
        
        # Layer 1: Keyword detection
        keyword_result = self._check_layer1_keywords(user_input)
        if keyword_result[0]:
            return keyword_result
        
        # Layer 2: LLM judgment (only when Layer 1 does not trigger)
        if self.use_llm_judgment and self.text_generator:
            return self._check_with_llm(user_input)
        
        # No trigger
        return False, "Layer 1: No keywords; Layer 2: LLM not available"
    
    def _check_layer1_keywords(self, user_input: str) -> Tuple[bool, str]:
        """
        Layer 1: Check if user input contains specific memory-related keywords
        
        Keywords: 之前, 刚才, 我们说过, 以前, 之前的, 原来那个, my preference, we decided
        
        Args:
            user_input: User's input text
        
        Returns:
            Tuple of (should_use_rag, reason)
        """
        user_input_lower = user_input.lower()
        
        for keyword in self.LAYER1_KEYWORDS:
            if keyword.lower() in user_input_lower:
                return True, f"Layer 1 keyword: '{keyword}'"
        
        return False, "Layer 1: No keywords found"
    
    def _check_with_llm(self, user_input: str) -> Tuple[bool, str]:
        """
        Use LLM to judge if RAG is needed
        
        Args:
            user_input: User's input text
        
        Returns:
            Tuple of (should_use_rag, reason)
        """
        if not self.text_generator:
            return False, "No text generator available"
        
        try:
            prompt = f"""判断以下用户输入是否需要检索长期记忆（RAG）来回答。

需要检索RAG的情况：
1. 用户明确询问之前提到过的信息（如"你记得之前..."）
2. 用户询问关于自己的个人信息（如"我的名字是什么"、"我喜欢什么"）
3. 用户询问需要从历史对话中获取的信息

不需要检索RAG的情况：
1. 一般性对话（如"你好"、"今天天气怎么样"）
2. 工具调用请求（如"现在几点了"）
3. 不需要历史信息的简单问答

用户输入：{user_input}

请只返回JSON格式：
{{"need_rag": true/false, "reason": "原因说明"}}

JSON:"""
            
            # Use LLM to judge
            response, _ = self.text_generator.chat(prompt, history=[])
            
            # Parse JSON response
            result = self._parse_llm_response(response)
            
            if result:
                need_rag = result.get("need_rag", False)
                reason = result.get("reason", "LLM judgment")
                return need_rag, reason
            
            return False, "Failed to parse LLM response"
            
        except Exception as e:
            print(f"[RAGTrigger] Error in LLM judgment: {e}")
            return False, f"LLM judgment error: {e}"
    
    def _parse_llm_response(self, response: str) -> Optional[dict]:
        """
        Parse LLM JSON response
        
        Args:
            response: LLM response text
        
        Returns:
            Parsed dictionary or None
        """
        if not response:
            return None
        
        # Try to extract JSON from response
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```"):
            # Extract content between ```json and ```
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                # Try to find JSON object
                match = re.search(r"\{.*?\}", response, re.DOTALL)
                if match:
                    response = match.group(0)
        
        # Try to find JSON object
        match = re.search(r"\{.*?\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                import json
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        return None
    
    def set_text_generator(self, text_generator):
        """Set text generator for LLM judgment"""
        self.text_generator = text_generator
