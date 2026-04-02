"""
Single LLM call for long-term RAG save: whether to store + importance (+ optional tags).
Runs on the same Qwen inference queue as other chat() callers.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LongTermSaveEvalResult:
    """Outcome of combined long-term save evaluation."""

    store_long_term: bool
    importance: float
    tags: List[str]
    reason: str
    source: str  # "llm" | "timeout" | "parse_error" | "no_generator"


class RAGSaveEvaluator:
    """
    One-shot JSON prompt: store_long_term, importance (0-1), optional tags, optional reason.
    """

    PROMPT_TEMPLATE = """你是对话记忆管理员。根据下面一轮「用户-助手」对话，判断是否应写入长期向量记忆（RAG），并给出重要性分数。

应写入长期记忆的情况（store_long_term=true）示例：
- 用户偏好、习惯、个人信息（姓名、爱好等）
- 约定、承诺、需要日后回忆的事实
- 对后续对话有参考价值的知识性内容

不应写入（store_long_term=false）示例：
- 纯寒暄、无信息增量（如只说了你好）
- 临时指令、一次性查询
- 助手表示不知道、无法回答且未提供新事实

用户：{user_input}
助手：{assistant_response}

请只返回一个 JSON 对象（不要 markdown 代码块），格式如下：
{{"store_long_term": true 或 false, "importance": 0.0 到 1.0 的数字, "tags": ["可选标签字符串"], "reason": "简短原因"}}

JSON:"""

    MULTI_TURN_PROMPT_TEMPLATE = """你是对话记忆管理员。根据下面多轮「用户-助手」对话，综合判断是否应写入长期向量记忆（RAG），并给出重要性分数。

多轮对话全文：
{transcript}

判断原则与单轮一致：
- 应写入（store_long_term=true）：用户偏好、习惯、个人信息、约定、承诺、对后续对话有参考价值的知识性内容等。
- 不应写入（store_long_term=false）：全程纯寒暄无信息增量、仅临时指令/一次性查询、无值得长期检索的内容等。

请只返回一个 JSON 对象（不要 markdown 代码块），格式如下：
{{"store_long_term": true 或 false, "importance": 0.0 到 1.0 的数字, "tags": ["可选标签字符串"], "reason": "简短原因"}}

JSON:"""

    def __init__(self, text_generator=None):
        self.text_generator = text_generator

    def set_text_generator(self, text_generator) -> None:
        self.text_generator = text_generator

    def evaluate(
        self,
        user_input: str,
        assistant_response: str,
        timeout_sec: Optional[float] = None,
        fallback_importance: float = 0.7,
    ) -> LongTermSaveEvalResult:
        """
        Call LLM once. On timeout: do not store (store_long_term=False).
        On parse failure: store with fallback_importance (conservative default).
        """
        if not self.text_generator or not hasattr(self.text_generator, "chat"):
            return LongTermSaveEvalResult(
                store_long_term=True,
                importance=fallback_importance,
                tags=[],
                reason="no text generator",
                source="no_generator",
            )

        prompt = self.PROMPT_TEMPLATE.format(
            user_input=user_input or "",
            assistant_response=assistant_response or "",
        )

        try:
            if timeout_sec is not None and timeout_sec > 0:
                try:
                    response, _ = self.text_generator.chat(
                        prompt, history=[], timeout=timeout_sec
                    )
                except FuturesTimeout:
                    return LongTermSaveEvalResult(
                        store_long_term=False,
                        importance=0.0,
                        tags=[],
                        reason="save LLM timeout",
                        source="timeout",
                    )
            else:
                response, _ = self.text_generator.chat(prompt, history=[])

            parsed = self._parse_response(response)
            if parsed is None:
                return LongTermSaveEvalResult(
                    store_long_term=True,
                    importance=fallback_importance,
                    tags=[],
                    reason="parse_error_fallback",
                    source="parse_error",
                )

            store = bool(parsed.get("store_long_term", True))
            imp = parsed.get("importance", fallback_importance)
            try:
                imp = float(imp)
            except (TypeError, ValueError):
                imp = fallback_importance
            imp = max(0.0, min(1.0, imp))

            tags_raw = parsed.get("tags") or []
            tags: List[str] = []
            if isinstance(tags_raw, list):
                tags = [str(t).strip() for t in tags_raw if str(t).strip()]
            elif isinstance(tags_raw, str) and tags_raw.strip():
                tags = [tags_raw.strip()]

            reason = str(parsed.get("reason", "") or "").strip() or "llm"

            return LongTermSaveEvalResult(
                store_long_term=store,
                importance=imp,
                tags=tags,
                reason=reason,
                source="llm",
            )
        except Exception as e:
            print(f"[RAGSaveEvaluator] Error: {e}")
            return LongTermSaveEvalResult(
                store_long_term=True,
                importance=fallback_importance,
                tags=[],
                reason=f"exception:{e}",
                source="parse_error",
            )

    @staticmethod
    def format_multi_turn_transcript(turns: List[Tuple[str, str]]) -> str:
        """turns: ordered (user, assistant) pairs; assistant may be empty for the last pending user message."""
        lines: List[str] = []
        for i, (u, a) in enumerate(turns, 1):
            lines.append(f"第{i}轮")
            lines.append(f"用户：{u or ''}")
            lines.append(f"助手：{a or ''}")
        return "\n".join(lines)

    def evaluate_multi_turn(
        self,
        turns: List[Tuple[str, str]],
        timeout_sec: Optional[float] = None,
        fallback_importance: float = 0.7,
        max_new_tokens: int = 512,
    ) -> LongTermSaveEvalResult:
        """
        Like evaluate(), but considers full multi-turn (user, assistant) history.
        turns: list of pairs in order; last assistant may be "" if only user spoke last.
        """
        if not turns:
            return LongTermSaveEvalResult(
                store_long_term=False,
                importance=0.0,
                tags=[],
                reason="empty_turns",
                source="parse_error",
            )

        if not self.text_generator or not hasattr(self.text_generator, "chat"):
            return LongTermSaveEvalResult(
                store_long_term=True,
                importance=fallback_importance,
                tags=[],
                reason="no text generator",
                source="no_generator",
            )

        transcript = self.format_multi_turn_transcript(turns)
        prompt = self.MULTI_TURN_PROMPT_TEMPLATE.format(transcript=transcript)

        try:
            if timeout_sec is not None and timeout_sec > 0:
                try:
                    response, _ = self.text_generator.chat(
                        prompt,
                        history=[],
                        timeout=timeout_sec,
                        max_new_tokens=max_new_tokens,
                    )
                except FuturesTimeout:
                    return LongTermSaveEvalResult(
                        store_long_term=False,
                        importance=0.0,
                        tags=[],
                        reason="save LLM timeout",
                        source="timeout",
                    )
            else:
                response, _ = self.text_generator.chat(
                    prompt, history=[], max_new_tokens=max_new_tokens
                )

            parsed = self._parse_response(response)
            if parsed is None:
                return LongTermSaveEvalResult(
                    store_long_term=True,
                    importance=fallback_importance,
                    tags=[],
                    reason="parse_error_fallback",
                    source="parse_error",
                )

            store = bool(parsed.get("store_long_term", True))
            imp = parsed.get("importance", fallback_importance)
            try:
                imp = float(imp)
            except (TypeError, ValueError):
                imp = fallback_importance
            imp = max(0.0, min(1.0, imp))

            tags_raw = parsed.get("tags") or []
            tags: List[str] = []
            if isinstance(tags_raw, list):
                tags = [str(t).strip() for t in tags_raw if str(t).strip()]
            elif isinstance(tags_raw, str) and tags_raw.strip():
                tags = [tags_raw.strip()]

            reason = str(parsed.get("reason", "") or "").strip() or "llm"

            return LongTermSaveEvalResult(
                store_long_term=store,
                importance=imp,
                tags=tags,
                reason=reason,
                source="llm",
            )
        except Exception as e:
            print(f"[RAGSaveEvaluator] multi_turn error: {e}")
            return LongTermSaveEvalResult(
                store_long_term=True,
                importance=fallback_importance,
                tags=[],
                reason=f"exception:{e}",
                source="parse_error",
            )

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        if not response:
            return None
        text = response.strip()
        if text.startswith("```"):
            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
            )
            if match:
                text = match.group(1)
            else:
                match = re.search(r"\{.*?\}", text, re.DOTALL)
                if match:
                    text = match.group(0)
        else:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                text = match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None
