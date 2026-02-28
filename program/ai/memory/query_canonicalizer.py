"""
Query Canonicalization layer for RAG retrieval.

In conversational context, "你"/"我"/"用户" (Chinese) and "you"/"I"/"user" (English)
all refer to the same entity (the person using the program). Different surface
forms can cause embedding mismatch. This module normalizes within each language:
Chinese → "用户", English → "user", so stored content stays in its language.
"""
import re
from typing import List, Tuple

# Chinese: normalize to "用户". Order matters (e.g., 咱们 before 咱)
_CHINESE_PAIRS: List[Tuple[str, str]] = [
    ("咱们", "用户"),
    ("本人", "用户"),
    ("您", "用户"),
    ("你", "用户"),
    ("我", "用户"),
    ("咱", "用户"),
    ("俺", "用户"),
]

# English: normalize to "user" (or "user's" for possessives). Word boundaries to avoid partials.
_ENGLISH_PAIRS: List[Tuple[str, str]] = [
    (r"\byour\b", "user's"),
    (r"\bmy\b", "user's"),
    (r"\bour\b", "user's"),
    (r"\byou\b", "user"),
    (r"\bI\b", "user"),
    (r"\bme\b", "user"),
    (r"\bwe\b", "user"),
    (r"\bus\b", "user"),
]


def canonicalize(text: str) -> str:
    """
    Normalize user-referring terms within each language: Chinese → "用户",
    English → "user". Keeps canonical form in the same language as the content.

    Args:
        text: Raw query or document text

    Returns:
        Canonicalized text with user references unified
    """
    if not text or not isinstance(text, str):
        return text

    result = text
    for pattern, replacement in _CHINESE_PAIRS:
        result = result.replace(pattern, replacement)
    for pattern, replacement in _ENGLISH_PAIRS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def canonicalize_for_retrieval(text: str) -> str:
    """
    Alias for canonicalize. Use this when preparing text for RAG retrieval.
    """
    return canonicalize(text)
