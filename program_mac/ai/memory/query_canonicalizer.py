"""
Query Canonicalization layer for RAG retrieval.

In conversational context, "你"/"我"/"用户" (Chinese), "you"/"I"/"user" (English),
and the actual user name (e.g. Carambola from setup.txt) all refer to the same
entity. This module normalizes to canonical forms within each language.
"""
import re
from typing import List, Optional, Tuple

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


def canonicalize(text: str, user_name: Optional[str] = None) -> str:
    """
    Normalize user-referring terms: Chinese → "用户", English → "user".
    If user_name is provided (from setup.txt), it is also replaced with "用户".

    Args:
        text: Raw query or document text
        user_name: Actual user name (e.g. Carambola); replaced with 用户 for retrieval

    Returns:
        Canonicalized text with user references unified
    """
    if not text or not isinstance(text, str):
        return text

    result = text
    # User name from setup.txt -> 用户 (align with Chinese canonical form)
    if user_name and user_name.strip():
        name = user_name.strip()
        result = result.replace(name, "用户")
    for pattern, replacement in _CHINESE_PAIRS:
        result = result.replace(pattern, replacement)
    for pattern, replacement in _ENGLISH_PAIRS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def canonicalize_for_retrieval(text: str, user_name: Optional[str] = None) -> str:
    """
    Alias for canonicalize. Use this when preparing text for RAG retrieval.
    """
    return canonicalize(text, user_name=user_name)
