"""
Load system prompt and role from the setup file.
Setup file: 'setup.txt' next to the `program` package (same folder as main.py).
"""
import os
import re
from typing import Dict, Optional

# Reorganize root = parent of utils
_REORGANIZE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SETUP_FILENAME = "setup.txt"
_SETUP_PATH = os.path.join(_REORGANIZE_ROOT, _SETUP_FILENAME)

# Section names (must match markers in setup.txt)
_SECTIONS = (
    "USER_NAME",
    "SYSTEM_PROMPT_ZH", "SYSTEM_PROMPT_EN", "ROLE_ZH", "ROLE_EN",
    "FORBIDDEN_WORDS_ZH", "FORBIDDEN_WORDS_EN",
)


def _parse_setup_file(path: str) -> Dict[str, str]:
    """
    Parse setup file and return a dict of section_name -> content (stripped).
    Empty or missing sections are not included (caller uses defaults).
    """
    result = {}
    if not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return result

    for name in _SECTIONS:
        start_marker = f"# ---------- {name} ----------"
        end_marker = f"# ---------- END {name} ----------"
        start_marker_alt = f"---------- {name} ----------"
        end_marker_alt = f"---------- END {name} ----------"
        # Allow with or without leading #
        pattern = re.escape(start_marker) + r"\s*\n(.*?)\s*" + re.escape(end_marker)
        m = re.search(pattern, text, re.DOTALL)
        if not m:
            pattern2 = re.escape(start_marker_alt) + r"\s*\n(.*?)\s*" + re.escape(end_marker_alt)
            m = re.search(pattern2, text, re.DOTALL)
        if m:
            raw = m.group(1)
            # Ignore lines that are only comments or blank; use the rest as content
            lines = [line for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
            content = "\n".join(lines).strip()
            if content:
                result[name] = content
    return result


def get_setup_path() -> str:
    """Return the path to the setup file."""
    return _SETUP_PATH


def load_setup() -> Dict[str, str]:
    """
    Load system prompt, role and forbidden words from setup file.
    Returns a dict with keys: USER_NAME, SYSTEM_PROMPT_ZH, SYSTEM_PROMPT_EN, ROLE_ZH, ROLE_EN,
    FORBIDDEN_WORDS_ZH, FORBIDDEN_WORDS_EN. Only keys with non-empty content are present.
    For forbidden words, value is newline-separated phrases (split into list by caller).
    """
    return _parse_setup_file(_SETUP_PATH)


def get_default_weather_city(setup: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Parse the user's home city from ROLE_ZH / ROLE_EN for the weather tool default.

    Recognizes common phrasing, e.g. 「目前住在上海」「住在上海」or English 「lives in Shanghai」.
    Returns None if not found or placeholder-only (e.g. unreplaced ``{...}``).
    """
    if setup is None:
        setup = load_setup()
    role_zh = setup.get("ROLE_ZH") or ""
    role_en = setup.get("ROLE_EN") or ""

    for pat in (
        r"(?:目前)?\s*住在\s*([^\s\n。，,；、#]+)",
        r"居住在\s*([^\s\n。，,；、#]+)",
    ):
        m = re.search(pat, role_zh)
        if m:
            city = m.group(1).strip()
            if city and "{" not in city:
                return city

    m = re.search(
        r"\b(?:live|lives|living|residing)\s+in\s+([^\n,;.#]+)",
        role_en,
        re.IGNORECASE,
    )
    if m:
        city = m.group(1).strip().rstrip(".")
        if city and "{" not in city:
            return city

    return None


def get_user_name() -> str:
    """
    Get user name from setup file. Used for ROLE placeholder and RAG canonicalization.
    Returns first non-empty line of USER_NAME section, or "User" if empty/missing.
    Read is a single file parse at first call; typically <10ms.
    """
    setup = load_setup()
    raw = setup.get("USER_NAME", "").strip()
    if not raw:
        return "User"
    # Use first non-comment line
    for line in raw.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return "User"


def parse_forbidden_words_list(raw: str) -> list:
    """Parse newline-separated forbidden words section into a list of stripped non-empty strings."""
    if not raw or not raw.strip():
        return []
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]


def _parse_bool_value(raw: str) -> Optional[bool]:
    s = raw.strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off", ""):
        return False
    return None


def _normalize_rag_option_key(key: str) -> Optional[str]:
    k = key.strip().lower().replace("-", "_")
    aliases = {
        "rag_use_llm_trigger": "use_llm_trigger",
        "use_llm_trigger": "use_llm_trigger",
        "rag_llm_trigger_timeout_sec": "llm_trigger_timeout_sec",
        "llm_trigger_timeout_sec": "llm_trigger_timeout_sec",
        "rag_always_retrieve": "always_retrieve",
        "always_retrieve": "always_retrieve",
        "rag_use_time_weight": "use_time_weight",
        "use_time_weight": "use_time_weight",
        "rag_time_decay_days": "time_decay_days",
        "time_decay_days": "time_decay_days",
        "rag_use_llm_long_term_eval": "use_llm_long_term_eval",
        "use_llm_long_term_eval": "use_llm_long_term_eval",
        "rag_save_llm_timeout_sec": "save_llm_timeout_sec",
        "save_llm_timeout_sec": "save_llm_timeout_sec",
        "rag_use_save_worker": "use_save_worker",
        "use_save_worker": "use_save_worker",
    }
    return aliases.get(k)


def get_rag_options(path: Optional[str] = None) -> Dict[str, object]:
    """
    Parse optional # ---------- RAG_OPTIONS ---------- section in setup.txt.
    Lines: key=value (booleans: true/false/1/0/yes/no). Lines starting with # ignored.

    Keys (aliases accepted with or without rag_ prefix):
        use_llm_trigger, llm_trigger_timeout_sec (float; seconds), always_retrieve,
        use_time_weight, time_decay_days,
        use_llm_long_term_eval, save_llm_timeout_sec (float), use_save_worker

    Omitted keys are not present in the returned dict (caller supplies defaults).
    """
    path = path or _SETUP_PATH
    result: Dict[str, object] = {}
    if not os.path.isfile(path):
        return result
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return result

    start_marker = "# ---------- RAG_OPTIONS ----------"
    end_marker = "# ---------- END RAG_OPTIONS ----------"
    si = text.find(start_marker)
    ei = text.find(end_marker)
    if si < 0 or ei < 0 or ei <= si:
        return result

    block = text[si + len(start_marker) : ei]
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        nk = _normalize_rag_option_key(key)
        if not nk:
            continue
        val = val.split("#", 1)[0].strip()
        if nk == "time_decay_days":
            try:
                result[nk] = int(val)
            except ValueError:
                pass
        elif nk == "llm_trigger_timeout_sec":
            try:
                result[nk] = float(val)
            except ValueError:
                pass
        elif nk == "save_llm_timeout_sec":
            try:
                result[nk] = float(val)
            except ValueError:
                pass
        else:
            b = _parse_bool_value(val)
            if b is not None:
                result[nk] = b
    return result
