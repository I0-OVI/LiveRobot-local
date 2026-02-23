"""
Load system prompt and role from the setup file.
Setup file: 'setup.txt' in the Reorganize root directory.
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
    Returns a dict with keys: SYSTEM_PROMPT_ZH, SYSTEM_PROMPT_EN, ROLE_ZH, ROLE_EN,
    FORBIDDEN_WORDS_ZH, FORBIDDEN_WORDS_EN. Only keys with non-empty content are present.
    For forbidden words, value is newline-separated phrases (split into list by caller).
    """
    return _parse_setup_file(_SETUP_PATH)


def parse_forbidden_words_list(raw: str) -> list:
    """Parse newline-separated forbidden words section into a list of stripped non-empty strings."""
    if not raw or not raw.strip():
        return []
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
