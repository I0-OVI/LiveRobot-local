"""
Local time-of-day context for generation and tools.
Uses datetime.now() (respects OS TZ if set).
"""
from __future__ import annotations

from datetime import datetime
from typing import Tuple


def _bucket(hour: int) -> Tuple[str, str, str]:
    """
    Returns (zh_label, en_label, hint_zh) where hint_zh is a short scene hint for LLM.
    """
    if 5 <= hour < 12:
        return "早晨/上午", "morning", "可描写日间、晨光等。"
    if 12 <= hour < 18:
        return "下午", "afternoon", "可描写午后、日光等。"
    if 18 <= hour < 22:
        return "傍晚", "evening", "避免写成正午烈日、户外暴晒；傍晚偏凉、天色渐暗更贴切。"
    # 22–5
    return "夜晚/深夜", "night", "避免烈日、晒太阳、正午户外暴晒等仅适合白天的描写；夜晚偏凉或室内为主。"


def local_time_context_block(user_input: str = "") -> str:
    """
    Short bilingual-safe block appended to system prompt before each generation.
    Grounds weather / outing / sunshine wording to actual local time.
    """
    now = datetime.now()
    line = now.strftime("%Y-%m-%d %H:%M")
    zh_period, en_period, scene_zh = _bucket(now.hour)
    has_cn = any("\u4e00" <= c <= "\u9fff" for c in (user_input or ""))
    if has_cn:
        return (
            f"【环境】当前本地时间：{line}（{zh_period}）。{scene_zh}"
            "涉及天气、出门、穿衣、晒太阳等时须与此时段一致。"
        )
    return (
        f"[Context] Local time: {line} ({en_period}). "
        "For weather, going out, clothing, or sunshine: stay consistent with this time of day; "
        "do not describe harsh midday sun or sunbathing when it is evening or night."
    )


def local_time_short_for_tool() -> str:
    """One line for tool results (e.g. weather), Chinese."""
    now = datetime.now()
    zh_period, _, _ = _bucket(now.hour)
    return f"本地当前：{now.strftime('%H:%M')}（{zh_period}）"
