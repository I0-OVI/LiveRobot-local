"""
Whitelist for keyword-triggered "open app" tool (Windows).
Built-in aliases + optional program/open_app_allowlist.txt (alias=path or alias=canonical).
"""
from __future__ import annotations

import os
import sys
import re
import shutil
import subprocess
from typing import Dict, Optional, Tuple

# --- Optional user file: one entry per line ---
#   别名=完整路径.exe
#   别名=canonical_id   (canonical_id must be a built-in id, e.g. notepad)
# Multiple aliases:  别名1|别名2=path_or_canonical
_ALLOWLIST_FILENAME = "open_app_allowlist.txt"


def _program_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_user_allowlist() -> Dict[str, str]:
    """alias_lower -> 'path:...' or 'canon:...'"""
    path = os.path.join(_program_dir(), _ALLOWLIST_FILENAME)
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                left, right = line.split("=", 1)
                left, right = left.strip(), right.strip()
                if not left or not right:
                    continue
                if right.lower().endswith(".exe") or ":\\" in right or right.startswith("\\\\"):
                    payload = f"path:{right}"
                else:
                    payload = f"canon:{right.lower().strip()}"
                for alias in left.split("|"):
                    a = alias.strip()
                    if a:
                        out[a.lower()] = payload
    except OSError:
        pass
    return out


_USER_ALLOWLIST = _load_user_allowlist()


def _win_exe(name: str) -> Optional[str]:
    p = shutil.which(name)
    if p and os.path.isfile(p):
        return p
    w = os.environ.get("WINDIR", r"C:\Windows")
    sys32 = os.path.join(w, "System32", name)
    if os.path.isfile(sys32):
        return sys32
    return None


def _edge_exe() -> Optional[str]:
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        p = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
        if os.path.isfile(p):
            return p
    return shutil.which("msedge.exe")


def _chrome_exe() -> Optional[str]:
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LocalAppData", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return shutil.which("chrome.exe")


def _steam_exe() -> Optional[str]:
    """Typical Steam installs on Windows."""
    for env in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env)
        if not base:
            continue
        p = os.path.join(base, "Steam", "steam.exe")
        if os.path.isfile(p):
            return p
    for root in ("C:\\", "D:\\", "E:\\", "F:\\"):
        p = os.path.join(root, "Steam", "steam.exe")
        if os.path.isfile(p):
            return p
    return shutil.which("steam.exe")


# canonical_id -> launcher
_CANONICAL_LAUNCH: Dict[str, Tuple[str, ...]] = {
    "notepad": ("exe", "notepad.exe"),
    "calc": ("exe", "calc.exe"),
    "mspaint": ("exe", "mspaint.exe"),
    "explorer": ("exe", "explorer.exe"),
    "cmd": ("exe", "cmd.exe"),
    "powershell": ("exe", "powershell.exe"),
    "taskmgr": ("exe", "taskmgr.exe"),
    "snippingtool": ("exe", "SnippingTool.exe"),
    "edge": ("edge", ""),
    "chrome": ("chrome", ""),
    "steam": ("steam", ""),
    "settings": ("startfile", "ms-settings:"),
}

# alias (lowercase) -> canonical_id
_DEFAULT_ALIAS_TO_CANONICAL: Dict[str, str] = {
    # notepad
    "记事本": "notepad",
    "文本文档": "notepad",
    "notepad": "notepad",
    # calc
    "计算器": "calc",
    "calculator": "calc",
    "calc": "calc",
    # paint
    "画图": "mspaint",
    "画图程序": "mspaint",
    "mspaint": "mspaint",
    "paint": "mspaint",
    # explorer
    "资源管理器": "explorer",
    "文件夹": "explorer",
    "此电脑": "explorer",
    "文件资源管理器": "explorer",
    "explorer": "explorer",
    # settings
    "设置": "settings",
    "系统设置": "settings",
    "settings": "settings",
    # browsers
    "edge": "edge",
    "edge浏览器": "edge",
    "microsoft edge": "edge",
    "chrome": "chrome",
    "谷歌浏览器": "chrome",
    # games / clients
    "steam": "steam",
    "蒸汽": "steam",
    "steam客户端": "steam",
    # shells
    "命令提示符": "cmd",
    "cmd": "cmd",
    "powershell": "powershell",
    "终端": "powershell",
    # misc
    "任务管理器": "taskmgr",
    "task manager": "taskmgr",
    "截图工具": "snippingtool",
    "截图": "snippingtool",
    "snipping tool": "snippingtool",
}


def _merged_alias_map() -> Dict[str, str]:
    m = dict(_DEFAULT_ALIAS_TO_CANONICAL)
    for alias_lower, payload in _USER_ALLOWLIST.items():
        if payload.startswith("canon:"):
            cid = payload.split(":", 1)[1].strip()
            if cid in _CANONICAL_LAUNCH:
                m[alias_lower] = cid
    return m


_ALIAS_TO_CANONICAL = _merged_alias_map()


def _clean_target_phrase(raw: str) -> str:
    s = raw.strip()
    s = re.sub(
        r"(?:谢谢|感谢|好吗|好吧|行吧|拜托|麻烦|一下|的)+$",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_open_app_target(text: str) -> Optional[str]:
    """
    If text matches a safe 'open X' pattern, return the target phrase (trimmed).
    """
    text = text.strip()
    if not text:
        return None
    patterns = [
        r"帮(?:我|您)?打开\s*(.+?)(?:[。！？\n]|$)",
        r"请打开\s*(.+?)(?:[。！？\n]|$)",
        r"(?i)\b(?:open|launch)\s+(?:the\s+)?(.+?)(?:[.!?\n]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            phrase = _clean_target_phrase(m.group(1))
            if phrase and len(phrase) <= 80:
                return phrase
    return None


def resolve_open_app(phrase: str) -> Optional[Dict[str, str]]:
    """
    Map user phrase to {"canonical": id} or {"path": abs_path}, or None if not allowed.
    Longest alias match wins (handles multi-word aliases).
    """
    phrase = _clean_target_phrase(phrase)
    if not phrase:
        return None
    pl = phrase.lower()
    # User path overrides: exact line alias
    if pl in _USER_ALLOWLIST and _USER_ALLOWLIST[pl].startswith("path:"):
        path = _USER_ALLOWLIST[pl].split(":", 1)[1].strip().strip('"')
        if path and os.path.isfile(path):
            return {"path": os.path.abspath(path)}
        return None
    # Exact alias (phrase as a whole)
    if pl in _ALIAS_TO_CANONICAL:
        return {"canonical": _ALIAS_TO_CANONICAL[pl]}
    if phrase in _DEFAULT_ALIAS_TO_CANONICAL:
        return {"canonical": _DEFAULT_ALIAS_TO_CANONICAL[phrase]}
    # Longest substring match: alias contained in phrase or phrase in alias
    aliases = sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    for alias in aliases:
        if len(alias) < 2:
            continue
        if alias in pl or pl in alias:
            return {"canonical": _ALIAS_TO_CANONICAL[alias]}
    return None


def launch_open_app(params: Dict[str, str]) -> Tuple[bool, str]:
    """
    Launch from resolve_open_app() dict: canonical or path.
    Returns (success, user-visible message in Chinese).
    """
    if sys.platform != "win32":
        return False, "打开程序功能仅在 Windows 下可用。"

    if params.get("path"):
        path = params["path"]
        if not path or not os.path.isfile(path):
            return False, "找不到允许列表里配置的程序文件，请检查 open_app_allowlist.txt。"
        try:
            os.startfile(path)
            return True, f"已启动：{os.path.basename(path)}"
        except OSError as e:
            return False, f"无法启动该程序：{e}"

    canonical = params.get("canonical") or ""
    spec = _CANONICAL_LAUNCH.get(canonical)
    if not spec:
        return False, "未知的程序代号。"

    kind, arg = spec[0], spec[1] if len(spec) > 1 else ""

    try:
        if kind == "exe":
            exe = _win_exe(arg)
            if not exe:
                return False, f"未在系统中找到 {arg}，可能未安装。"
            subprocess.Popen([exe], close_fds=True)
            return True, f"已启动：{canonical}"
        if kind == "edge":
            exe = _edge_exe()
            if not exe:
                return False, "未找到 Microsoft Edge。"
            subprocess.Popen([exe], close_fds=True)
            return True, "已启动 Edge 浏览器。"
        if kind == "chrome":
            exe = _chrome_exe()
            if not exe:
                return False, "未找到 Google Chrome。"
            subprocess.Popen([exe], close_fds=True)
            return True, "已启动 Chrome 浏览器。"
        if kind == "steam":
            exe = _steam_exe()
            if not exe:
                return False, (
                    "未找到 Steam（常见路径未检测到）。"
                    "可在 program/open_app_allowlist.txt 中添加：steam=C:\\你的路径\\steam.exe"
                )
            subprocess.Popen([exe], close_fds=True)
            return True, "已启动 Steam。"
        if kind == "startfile":
            os.startfile(arg)
            return True, "已打开设置。"
    except OSError as e:
        return False, f"启动失败：{e}"

    return False, "启动失败。"
