"""
Whitelist for keyword-triggered "open app" tool (macOS).
Built-in aliases + optional program_mac/open_app_allowlist.txt (alias=path or alias=canonical).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# --- Optional user file: one entry per line ---
#   别名=/Applications/Foo.app
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
                rl = right.strip().strip('"')
                is_win_path = rl.lower().endswith(".exe") or ":\\" in rl or rl.startswith("\\\\")
                is_mac_path = rl.startswith("/") or rl.endswith(".app")
                if is_win_path or is_mac_path:
                    payload = f"path:{rl}"
                else:
                    payload = f"canon:{rl.lower().strip()}"
                for alias in left.split("|"):
                    a = alias.strip()
                    if a:
                        out[a.lower()] = payload
    except OSError:
        pass
    return out


_USER_ALLOWLIST = _load_user_allowlist()


def _app_search_roots() -> List[str]:
    home = os.path.expanduser("~")
    return [
        "/Applications",
        "/System/Applications",
        os.path.join(home, "Applications"),
        "/System/Applications/Utilities",
    ]


def _find_bundle_named(name: str) -> Optional[str]:
    """Find Foo.app under standard dirs (exact folder name match, case-insensitive)."""
    if not name.endswith(".app"):
        name = name + ".app"
    lower = name.lower()
    for root in _app_search_roots():
        if not os.path.isdir(root):
            continue
        try:
            for entry in os.listdir(root):
                if entry.lower() == lower:
                    p = os.path.join(root, entry)
                    if os.path.isdir(p):
                        return os.path.abspath(p)
        except OSError:
            continue
    return None


def _find_bundle_substring(token: str) -> Optional[str]:
    """First .app bundle whose name contains token (case-insensitive)."""
    tl = token.lower()
    best: Optional[Tuple[int, str]] = None
    for root in _app_search_roots():
        if not os.path.isdir(root):
            continue
        try:
            for entry in os.listdir(root):
                if not entry.endswith(".app"):
                    continue
                if tl in entry.lower():
                    p = os.path.join(root, entry)
                    if os.path.isdir(p):
                        score = len(entry)
                        if best is None or score < best[0]:
                            best = (score, os.path.abspath(p))
        except OSError:
            continue
    return best[1] if best else None


def _edge_app() -> Optional[str]:
    for cand in ("Microsoft Edge.app",):
        p = _find_bundle_named(cand)
        if p:
            return p
    return _find_bundle_substring("Microsoft Edge")


def _chrome_app() -> Optional[str]:
    p = _find_bundle_named("Google Chrome.app")
    if p:
        return p
    return _find_bundle_substring("Chrome")


def _steam_app() -> Optional[str]:
    p = _find_bundle_named("Steam.app")
    if p:
        return p
    return shutil.which("steam") or None


def _open_bundle(path: str) -> None:
    """Launch a .app bundle by path (use `open path/to/Foo.app`, not `-a`)."""
    path = os.path.abspath(path)
    subprocess.Popen(["open", path], close_fds=True)


def _open_app_by_name(name: str) -> None:
    subprocess.Popen(["open", "-a", name], close_fds=True)


# canonical_id -> (kind, arg)
# kind: bundle_path | app_name | edge | chrome | steam | open_path | open_url | screenshot
_CANONICAL_LAUNCH: Dict[str, Tuple[str, ...]] = {
    "notepad": ("app_name", "TextEdit"),
    "calc": ("bundle_path", "/System/Applications/Calculator.app"),
    "mspaint": ("bundle_path", "/System/Applications/Preview.app"),
    "explorer": ("open_path", "~"),
    "cmd": ("bundle_path", "/System/Applications/Utilities/Terminal.app"),
    "powershell": ("bundle_path", "/System/Applications/Utilities/Terminal.app"),
    "taskmgr": ("bundle_path", "/System/Applications/Utilities/Activity Monitor.app"),
    "snippingtool": ("screenshot", ""),
    "edge": ("edge", ""),
    "chrome": ("chrome", ""),
    "steam": ("steam", ""),
    "settings": ("open_url", "x-apple.systempreferences:"),
}

_DEFAULT_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "记事本": "notepad",
    "文本文档": "notepad",
    "notepad": "notepad",
    "计算器": "calc",
    "calculator": "calc",
    "calc": "calc",
    "画图": "mspaint",
    "画图程序": "mspaint",
    "mspaint": "mspaint",
    "paint": "mspaint",
    "资源管理器": "explorer",
    "文件夹": "explorer",
    "此电脑": "explorer",
    "文件资源管理器": "explorer",
    "explorer": "explorer",
    "设置": "settings",
    "系统设置": "settings",
    "settings": "settings",
    "edge": "edge",
    "edge浏览器": "edge",
    "microsoft edge": "edge",
    "chrome": "chrome",
    "谷歌浏览器": "chrome",
    "steam": "steam",
    "蒸汽": "steam",
    "steam客户端": "steam",
    "命令提示符": "cmd",
    "cmd": "cmd",
    "powershell": "powershell",
    "终端": "powershell",
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


def _path_is_valid_launch_target(path: str) -> bool:
    path = os.path.abspath(os.path.expanduser(path.strip().strip('"')))
    if os.path.isfile(path):
        return os.access(path, os.X_OK) or path.lower().endswith(".app")
    if path.endswith(".app") and os.path.isdir(path):
        return True
    return False


def resolve_open_app(phrase: str) -> Optional[Dict[str, str]]:
    """
    Map user phrase to {"canonical": id} or {"path": abs_path}, or None if not allowed.
    Longest alias match wins (handles multi-word aliases).
    """
    phrase = _clean_target_phrase(phrase)
    if not phrase:
        return None
    pl = phrase.lower()
    if pl in _USER_ALLOWLIST and _USER_ALLOWLIST[pl].startswith("path:"):
        path = _USER_ALLOWLIST[pl].split(":", 1)[1].strip().strip('"')
        path = os.path.abspath(os.path.expanduser(path))
        if _path_is_valid_launch_target(path):
            return {"path": path}
        return None
    if pl in _ALIAS_TO_CANONICAL:
        return {"canonical": _ALIAS_TO_CANONICAL[pl]}
    if phrase in _DEFAULT_ALIAS_TO_CANONICAL:
        return {"canonical": _DEFAULT_ALIAS_TO_CANONICAL[phrase]}
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
    if sys.platform != "darwin":
        return False, "打开程序功能仅在 macOS 下可用。"

    if params.get("path"):
        path = os.path.abspath(os.path.expanduser(params["path"].strip().strip('"')))
        if not _path_is_valid_launch_target(path):
            return False, "找不到允许列表里配置的程序，请检查 open_app_allowlist.txt。"
        try:
            if path.endswith(".app") and os.path.isdir(path):
                _open_bundle(path)
            else:
                subprocess.Popen([path], close_fds=True)
            return True, f"已启动：{os.path.basename(path)}"
        except OSError as e:
            return False, f"无法启动该程序：{e}"

    canonical = params.get("canonical") or ""
    spec = _CANONICAL_LAUNCH.get(canonical)
    if not spec:
        return False, "未知的程序代号。"

    kind, arg = spec[0], spec[1] if len(spec) > 1 else ""

    try:
        if kind == "bundle_path":
            p = os.path.abspath(os.path.expanduser(arg))
            if not (p.endswith(".app") and os.path.isdir(p)):
                return False, f"未找到系统应用：{arg}"
            _open_bundle(p)
            return True, f"已启动：{canonical}"
        if kind == "app_name":
            _open_app_by_name(arg)
            return True, f"已启动：{canonical}"
        if kind == "open_path":
            target = os.path.expanduser(arg)
            subprocess.Popen(["open", target], close_fds=True)
            return True, "已打开文件夹（访达）。"
        if kind == "open_url":
            subprocess.Popen(["open", arg], close_fds=True)
            return True, "已打开系统设置。"
        if kind == "screenshot":
            subprocess.Popen(["open", "-b", "com.apple.screenshot.launcher"], close_fds=True)
            return True, "已打开截图。"
        if kind == "edge":
            app = _edge_app()
            if not app:
                return False, "未找到 Microsoft Edge（请在 /Applications 安装或使用 open_app_allowlist.txt 指定路径）。"
            _open_bundle(app)
            return True, "已启动 Edge 浏览器。"
        if kind == "chrome":
            app = _chrome_app()
            if not app:
                return False, "未找到 Google Chrome。"
            _open_bundle(app)
            return True, "已启动 Chrome 浏览器。"
        if kind == "steam":
            app = _steam_app()
            if not app:
                return False, (
                    "未找到 Steam。"
                    "可在 program_mac/open_app_allowlist.txt 中添加：steam=/Applications/Steam.app"
                )
            if app.endswith(".app") or os.path.isdir(app):
                _open_bundle(app)
            else:
                subprocess.Popen([app], close_fds=True)
            return True, "已启动 Steam。"
    except OSError as e:
        return False, f"启动失败：{e}"

    return False, "启动失败。"
