"""
Look up macOS application bundles (PATH + /Applications + Spotlight) and optionally
append 别名=路径 to program_mac/open_app_allowlist.txt for the chatbot open-app tool.

Run from repository root or from program_mac/:

    cd program_mac
    python tools/app_path_lookup.py wechat
    python tools/app_path_lookup.py steam --write --alias steam
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

_PROGRAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_ROOT not in sys.path:
    sys.path.insert(0, _PROGRAM_ROOT)

from utils.path_config import get_open_app_allowlist_path  # noqa: E402


def _app_roots() -> List[str]:
    home = os.path.expanduser("~")
    return [
        "/Applications",
        "/System/Applications",
        os.path.join(home, "Applications"),
        "/System/Applications/Utilities",
    ]


def _iter_application_bundles() -> List[Tuple[str, str]]:
    """List (bundle_folder_name, abs_path_to_bundle) for *.app under standard dirs."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for root in _app_roots():
        if not os.path.isdir(root):
            continue
        try:
            for name in os.listdir(root):
                if not name.endswith(".app"):
                    continue
                p = os.path.join(root, name)
                if not os.path.isdir(p):
                    continue
                ap = os.path.abspath(p)
                if ap.lower() not in seen:
                    seen.add(ap.lower())
                    out.append((name, ap))
        except OSError:
            continue
    return out


def _mdfind_apps(query: str) -> List[Tuple[str, str]]:
    """Spotlight: return (basename, path) for application bundles."""
    q = query.strip()
    if not q:
        return []
    try:
        proc = subprocess.run(
            [
                "mdfind",
                f"kMDItemContentType == 'com.apple.application-bundle' && "
                f"(kMDItemDisplayName == '*{q}*'c || kMDItemFSName == '*{q}*'c)",
            ],
            capture_output=True,
            text=True,
            timeout=12,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    out: List[Tuple[str, str]] = []
    seen = set()
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.endswith(".app"):
            continue
        if not os.path.isdir(line):
            continue
        base = os.path.basename(line)
        ap = os.path.abspath(line)
        if ap.lower() not in seen:
            seen.add(ap.lower())
            out.append((base, ap))
    return out


def _which_candidates(q: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    base = q.strip()
    if not base:
        return out
    for v in (base, base + ".app"):
        p = shutil.which(v)
        if p and os.path.isfile(p):
            np = os.path.normpath(p)
            out.append((os.path.basename(np), np))
    return out


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip()).lower()


def find_executables(query: str) -> List[Dict[str, str]]:
    """
    Return list of {exe, path, source} matching query (substring on bundle name or path).
    """
    nq = _normalize_query(query)
    if not nq:
        return []

    matches: List[Dict[str, str]] = []
    seen_path = set()

    for exe_name, path in _which_candidates(query):
        pl = path.lower()
        if nq in exe_name.lower() or nq in pl:
            if pl not in seen_path:
                seen_path.add(pl)
                matches.append({"exe": exe_name, "path": path, "source": "PATH"})

    for bundle_name, path in _iter_application_bundles():
        pl = path.lower()
        bn = bundle_name.lower()
        if nq in bn or nq in pl:
            if pl not in seen_path:
                seen_path.add(pl)
                matches.append({"exe": bundle_name, "path": path, "source": "Applications"})

    for bundle_name, path in _mdfind_apps(query):
        pl = path.lower()
        if pl not in seen_path:
            seen_path.add(pl)
            matches.append({"exe": bundle_name, "path": path, "source": "Spotlight"})

    matches.sort(key=lambda x: (len(x["exe"]), x["exe"].lower()))
    return matches


def _read_allowlist_entries(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n\r")
                s = raw.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                left, _ = s.split("=", 1)
                for a in left.split("|"):
                    key = a.strip().lower()
                    if key:
                        out[key] = raw
    except OSError:
        pass
    return out


def _is_valid_mac_launch_target(path: str) -> bool:
    p = os.path.normpath(os.path.abspath(os.path.expanduser(path)))
    if p.endswith(".app") and os.path.isdir(p):
        return True
    return os.path.isfile(p) and os.access(p, os.X_OK)


def write_allowlist_line(alias: str, target_path: str, force: bool = False) -> Tuple[bool, str]:
    """
    Append 别名=路径 to open_app_allowlist.txt, or replace existing alias if force.
    """
    path = get_open_app_allowlist_path()
    target_path = os.path.normpath(os.path.abspath(os.path.expanduser(target_path.strip())))
    if not _is_valid_mac_launch_target(target_path):
        return False, f"不是有效的 .app 包或可执行文件：{target_path}"

    alias = alias.strip()
    if not alias or "=" in alias or "|" in alias:
        return False, "别名无效（不要包含 = 或 |）。"

    line = f"{alias}={target_path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    entries = _read_allowlist_entries(path)
    key = alias.lower()
    if key in entries and not force:
        return False, f"别名已存在：{alias}（使用 --force 覆盖）"

    if key in entries and force:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            return False, str(e)
        new_lines: List[str] = []
        replaced = False
        for ln in lines:
            raw = ln.rstrip("\n\r")
            st = raw.strip()
            if st and not st.startswith("#") and "=" in st:
                left, _ = st.split("=", 1)
                if any(a.strip().lower() == key for a in left.split("|")):
                    new_lines.append(line + "\n")
                    replaced = True
                    continue
            new_lines.append(ln)
        if not replaced:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] = new_lines[-1].rstrip("\n") + "\n"
            new_lines.append(line + "\n")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except OSError as e:
            return False, str(e)
        return True, f"已更新 {path}\n{line}"

    prefix = ""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "rb") as fb:
                fb.seek(-1, os.SEEK_END)
                if fb.read(1) != b"\n":
                    prefix = "\n"
        except OSError:
            prefix = "\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(prefix + line + "\n")
    except OSError as e:
        return False, str(e)

    return True, f"已写入 {path}\n{line}"


def main() -> int:
    if sys.platform != "darwin":
        print("此工具仅在 macOS 下可用。", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="查找应用 .app 路径并写入 open_app_allowlist.txt")
    p.add_argument("query", help="关键词（应用名的一部分，如 wechat、steam）")
    p.add_argument(
        "--write",
        action="store_true",
        help="将选中的结果写入 open_app_allowlist.txt",
    )
    p.add_argument(
        "--alias",
        default=None,
        help="写入时使用的别名（默认与 query 相同）",
    )
    p.add_argument(
        "--pick",
        type=int,
        default=1,
        metavar="N",
        help="多条结果时选第 N 条（从 1 开始，默认 1）",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="若别名已存在则覆盖该行",
    )
    args = p.parse_args()

    found = find_executables(args.query)
    if not found:
        print(f"未找到与 {args.query!r} 匹配的应用（已查 PATH、Applications 与 Spotlight）。")
        print("可改用「别名=/Applications/Your.app」手动编辑 open_app_allowlist.txt。")
        return 2

    print(f"共 {len(found)} 条（--pick 选择序号）：\n")
    for i, item in enumerate(found, 1):
        print(f"  [{i}] ({item['source']}) {item['exe']}")
        print(f"      {item['path']}\n")

    if not args.write:
        print("若需写入白名单，请加：  --write [--alias 显示名] [--pick N]")
        return 0

    idx = args.pick - 1
    if idx < 0 or idx >= len(found):
        print("--pick 超出范围。", file=sys.stderr)
        return 1

    chosen = found[idx]
    alias = args.alias if args.alias is not None else args.query.strip()
    ok, msg = write_allowlist_line(alias, chosen["path"], force=args.force)
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
