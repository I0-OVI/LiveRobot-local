"""
Look up Windows executable paths (PATH + App Paths registry) and optionally
append 别名=路径 to program/open_app_allowlist.txt for the chatbot open-app tool.

Run from repository root or from program/:

    cd program
    python tools/app_path_lookup.py wechat
    python tools/app_path_lookup.py steam --write --alias steam
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from typing import Dict, List, Tuple

# Allow importing utils when run as script
_PROGRAM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROGRAM_ROOT not in sys.path:
    sys.path.insert(0, _PROGRAM_ROOT)

from utils.path_config import get_open_app_allowlist_path  # noqa: E402


def _iter_app_paths_registry() -> List[Tuple[str, str]]:
    """Enumerate (exe_name, resolved_path) from App Paths registry."""
    try:
        import winreg
    except ImportError:
        return []

    results: List[Tuple[str, str]] = []
    hives_bases = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"),
    ]
    for hive, base in hives_bases:
        try:
            key = winreg.OpenKey(hive, base)
        except OSError:
            continue
        i = 0
        while True:
            try:
                subname = winreg.EnumKey(key, i)
                i += 1
            except OSError:
                break
            if not subname.lower().endswith(".exe"):
                continue
            try:
                sk = winreg.OpenKey(key, subname)
                try:
                    default, _ = winreg.QueryValueEx(sk, "")
                except OSError:
                    default = ""
                finally:
                    winreg.CloseKey(sk)
                if default and os.path.isfile(default):
                    results.append((subname, os.path.normpath(default)))
            except OSError:
                pass
        winreg.CloseKey(key)
    return results


def _which_candidates(q: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    base = q.strip()
    if not base:
        return out
    variants = [base, base + ".exe", base + ".EXE"]
    if base.lower().endswith(".exe"):
        variants.append(base)
    seen = set()
    for v in variants:
        p = shutil.which(v)
        if p and os.path.isfile(p):
            np = os.path.normpath(p)
            key = (os.path.basename(np).lower(), np.lower())
            if key not in seen:
                seen.add(key)
                out.append((os.path.basename(np), np))
    return out


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip()).lower()


def find_executables(query: str) -> List[Dict[str, str]]:
    """
    Return list of {exe, path, source} matching query (substring on exe name or path).
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

    for exe_name, path in _iter_app_paths_registry():
        pl = path.lower()
        if nq in exe_name.lower() or nq in pl:
            if pl not in seen_path:
                seen_path.add(pl)
                matches.append({"exe": exe_name, "path": path, "source": "App Paths"})

    matches.sort(key=lambda x: (len(x["exe"]), x["exe"].lower()))
    return matches


def _read_allowlist_entries(path: str) -> Dict[str, str]:
    """alias_lower -> full line (without trailing newline) for non-comment lines with =."""
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


def write_allowlist_line(alias: str, exe_path: str, force: bool = False) -> Tuple[bool, str]:
    """
    Append 别名=路径 to open_app_allowlist.txt, or replace existing alias if force.
    """
    path = get_open_app_allowlist_path()
    exe_path = os.path.normpath(exe_path)
    if not exe_path.lower().endswith(".exe") or not os.path.isfile(exe_path):
        return False, f"不是有效的 .exe 文件：{exe_path}"

    alias = alias.strip()
    if not alias or "=" in alias or "|" in alias:
        return False, "别名无效（不要包含 = 或 |）。"

    line = f"{alias}={exe_path}"
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
    if sys.platform != "win32":
        print("此工具仅在 Windows 下可用。", file=sys.stderr)
        return 1

    p = argparse.ArgumentParser(description="查找应用 .exe 路径并写入 open_app_allowlist.txt")
    p.add_argument("query", help="关键词（可执行文件名的一部分或路径片段，如 wechat、steam）")
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
        print(f"未找到与 {args.query!r} 匹配的已安装 .exe（已查 PATH 与 App Paths 注册表）。")
        print("可改用「别名=完整路径」手动编辑 open_app_allowlist.txt。")
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
