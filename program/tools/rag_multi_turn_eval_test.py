"""
多轮对话 RAG 纳入测试：加载本地 Qwen，调用 RAGSaveEvaluator.evaluate_multi_turn，
输出「是否应写入长期记忆」及重要性、原因。

用法（在 program 目录下）:
  python tools/rag_multi_turn_eval_test.py
  python tools/rag_multi_turn_eval_test.py --file transcript.txt
  python tools/rag_multi_turn_eval_test.py --llm-preset qwen3.5-4b

对话文本格式（每行一条，前缀不区分大小写）:
  U: 或 用户:  — 用户消息
  A: 或 助手:  — 助手消息
文件或交互结束时用单独一行 END 表示结束输入（交互模式）；文件不需要 END。
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAM_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROGRAM_DIR not in sys.path:
    sys.path.insert(0, _PROGRAM_DIR)

from utils.path_config import get_current_dir
from utils.llm_presets import LLM_PRESETS, resolve_llm_preset
from ai.text_generator import QwenTextGenerator
from ai.memory.rag_save_evaluator import RAGSaveEvaluator


_U_PREFIX = re.compile(r"^\s*(?:U|用户|user)\s*[:：]\s*(.*)$", re.IGNORECASE | re.DOTALL)
_A_PREFIX = re.compile(r"^\s*(?:A|助手|assistant)\s*[:：]\s*(.*)$", re.IGNORECASE | re.DOTALL)


def parse_transcript_text(text: str) -> list[tuple[str, str]]:
    """
    Parse lines like 'U: ...' / 'A: ...' into ordered (user, assistant) pairs.
    """
    messages: list[tuple[str, str]] = []  # ("user"|"assistant", content)
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        mu = _U_PREFIX.match(line)
        ma = _A_PREFIX.match(line)
        if mu:
            messages.append(("user", mu.group(1)))
        elif ma:
            messages.append(("assistant", ma.group(1)))
        else:
            raise ValueError(
                f"无法解析的行（需要 U:/用户: 或 A:/助手: 前缀）: {line[:80]!r}"
            )

    if not messages:
        raise ValueError("没有有效的对话行")

    turns: list[tuple[str, str]] = []
    i = 0
    while i < len(messages):
        role, content = messages[i]
        if role == "user":
            u = content
            if i + 1 < len(messages) and messages[i + 1][0] == "assistant":
                turns.append((u, messages[i + 1][1]))
                i += 2
            else:
                turns.append((u, ""))
                i += 1
        else:
            print(
                f"[警告] 跳过开头的助手消息（缺少对应的用户行）: {content[:60]!r}...",
                file=sys.stderr,
            )
            i += 1

    if not turns:
        raise ValueError("解析后没有有效的用户轮次")
    return turns


def read_transcript_from_path(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def interactive_collect() -> str:
    print(
        "输入多轮对话，每行一条：\n"
        "  U: … 或 用户: …  — 用户\n"
        "  A: … 或 助手: …  — 助手\n"
        "输入完成后单独一行键入 END 并回车（或 Ctrl+Z 再回车结束）。\n"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def print_result(res) -> None:
    yn = "是" if res.store_long_term else "否"
    print()
    print("=" * 56)
    print("RAG 纳入判定")
    print("=" * 56)
    print(f"是否应写入长期记忆（RAG）: {yn}")
    print(f"重要性 (0–1): {res.importance:.4f}")
    if res.tags:
        print(f"标签: {', '.join(res.tags)}")
    print(f"原因: {res.reason}")
    print(f"来源: {res.source}")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多轮对话 RAG 纳入测试（需加载本地 Qwen）"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="对话文本文件（UTF-8），格式见脚本顶部说明",
    )
    parser.add_argument(
        "--llm-preset",
        type=str,
        default="qwen2.5-7b",
        help="LLM 预设 id（默认 qwen2.5-7b）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="单次评估超时秒数（默认 120）",
    )
    args = parser.parse_args()

    current = get_current_dir()
    resolved_name, resolved_cache, trust_rc, used_preset = resolve_llm_preset(
        args.llm_preset,
        model_name_override=None,
        model_cache_dir_override=None,
        current_dir_for_models=current,
    )
    _preset_cfg = LLM_PRESETS.get(used_preset, {})
    hub_pq = bool(_preset_cfg.get("hub_pre_quantized", False))

    print(f"[RAG Multi-turn Test] 预设: {used_preset} → {resolved_name}")
    print(f"[RAG Multi-turn Test] 缓存目录: {os.path.abspath(resolved_cache)}")
    print("[RAG Multi-turn Test] 正在加载模型（首次可能较慢）…")

    gen = QwenTextGenerator(
        model_name=resolved_name,
        cache_dir=resolved_cache,
        trust_remote_code=trust_rc,
        hub_pre_quantized=hub_pq,
    )
    evaluator = RAGSaveEvaluator(text_generator=gen)

    if args.file:
        raw = read_transcript_from_path(args.file)
    else:
        raw = interactive_collect()

    try:
        turns = parse_transcript_text(raw)
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)

    print("\n--- 解析后的轮次 ---")
    for i, (u, a) in enumerate(turns, 1):
        print(f"第{i}轮 用户: {u[:200]}{'…' if len(u) > 200 else ''}")
        print(f"      助手: {a[:200]}{'…' if len(a) > 200 else ''}")

    print("\n[RAG Multi-turn Test] 正在调用评估模型…")
    result = evaluator.evaluate_multi_turn(
        turns,
        timeout_sec=args.timeout,
    )
    print_result(result)


if __name__ == "__main__":
    main()
