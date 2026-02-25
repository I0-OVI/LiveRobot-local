"""
RAG 测试/对比程序：独立于 Qwen 输出模型，用于比较「使用 RAG」与「未使用 RAG」的差异，并展示 RAG 检索情况。
不加载任何 LLM，仅使用 RAG 的检索与 enhanced_prompt 构建逻辑。
使用方式（在 Reorganize 目录下）：
  python tools/rag_test_compare.py "你的查询"
  python tools/rag_test_compare.py --query "你的查询" [--top-k 5] [--threshold 0.7]
  python tools/rag_test_compare.py --interactive   # 交互式输入多轮查询
"""
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REORGANIZE_DIR = os.path.dirname(_SCRIPT_DIR)
if _REORGANIZE_DIR not in sys.path:
    sys.path.insert(0, _REORGANIZE_DIR)

from utils.path_config import get_current_dir


def get_memory_path():
    return os.path.join(get_current_dir(), "memory_db")


def build_enhanced_prompt(rag_memories: list) -> str:
    """与 MemoryCoordinator 一致：根据检索到的 RAG 记忆构建 enhanced_prompt 片段"""
    if not rag_memories:
        return ""
    memory_lines = []
    for i, memory in enumerate(rag_memories, 1):
        metadata = memory.get("metadata", {})
        mem_user = metadata.get("user_input", "")
        mem_assistant = metadata.get("assistant_response", "")
        memory_lines.append(f"{i}. {mem_user} -> {mem_assistant}")
    memory_text = "\n".join(memory_lines)
    return f"\n相关记忆：\n{memory_text}\n"


def run_one_query(rag, query: str, top_k: int = 3, similarity_threshold: float = 0.7) -> dict:
    """
    对单条 query 执行：无 RAG 上下文、有 RAG 检索、enhanced_prompt、检索详情。
    返回 dict：no_rag_prompt, with_rag_prompt, rag_memories, enhanced_prompt_snippet, retrieval_details
    """
    # 无 RAG：不注入任何记忆
    no_rag_prompt = "(无 RAG：不注入任何相关记忆)\n"

    # 有 RAG：检索
    rag_memories = rag.get_relevant_memories(
        query_text=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # 检索详情（用于报告）
    retrieval_details = []
    for m in rag_memories:
        retrieval_details.append({
            "id": m.get("id"),
            "similarity": m.get("similarity"),
            "distance": m.get("distance"),
            "user_input": (m.get("metadata") or {}).get("user_input", ""),
            "assistant_response": (m.get("metadata") or {}).get("assistant_response", ""),
            "document": (m.get("document", ""))[:200],
        })

    enhanced_prompt_snippet = build_enhanced_prompt(rag_memories)
    with_rag_prompt = no_rag_prompt + ("(有 RAG：以下将注入到 system prompt 的「相关记忆」部分)\n" + enhanced_prompt_snippet if enhanced_prompt_snippet else "(有 RAG：但未检索到满足阈值的记忆，注入为空)\n")

    return {
        "no_rag_prompt": no_rag_prompt,
        "with_rag_prompt": with_rag_prompt,
        "rag_memories": rag_memories,
        "enhanced_prompt_snippet": enhanced_prompt_snippet,
        "retrieval_details": retrieval_details,
    }


def print_report(query: str, result: dict, top_k: int, threshold: float):
    """打印单次查询的对比报告与检索情况"""
    print("\n" + "=" * 60)
    print("查询 (Query)")
    print("=" * 60)
    print(query)
    print()

    print("=" * 60)
    print("RAG 检索情况 (top_k={}, similarity_threshold={})".format(top_k, threshold))
    print("=" * 60)
    details = result["retrieval_details"]
    if not details:
        print("  未检索到满足阈值的记忆。")
    else:
        for i, d in enumerate(details, 1):
            print(f"  [{i}] id={d['id']}")
            print(f"      similarity={d['similarity']:.4f}, distance={d['distance']:.4f}")
            print(f"      user_input: {d['user_input'][:100]}...")
            print(f"      assistant_response: {d['assistant_response'][:100]}...")
            print()
    print()

    print("=" * 60)
    print("对比：未使用 RAG vs 使用 RAG（注入到 system 的「相关记忆」部分）")
    print("=" * 60)
    print("--- 未使用 RAG 时（传给模型的上下文） ---")
    print(result["no_rag_prompt"])
    print("--- 使用 RAG 时（将以下片段拼接到 system prompt） ---")
    print(result["with_rag_prompt"])
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 测试对比：比较使用 RAG 与未使用 RAG 的差异，并展示检索情况（不加载 Qwen）"
    )
    parser.add_argument("query", nargs="?", type=str, default=None, help="单次查询内容")
    parser.add_argument("--query", dest="query_opt", type=str, default=None, help="或用 --query 传入查询")
    parser.add_argument("--top-k", type=int, default=3, help="RAG 检索 top_k")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值")
    parser.add_argument("--interactive", action="store_true", help="交互式输入多轮查询")
    args = parser.parse_args()

    memory_path = get_memory_path()
    if not os.path.exists(memory_path):
        print(f"[RAG 测试] 未找到 memory_db 目录: {memory_path}")
        print("请先运行主程序或使用 tools/rag_manage.py add 写入若干条 RAG 后再测试。")
        sys.exit(1)

    try:
        from ai.memory.rag_memory import RAGMemory
    except ImportError as e:
        print(f"[RAG 测试] 无法导入 RAGMemory: {e}")
        sys.exit(1)

    rag = RAGMemory(
        persist_directory=memory_path,
        collection_name="conversation_memories",
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )
    count = rag.count()
    print(f"[RAG 测试] 已加载 RAG 库，当前记忆数: {count}")

    if args.interactive:
        print("进入交互模式，输入查询后回车查看对比与检索情况；空行退出。\n")
        while True:
            try:
                q = input("Query> ").strip()
            except EOFError:
                break
            if not q:
                break
            result = run_one_query(rag, q, top_k=args.top_k, similarity_threshold=args.threshold)
            print_report(q, result, args.top_k, args.threshold)
        return

    query = args.query or args.query_opt
    if not query:
        parser.print_help()
        print("\n请提供 query，例如: python tools/rag_test_compare.py \"你好，我叫小明\"")
        sys.exit(0)

    result = run_one_query(rag, query, top_k=args.top_k, similarity_threshold=args.threshold)
    print_report(query, result, args.top_k, args.threshold)


if __name__ == "__main__":
    main()
