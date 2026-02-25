"""
RAG 人为管理工具：负责 RAG 知识库的手动写入与删除。
与主程序共用同一 memory_db，不经过 importance 过滤，直接写入/删除。
使用方式（在 Reorganize 目录下）：
  python tools/rag_manage.py add "用户问题或关键词" "助手回答或知识内容"
  python tools/rag_manage.py delete <memory_id>
  python tools/rag_manage.py list [--limit N] [--offset O]
  python tools/rag_manage.py clear
"""
import os
import sys
import uuid
import argparse
from datetime import datetime

# 将 Reorganize 加入 path，便于引用 ai / utils
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REORGANIZE_DIR = os.path.dirname(_SCRIPT_DIR)
if _REORGANIZE_DIR not in sys.path:
    sys.path.insert(0, _REORGANIZE_DIR)

from utils.path_config import get_current_dir


def get_memory_path():
    """与 main 默认一致：Reorganize 下的 memory_db（由 path_config 的 get_current_dir 决定）"""
    current_dir = get_current_dir()
    return os.path.join(current_dir, "memory_db")


def create_store_and_embedder(persist_directory: str, collection_name: str = "conversation_memories"):
    """创建 MemoryVectorStore 和 MemoryEmbedder（用于人为写入时生成 embedding）"""
    from ai.memory.memory_vector_store import MemoryVectorStore
    from ai.memory.memory_embedder import MemoryEmbedder

    store = MemoryVectorStore(persist_directory=persist_directory, collection_name=collection_name)
    embedder = MemoryEmbedder()
    return store, embedder


def cmd_add(store, embedder, user_input: str, assistant_response: str, importance: float = 1.0) -> str:
    """人为写入一条 RAG 记忆（不经过 importance 过滤）"""
    memory_id = str(uuid.uuid4())
    combined_text = f"{user_input} {assistant_response}"
    embedding = embedder.embed_text(combined_text)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "importance": importance,
        "user_input": user_input,
        "assistant_response": assistant_response,
    }
    ok = store.add_memory(
        memory_id=memory_id,
        user_input=user_input,
        assistant_response=assistant_response,
        embedding=embedding,
        metadata=metadata,
    )
    if not ok:
        raise RuntimeError("写入 RAG 失败")
    return memory_id


def cmd_delete(store, memory_id: str) -> bool:
    """按 ID 删除一条 RAG 记忆"""
    return store.delete_memory(memory_id)


def cmd_list(store, limit: int = 50, offset: int = 0) -> list:
    """列出 RAG 记忆（分页）"""
    all_memories = store.get_all_memories(limit=limit + offset)
    return all_memories[offset : offset + limit]


def cmd_clear(store) -> bool:
    """清空所有 RAG 记忆"""
    return store.clear()


def main():
    parser = argparse.ArgumentParser(
        description="RAG 人为管理：写入/删除/列出/清空 RAG 知识库（与主程序共用 memory_db）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add: 人为写入
    p_add = subparsers.add_parser("add", help="写入一条 RAG 知识（问题/关键词 + 回答/内容）")
    p_add.add_argument("user_input", type=str, help="用户问题或关键词（用于检索）")
    p_add.add_argument("assistant_response", type=str, help="助手回答或知识内容")
    p_add.add_argument("--importance", type=float, default=1.0, help="重要性 0.0~1.0，默认 1.0")

    # delete: 按 ID 删除
    p_del = subparsers.add_parser("delete", help="按 memory_id 删除一条")
    p_del.add_argument("memory_id", type=str, help="要删除的记忆 ID")

    # list: 列出
    p_list = subparsers.add_parser("list", help="列出 RAG 记忆（分页）")
    p_list.add_argument("--limit", type=int, default=50, help="最多显示条数")
    p_list.add_argument("--offset", type=int, default=0, help="跳过前 N 条")

    # clear: 清空
    subparsers.add_parser("clear", help="清空所有 RAG 记忆（谨慎使用）")

    args = parser.parse_args()
    memory_path = get_memory_path()
    collection_name = "conversation_memories"

    if args.command == "clear":
        try:
            store, _ = create_store_and_embedder(memory_path, collection_name)
            if cmd_clear(store):
                print("[RAG 管理] 已清空所有 RAG 记忆。")
            else:
                print("[RAG 管理] 清空失败。")
        except Exception as e:
            print(f"[RAG 管理] 错误: {e}")
            sys.exit(1)
        return

    if args.command == "list":
        try:
            store, _ = create_store_and_embedder(memory_path, collection_name)
            total = store.count()
            memories = cmd_list(store, limit=args.limit, offset=args.offset)
            print(f"[RAG 管理] 共 {total} 条（本页 limit={args.limit}, offset={args.offset}）\n")
            for i, m in enumerate(memories, 1):
                mid = m.get("id", "?")
                meta = m.get("metadata", {})
                ui = (meta.get("user_input") or "")[:80]
                ar = (meta.get("assistant_response") or "")[:80]
                ts = meta.get("timestamp", "?")
                print(f"  [{i}] id={mid}")
                print(f"      time={ts}")
                print(f"      user: {ui}...")
                print(f"      asst: {ar}...")
                print()
        except Exception as e:
            print(f"[RAG 管理] 错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "delete":
        try:
            store, _ = create_store_and_embedder(memory_path, collection_name)
            if cmd_delete(store, args.memory_id):
                print(f"[RAG 管理] 已删除: {args.memory_id}")
            else:
                print(f"[RAG 管理] 删除失败或 ID 不存在: {args.memory_id}")
                sys.exit(1)
        except Exception as e:
            print(f"[RAG 管理] 错误: {e}")
            sys.exit(1)
        return

    if args.command == "add":
        try:
            store, embedder = create_store_and_embedder(memory_path, collection_name)
            mid = cmd_add(
                store, embedder,
                user_input=args.user_input,
                assistant_response=args.assistant_response,
                importance=getattr(args, "importance", 1.0),
            )
            print(f"[RAG 管理] 已写入，memory_id={mid}")
        except Exception as e:
            print(f"[RAG 管理] 错误: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
