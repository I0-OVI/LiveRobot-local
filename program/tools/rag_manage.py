"""
Manual RAG management utility for adding/removing memories.
Uses the same memory_db as the main app and writes/deletes directly
without importance filtering.
Usage (run inside Reorganize):
  python tools/rag_manage.py add "user question or keywords" "assistant answer or knowledge text"
  python tools/rag_manage.py delete <memory_id>
  python tools/rag_manage.py list [--limit N] [--offset O]
  python tools/rag_manage.py clear
"""
import os
import sys
import uuid
import argparse
from datetime import datetime

# Add Reorganize to sys.path for ai/utils imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REORGANIZE_DIR = os.path.dirname(_SCRIPT_DIR)
if _REORGANIZE_DIR not in sys.path:
    sys.path.insert(0, _REORGANIZE_DIR)

from utils.path_config import get_current_dir


def get_memory_path():
    """Use the same default memory_db location as main."""
    current_dir = get_current_dir()
    return os.path.join(current_dir, "memory_db")


def create_store(persist_directory: str, collection_name: str = "conversation_memories"):
    """Create MemoryVectorStore (no embedder needed)."""
    from ai.memory.memory_vector_store import MemoryVectorStore

    store = MemoryVectorStore(persist_directory=persist_directory, collection_name=collection_name)
    return store


def create_embedder():
    """Create MemoryEmbedder (only needed for add)."""
    from ai.memory.memory_embedder import MemoryEmbedder
    return MemoryEmbedder()


def cmd_add(store, embedder, user_input: str, assistant_response: str, importance: float = 1.0) -> str:
    """Manually add one RAG memory (without importance filtering)."""
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
        raise RuntimeError("Failed to write RAG memory")
    return memory_id


def cmd_delete(store, memory_id: str) -> bool:
    """Delete one RAG memory by ID."""
    return store.delete_memory(memory_id)


def cmd_list(store, limit: int = 50, offset: int = 0) -> list:
    """List RAG memories with pagination."""
    all_memories = store.get_all_memories(limit=limit + offset)
    return all_memories[offset : offset + limit]


def cmd_clear(store) -> bool:
    """Clear all RAG memories."""
    return store.clear()


def cmd_merge(persist_directory: str, collection_name: str = "conversation_memories") -> int:
    """Run one manual merge pass for similar memories."""
    from ai.memory.rag_memory import RAGMemory
    rag = RAGMemory(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    return rag.merge_similar_memories()


def main():
    parser = argparse.ArgumentParser(
        description="Manual RAG management: add/delete/list/clear/merge (shared memory_db with main)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add: manual insert
    p_add = subparsers.add_parser("add", help="Add one RAG memory (question/keywords + answer/content)")
    p_add.add_argument("user_input", type=str, help="User question or keywords (used for retrieval)")
    p_add.add_argument("assistant_response", type=str, help="Assistant answer or knowledge content")
    p_add.add_argument("--importance", type=float, default=1.0, help="Importance score 0.0~1.0 (default: 1.0)")

    # delete: by memory ID
    p_del = subparsers.add_parser("delete", help="Delete one memory by memory_id")
    p_del.add_argument("memory_id", type=str, help="Memory ID to delete")

    # list: paginated listing
    p_list = subparsers.add_parser("list", help="List RAG memories (paginated)")
    p_list.add_argument("--limit", type=int, default=50, help="Max number of rows to display")
    p_list.add_argument("--offset", type=int, default=0, help="Skip first N rows")

    # clear and merge
    subparsers.add_parser("clear", help="Clear all RAG memories (use with caution)")
    subparsers.add_parser("merge", help="Manually merge similar memories (slower, run occasionally)")

    args = parser.parse_args()
    memory_path = get_memory_path()
    collection_name = "conversation_memories"

    if args.command == "clear":
        try:
            store = create_store(memory_path, collection_name)
            if cmd_clear(store):
                print("[RAG Manage] Cleared all RAG memories.")
            else:
                print("[RAG Manage] Failed to clear memories.")
        except Exception as e:
            print(f"[RAG Manage] Error: {e}")
            sys.exit(1)
        return

    if args.command == "merge":
        try:
            merged_count = cmd_merge(memory_path, collection_name)
            print(f"[RAG Manage] Merge completed, processed count: {merged_count}")
        except Exception as e:
            print(f"[RAG Manage] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "list":
        try:
            store = create_store(memory_path, collection_name)
            total = store.count()
            memories = cmd_list(store, limit=args.limit, offset=args.offset)
            print(f"[RAG Manage] Total: {total} (page limit={args.limit}, offset={args.offset})\n")
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
            print(f"[RAG Manage] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "delete":
        try:
            store = create_store(memory_path, collection_name)
            if cmd_delete(store, args.memory_id):
                print(f"[RAG Manage] Deleted: {args.memory_id}")
            else:
                print(f"[RAG Manage] Delete failed or ID not found: {args.memory_id}")
                sys.exit(1)
        except Exception as e:
            print(f"[RAG Manage] Error: {e}")
            sys.exit(1)
        return

    if args.command == "add":
        try:
            store = create_store(memory_path, collection_name)
            embedder = create_embedder()
            mid = cmd_add(
                store, embedder,
                user_input=args.user_input,
                assistant_response=args.assistant_response,
                importance=getattr(args, "importance", 1.0),
            )
            print(f"[RAG Manage] Added, memory_id={mid}")
        except Exception as e:
            print(f"[RAG Manage] Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
