"""
View stored conversations (Replay + RAG)
Run from Reorganize directory: python -m ai.memory.view_memories
"""
import os
import sys
import json

# Add parent to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REORGANIZE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REORGANIZE_DIR not in sys.path:
    sys.path.insert(0, _REORGANIZE_DIR)

from utils.path_config import get_current_dir


def view_replay_sessions(replay_path: str):
    """View Replay from single replay.json (sliding window)"""
    print("\n" + "=" * 60)
    print("Replay Memory (Sliding Window)")
    print("=" * 60)
    print(f"Path: {replay_path}")
    
    replay_file = os.path.join(replay_path, "replay.json")
    if not os.path.exists(replay_file):
        print("  [No replay.json found]")
        return
    
    try:
        with open(replay_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        turns = data.get("turns", [])
        total_tokens = data.get("total_tokens", 0)
        max_turns = data.get("max_turns", "?")
        updated = data.get("updated_at", "?")
        
        print(f"  Turns: {len(turns)} / max {max_turns}, Tokens: {total_tokens}")
        print(f"  Updated: {updated}\n")
        
        for j, turn in enumerate(turns[:10], 1):  # Show first 10 turns
            user = turn.get("user_input", "")[:80]
            resp = turn.get("assistant_response", "")[:80]
            print(f"  [{j}] User: {user}...")
            print(f"      Assistant: {resp}...")
        if len(turns) > 10:
            print(f"  ... and {len(turns) - 10} more turns")
    except Exception as e:
        print(f"  [Error reading replay.json: {e}]")


def view_rag_memories(memory_path: str):
    """View RAG memories from ChromaDB"""
    print("\n" + "=" * 60)
    print("RAG Memory (Long-term Vector Store)")
    print("=" * 60)
    print(f"Path: {memory_path}")
    
    try:
        from .memory_vector_store import MemoryVectorStore
    except ImportError:
        print("  [ChromaDB not installed. Run: pip install chromadb]")
        return
    
    if not os.path.exists(memory_path):
        print("  [No memory_db directory found]")
        return
    
    try:
        store = MemoryVectorStore(persist_directory=memory_path)
        count = store.count()
        print(f"  Total memories: {count}\n")
        
        if count == 0:
            print("  [No memories stored yet]")
            return
        
        memories = store.get_all_memories(limit=20)
        for i, mem in enumerate(memories, 1):
            meta = mem.get("metadata", {})
            user_input = meta.get("user_input", "")[:100]
            assistant_response = meta.get("assistant_response", "")[:100]
            timestamp = meta.get("timestamp", "?")
            mem_id = mem.get("id", "?")
            
            print(f"--- Memory {i}: {mem_id} ---")
            print(f"  Time: {timestamp}")
            print(f"  User: {user_input}...")
            print(f"  Assistant: {assistant_response}...")
            print()
        
        if count > 20:
            print(f"  ... and {count - 20} more memories")
    except Exception as e:
        print(f"  [Error: {e}]")
        import traceback
        traceback.print_exc()


def main():
    current_dir = get_current_dir()
    memory_path = os.path.join(current_dir, "memory_db")
    replay_path = os.path.join(current_dir, "replay_db")
    
    print("\nStored Conversations Viewer")
    print(f"Base directory: {current_dir}")
    
    view_replay_sessions(replay_path)
    view_rag_memories(memory_path)
    
    print("\n" + "=" * 60)
    print("Tip: Replay = sliding window (replay.json); RAG = long-term vector store")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
