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
    """View Replay session JSON files"""
    print("\n" + "=" * 60)
    print("Replay Memory (Session History)")
    print("=" * 60)
    print(f"Path: {replay_path}")
    
    if not os.path.exists(replay_path):
        print("  [No replay_db directory found]")
        return
    
    session_files = [
        f for f in os.listdir(replay_path)
        if f.endswith('.json') and f.startswith('session_')
    ]
    session_files.sort(key=lambda f: os.path.getmtime(os.path.join(replay_path, f)), reverse=True)
    
    if not session_files:
        print("  [No session files found]")
        return
    
    print(f"  Found {len(session_files)} session(s)\n")
    
    for i, fname in enumerate(session_files[:10], 1):  # Show latest 10
        fpath = os.path.join(replay_path, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            session_id = data.get("session_id", fname)
            turns = data.get("turns", [])
            total_tokens = data.get("total_tokens", 0)
            created = data.get("created_at", "?")
            
            print(f"--- Session {i}: {session_id} ---")
            print(f"  Created: {created}")
            print(f"  Turns: {len(turns)}, Tokens: {total_tokens}")
            for j, turn in enumerate(turns[:5], 1):  # Show first 5 turns
                user = turn.get("user_input", "")[:80]
                resp = turn.get("assistant_response", "")[:80]
                print(f"  [{j}] User: {user}...")
                print(f"      Assistant: {resp}...")
            if len(turns) > 5:
                print(f"  ... and {len(turns) - 5} more turns")
            print()
        except Exception as e:
            print(f"  [Error reading {fname}: {e}]\n")


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
    print("Tip: Replay = recent session history; RAG = long-term important memories")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
