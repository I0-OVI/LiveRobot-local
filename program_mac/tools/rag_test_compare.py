"""
RAG test utility independent from Qwen output model.
Shows RAG retrieval results and the enhanced_prompt snippet that would be injected.
No LLM is loaded; only retrieval and enhanced_prompt construction are used.
Usage (run inside Reorganize):
  python tools/rag_test_compare.py "your query"
  python tools/rag_test_compare.py --query "your query" [--top-k 5] [--threshold 0.7]
  python tools/rag_test_compare.py --interactive   # interactive multi-query mode
"""
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REORGANIZE_DIR = os.path.dirname(_SCRIPT_DIR)
if _REORGANIZE_DIR not in sys.path:
    sys.path.insert(0, _REORGANIZE_DIR)

from utils.path_config import get_memory_db_path


def build_enhanced_prompt(rag_memories: list, include_similarity: bool = True) -> str:
    """Build enhanced_prompt snippet from retrieved RAG memories."""
    if not rag_memories:
        return ""
    memory_lines = []
    for i, memory in enumerate(rag_memories, 1):
        sim = memory.get("similarity")
        metadata = memory.get("metadata", {})
        mem_user = metadata.get("user_input", "")
        mem_assistant = metadata.get("assistant_response", "")
        prefix = (f"[similarity {sim:.4f}] " if sim is not None else "[similarity N/A] ") if include_similarity else ""
        memory_lines.append(f"{i}. {prefix}{mem_user} -> {mem_assistant}")
    memory_text = "\n".join(memory_lines)
    return f"\nRelevant memories:\n{memory_text}\n"


def run_one_query(rag, query: str, top_k: int = 3, similarity_threshold: float = 0.7, verbose: bool = False) -> dict:
    """
    Run one query: RAG retrieval, enhanced_prompt snippet, and retrieval details.
    Returns dict with rag_memories, enhanced_prompt_snippet, retrieval_details, raw_below_threshold.
    """
    # RAG retrieval with threshold
    rag_memories = rag.get_relevant_memories(
        query_text=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    # When no hit, fetch raw top results with threshold=0 to show similarity for each record
    raw_below_threshold = []
    if not rag_memories:
        raw_all = rag.retriever.retrieve(
            query_text=query,
            top_k=top_k,
            similarity_threshold=0.0,
        )
        for m in raw_all:
            raw_below_threshold.append({
                "id": m.get("id"),
                "similarity": m.get("similarity", 0.0),
                "user_input": (m.get("metadata") or {}).get("user_input", ""),
                "assistant_response": (m.get("metadata") or {}).get("assistant_response", ""),
            })

    # Retrieval details for report output
    retrieval_details = []
    for m in rag_memories:
        retrieval_details.append({
            "id": m.get("id"),
            "similarity": m.get("similarity", 0.0),
            "distance": m.get("distance", 0.0),
            "user_input": (m.get("metadata") or {}).get("user_input", ""),
            "assistant_response": (m.get("metadata") or {}).get("assistant_response", ""),
            "document": (m.get("document", ""))[:200],
        })

    enhanced_prompt_snippet = build_enhanced_prompt(rag_memories)

    return {
        "rag_memories": rag_memories,
        "enhanced_prompt_snippet": enhanced_prompt_snippet,
        "retrieval_details": retrieval_details,
        "raw_below_threshold": raw_below_threshold,
    }


def print_report(query: str, result: dict, top_k: int, threshold: float, verbose: bool = False):
    """Print comparison and retrieval report for one query."""
    print("\n" + "=" * 60)
    print("Query")
    print("=" * 60)
    print(query)
    print()

    print("=" * 60)
    print("RAG Retrieval (top_k={}, similarity_threshold={})".format(top_k, threshold))
    print("=" * 60)
    details = result["retrieval_details"]
    raw_below = result.get("raw_below_threshold", [])
    if not details:
        print("  No memories passed the threshold.")
        if raw_below:
            print("\n  Top results (below threshold, consider lowering --threshold):")
            for i, d in enumerate(raw_below, 1):
                sim = d.get("similarity", 0.0)
                ui = (d.get("user_input") or "")[:100]
                ar = (d.get("assistant_response") or "")[:100]
                print(f"  [{i}] id={d.get('id')}  similarity={sim:.4f}")
                print(f"      user_input: {ui}...")
                print(f"      assistant_response: {ar}...")
                print()
    else:
        for i, d in enumerate(details, 1):
            sim = d.get("similarity")
            sim_str = f"{sim:.4f}" if sim is not None else "N/A"
            print(f"  [{i}] id={d['id']}  similarity={sim_str}")
            print(f"      user_input: {d['user_input'][:100]}...")
            print(f"      assistant_response: {d['assistant_response'][:100]}...")
            print()
    print()

    print("=" * 60)
    print("Injected snippet (appended to system prompt)")
    print("=" * 60)
    snippet = result["enhanced_prompt_snippet"]
    if snippet:
        print(snippet)
    else:
        print("  (empty — no memory passed the threshold)")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="RAG test: show retrieval details and injected prompt snippet (no Qwen load)"
    )
    parser.add_argument("query", nargs="?", type=str, default=None, help="Single query text")
    parser.add_argument("--query", dest="query_opt", type=str, default=None, help="Alternative way to pass query")
    parser.add_argument("--top-k", type=int, default=5, help="RAG retrieval top_k")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold (default: 0.7, stricter)")
    parser.add_argument("--verbose", action="store_true", help="Additional debug output")
    parser.add_argument("--interactive", action="store_true", help="Interactive multi-query mode")
    args = parser.parse_args()

    memory_path = get_memory_db_path()
    if not os.path.exists(memory_path):
        print(f"[RAG Test] memory_db not found: {memory_path}")
        print("Run the main app first, or add some memories via tools/rag_manage.py add before testing.")
        sys.exit(1)

    try:
        from ai.memory.rag_memory import RAGMemory
    except ImportError as e:
        print(f"[RAG Test] Failed to import RAGMemory: {e}")
        sys.exit(1)

    from utils.setup_loader import get_user_name
    user_name = get_user_name()
    rag = RAGMemory(
        persist_directory=memory_path,
        collection_name="conversation_memories",
        top_k=args.top_k,
        similarity_threshold=args.threshold,
        user_name=user_name,
    )
    count = rag.count()
    print(f"[RAG Test] Loaded RAG store at {os.path.abspath(memory_path)}, count: {count}, user_name: {user_name}")

    if args.interactive:
        print("Interactive mode: enter query and press Enter; empty line to exit.\n")
        while True:
            try:
                q = input("Query> ").strip()
            except EOFError:
                break
            if not q:
                break
            result = run_one_query(rag, q, top_k=args.top_k, similarity_threshold=args.threshold, verbose=args.verbose)
            print_report(q, result, args.top_k, args.threshold, verbose=args.verbose)
        return

    query = args.query or args.query_opt
    if not query:
        parser.print_help()
        print('\nPlease provide a query, e.g.: python tools/rag_test_compare.py "Hello, my name is Xiaoming"')
        sys.exit(0)

    result = run_one_query(rag, query, top_k=args.top_k, similarity_threshold=args.threshold, verbose=args.verbose)
    print_report(query, result, args.top_k, args.threshold, verbose=args.verbose)


if __name__ == "__main__":
    main()
