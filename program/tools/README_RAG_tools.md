# RAG Management and Testing Tools

Run the following commands inside the **Reorganize** directory (shared `memory_db` with the main app).

## 1. Manual RAG Management (Add / Delete / Merge)

RAG stores **query–answer pairs**: the first argument is the query (used for retrieval), the second is the answer (the content to recall).

- **Add one RAG memory** (query + answer; bypasses importance filtering):
  ```bash
  python program/tools/rag_manage.py add "<query>" "<answer>"
  python program/tools/rag_manage.py add "<query>" "<answer>" --importance 0.8
  ```
  Example: `python program/tools/rag_manage.py add "What library for ML?" "I usually use PyTorch."`
- **Delete by ID**:
  ```bash
  python program/tools/rag_manage.py delete <memory_id>
  ```
- **List RAG memories** (paginated):
  ```bash
  python program/tools/rag_manage.py list
  python program/tools/rag_manage.py list --limit 20 --offset 0
  ```
- **Clear all RAG memories** (use with caution):
  ```bash
  python program/tools/rag_manage.py clear
  ```
- **Manually merge similar memories** (slower, recommended infrequently):
  ```bash
  python program/tools/rag_manage.py merge
  ```

## 2. RAG Test / Comparison (Independent of Qwen)

This tool does not load Qwen. It only performs retrieval and context comparison to show:
- the difference between **with RAG** and **without RAG**
- detailed RAG retrieval behavior

- **Single query**:
  ```bash
  python program/tools/rag_test_compare.py "your query"
  python program/tools/rag_test_compare.py --query "your query" --top-k 5 --threshold 0.7
  ```
- **Interactive multi-query mode**:
  ```bash
  python program/tools/rag_test_compare.py --interactive
  ```

Output includes:
- **RAG retrieval details**: id, similarity, user_input, assistant_response for each hit
- **Comparison view**: context passed without RAG vs memory snippet injected into the system prompt with RAG

## 3. Main Program Auto-Merge Mode

The main program supports periodic auto-merge: merge similar memories every N successful saves.
- Parameter: `rag_auto_merge_every` (default `50`; set to `0` to disable)
- This avoids high latency from merging on every write while keeping periodic deduplication.
