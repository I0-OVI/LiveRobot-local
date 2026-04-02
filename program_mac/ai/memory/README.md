# RAG & Replay Memory

Long-term **RAG** (Chroma + embeddings) and short-term **Replay** (sliding window JSON) for the desktop pet. The running app uses **`MemoryCoordinator`** as the single main integration path.

## Architecture (main flow)

| Piece | Role |
|--------|------|
| **ReplayMemory** | Recent turns, token budget, `replay_db/replay.json` |
| **RAGMemory** | Chroma persist dir, importance filter, merge, optional cleanup |
| **MemoryRetriever** | Canonicalize query/doc → embed → cosine similarity + optional time weight |
| **RAGTrigger** | Keyword layer 1; optional LLM layer 2 (`use_llm_trigger`); optional timeout |
| **MemoryCoordinator** | Wires replay + RAG + summarizer + importance calculator + trigger + optional save evaluator |
| **RAGSaveEvaluator** | Optional **single** LLM JSON: `store_long_term` + `importance` (+ tags); used on async save path |

### Qwen inference queue (`QwenTextGenerator`)

All **`chat()`** and **`chat_stream()`** calls are handled by **one daemon worker** pulling jobs from an internal `queue.Queue`. Only that worker thread runs `model.chat` / `model.generate`, so concurrent callers (main reply stream, RAG Layer2 trigger LLM, async RAG save LLM, summarizer, etc.) are **serialized FIFO** instead of racing on the GPU. The thread that calls `chat_stream` only **consumes** token chunks from a per-job output queue.

**Effect on the next user turn:** if a background RAG save still needs a Qwen `chat()` (combined save eval or legacy importance), that job waits in the **same FIFO** as the next reply. So the next reply may start slightly later; behavior is predictable, not undefined concurrency.

### RAG async save path

1. **Replay** is updated **synchronously** in `save_conversation` (always).
2. **Long-term RAG** runs **asynchronously** (default): either a **single `rag_save_worker`** thread pulling jobs from a queue (`use_save_worker=true`), or **one daemon thread per turn** (`use_save_worker=false`).
3. **`use_llm_long_term_eval`**: one LLM call returns JSON (`store_long_term`, `importance`, optional `tags`/`reason`). If `store_long_term` is false, the vector write is skipped. Runs **after** the user-visible reply (no extra delay before first token).
4. **Legacy path** (long-term eval off): optional `ImportanceCalculator` LLM only, as before.
5. **`save_llm_timeout_sec`**: applies to save-side LLM (`chat` with timeout). On **timeout** of the combined eval, the turn is **not** written to long-term RAG. On **parse error**, a conservative default importance (`0.7`) is used and the memory is stored. For combined eval, **`save_llm_timeout_sec <= 0`** disables the combined JSON call and uses **legacy importance-only** LLM (if `use_llm_evaluation` is still true).
6. **`persist()`** / agent stop: the coordinator **`join()`s** the save job queue first (when using the save worker) so pending RAG writes finish before flushing Replay/Chroma to disk.

### RAG Layer2 LLM timeout

- **`llm_trigger_timeout_sec`** in `setup.txt` → `RAG_OPTIONS`, or **`rag_llm_trigger_timeout_sec`** on `AIAgent`, caps how long Layer2 may block waiting for the trigger LLM (`None` = no limit).
- **`<= 0`**: disables Layer2 entirely (keywords only), even if `use_llm_trigger=true`.
- If **`FuturesTimeout`** fires, that turn skips RAG retrieval; the underlying job may still run later when it reaches the front of the inference queue.

**Legacy (scripts / advanced):** `MemoryManager` and `RAGIntegration` still import from `ai.memory`. Prefer **`MemoryCoordinator`** in new code. `RAGIntegration(memory_coordinator)` works for “always retrieve and append to system prompt” without the trigger gate.

## Dependencies

```bash
pip install chromadb sentence-transformers
```

## Configuration

### `AIAgent` (program_mac/main.py on macOS; program/main.py on Windows)

- `use_rag` — disable all RAG+Replay coordinator features when `False`
- `replay_*` — token budget, persist, max turns
- `rag_*` — top_k, similarity threshold, summary interval, importance, merge, auto-merge interval, etc.
- `rag_use_llm_trigger`, `rag_llm_trigger_timeout_sec`, `rag_always_retrieve`, `rag_use_time_weight`, `rag_time_decay_days` — if `None`, values come from **`setup.txt` → `RAG_OPTIONS`** (where present); if set, they override the file.
- `rag_use_llm_long_term_eval`, `rag_save_llm_timeout_sec`, `rag_use_save_worker` — same override rules.

### `setup.txt` → `RAG_OPTIONS`

- `use_llm_trigger` — LLM decides whether to retrieve when keywords do not match (extra latency).
- `llm_trigger_timeout_sec` — float seconds; max wait for that LLM call. Omit for no limit; `0` disables Layer2 LLM.
- `always_retrieve` — if the vector store is non-empty, always run retrieval; injection still requires similarity ≥ `rag_similarity_threshold`.
- `use_time_weight` / `time_decay_days` — recency bias in `MemoryRetriever`.
- `use_llm_long_term_eval` — single LLM JSON for long-term store decision + importance on async save (default `false`).
- `save_llm_timeout_sec` — float; max wait for save-side LLM. Omit = no limit. With long-term eval on, `0` switches to legacy importance-only LLM.
- `use_save_worker` — `true` (default): one background worker + queue for RAG saves; `false`: thread per save.

## Storage paths (this repo)

- **Replay:** `program_mac/utils/replay_db/replay.json` (path from `get_current_dir()` in `utils/path_config.py`; use `program/` on Windows).
- **RAG:** `program_mac/utils/memory_db/` (default `get_memory_db_path()`).

## Usage examples

```python
from main import AIAgent

agent = AIAgent(
    use_rag=True,
    rag_top_k=3,
    rag_similarity_threshold=0.7,
    memory_persist_path=None,  # None → utils/memory_db
)
agent.start()
```

### Statistics and maintenance

```python
if agent.memory_coordinator:
    stats = agent.memory_coordinator.get_stats()
    print(stats["rag"])

    # Same semantics as legacy MemoryManager.cleanup_old_memories
    deleted = agent.memory_coordinator.cleanup_old_memories(days=30)
    agent.memory_coordinator.clear_all_rag_memories()  # wipe long-term store
```

`ImportanceCalculator` and periodic **summaries** need the text generator: it is set on the coordinator at **`AIAgent` init** and again after `load_model()` in `start()`.

## How generation uses memory

1. **Replay** history is always loaded (within token budget).
2. **RAG** runs only when the trigger says so **or** `always_retrieve` is on (and store non-empty), unless you use `force_use_rag=True` in `get_context_for_generation`.
3. Retrieved chunks are appended as an **enhanced prompt** snippet (Chinese instruction + bullet memories).

## Retrieval details

- Default embedder: `paraphrase-multilingual-MiniLM-L12-v2` (see `MemoryEmbedder`).
- Chroma distance → similarity heuristic, then **cosine re-score** on canonicalized query/document text.
- **Query canonicalization** (`query_canonicalizer.py`): aligns 你/我/用户名等，改善中英文代词匹配。

## CLI tools

See **`tools/README_RAG_tools.md`** (`rag_manage.py`, `rag_test_compare.py`). Run commands with **`program_mac`** (or **`program`** on Windows) as the working directory so imports resolve.

## View memories

From `program_mac` (with `PYTHONPATH` including `program_mac` if needed):

```bash
cd program_mac
python -m ai.memory.view_memories
```

## Troubleshooting

- **Import errors:** install `chromadb` and `sentence-transformers`.
- **Low recall:** lower `rag_similarity_threshold` or enable `always_retrieve` (accept extra embedding cost per turn).
- **Missed “do you remember”:** enable more keywords (see `rag_trigger.py`) or `use_llm_trigger=true` in `RAG_OPTIONS` (latency tradeoff).
- **Next reply feels slow after fast multi-turn chat:** save-side LLM + main reply share one FIFO; enable `use_llm_long_term_eval=false` or tighten `save_llm_timeout_sec`, or keep `use_save_worker=true` to avoid piling up many save threads.
