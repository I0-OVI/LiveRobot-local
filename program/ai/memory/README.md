# RAG Memory System

RAG (Retrieval-Augmented Generation) memory system for Rubus AI Agent.

## Overview

The RAG memory system enables Rubus to:
- Persistently store conversation history
- Retrieve relevant memories based on current conversation
- Enhance prompts with retrieved memories for better context awareness

## Architecture

The system consists of 5 core modules:

1. **MemoryVectorStore** - Vector database using ChromaDB
2. **MemoryEmbedder** - Text embedding using sentence-transformers
3. **MemoryRetriever** - Memory retrieval with similarity search
4. **MemoryManager** - Complete memory lifecycle management
5. **RAGIntegration** - Integration with text generation

## Installation

Install required dependencies:

```bash
pip install chromadb sentence-transformers
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

The RAG system is automatically enabled by default when creating an AIAgent:

```python
from main import AIAgent

agent = AIAgent(
    use_rag=True,  # Enable RAG (default: True)
    rag_top_k=3,  # Number of memories to retrieve (default: 3)
    rag_similarity_threshold=0.7,  # Minimum similarity (default: 0.7)
    memory_persist_path="./memory_db"  # Memory storage path (optional)
)

agent.start()
```

### Disable RAG

To disable RAG:

```python
agent = AIAgent(use_rag=False)
```

### Configuration Options

- `use_rag`: Enable/disable RAG system (default: True)
- `rag_top_k`: Number of top memories to retrieve (default: 3)
- `rag_similarity_threshold`: Minimum similarity score 0.0-1.0 (default: 0.7)
- `memory_persist_path`: Path to persist memory database (default: "./memory_db")

## How to View Stored Conversations

### Storage Locations

- **Replay Memory** (sliding window): `{Reorganize}/utils/replay_db/replay.json` — Single JSON, max N turns (default 50)
- **RAG Memory** (long-term): `{Reorganize}/utils/memory_db/` — ChromaDB vector store

### Using the Viewer Script

```bash
# From Reorganize directory
cd main/main_live2d/Reorganize
python -m ai.memory.view_memories
```

Or from project root:

```bash
python main/main_live2d/Reorganize/main.py  # Run agent first to create data
cd main/main_live2d/Reorganize
python -m ai.memory.view_memories
```

### Manual Viewing

1. **Replay (JSON)**: Open `utils/replay_db/replay.json` in a text editor
2. **RAG (ChromaDB)**: Use the viewer script above, or connect via ChromaDB API

## How It Works

1. **Memory Storage**: Each conversation is automatically saved to the vector database
2. **Memory Retrieval**: When generating a response, relevant memories are retrieved based on similarity
3. **Prompt Enhancement**: Retrieved memories are formatted and added to the system prompt
4. **Context Awareness**: The LLM uses enhanced prompts to generate more contextually aware responses

## Memory Format

Memories are stored with the following structure:

```python
{
    "id": "unique-memory-id",
    "user_input": "User's question",
    "assistant_response": "Assistant's reply",
    "timestamp": "2024-01-01T12:00:00",
    "importance": 1.0,
    "tags": [],
    "embedding": [0.1, 0.2, ...]  # Vector embedding
}
```

## Memory Management

### Get Memory Statistics

```python
if agent.memory_manager:
    stats = agent.memory_manager.get_memory_stats()
    print(f"Total memories: {stats['total_memories']}")
```

### Cleanup Old Memories

```python
if agent.memory_manager:
    # Keep only memories from last 30 days
    deleted = agent.memory_manager.cleanup_old_memories(days=30)
    print(f"Deleted {deleted} old memories")
```

### Clear All Memories

```python
if agent.memory_manager:
    agent.memory_manager.clear_all_memories()
```

## Performance Considerations

- **Embedding Model**: Uses `paraphrase-multilingual-MiniLM-L12-v2` by default (supports Chinese and English)
- **Vector Database**: ChromaDB automatically persists data
- **Memory Usage**: Embeddings are stored in memory for fast retrieval
- **Initialization**: First run may be slower due to model download

## Troubleshooting

### Import Errors

If you see import errors, make sure dependencies are installed:

```bash
pip install chromadb sentence-transformers
```

### Memory Not Persisting

ChromaDB automatically persists data. If memories are lost:
- Check that the `memory_persist_path` directory exists and is writable
- Ensure `agent.stop()` is called to persist memory before exit

### Low Similarity Scores

If retrieved memories have low similarity:
- Lower the `rag_similarity_threshold` (e.g., 0.5)
- Increase `rag_top_k` to retrieve more memories
- Check that conversations are being saved correctly

## Technical Details

### Embedding Model

- Default: `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Alternative: `BAAI/bge-small-zh-v1.5` (512 dimensions, Chinese-optimized)

### Vector Database

- Uses ChromaDB with persistent storage
- Automatic indexing and similarity search
- Supports metadata filtering

### Retrieval Strategy

- Top-K retrieval with similarity threshold
- Optional time-based weighting (disabled by default)
- Cosine similarity for vector comparison

### Query Canonicalization

Pronouns like "你"/"我" (Chinese) and "you"/"I" (English) refer to the same entity. A **Query Canonicalization** layer normalizes within each language: Chinese → "用户", English → "user", so stored content keeps its language and "你叫什么" / "我叫什么" / "what is your name" match semantically. Applied at both retrieval (query + document) and storage (new memories).
