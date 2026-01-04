# code-memory

Local MCP memory server for OpenCode (VS Code) using SQLite + sqlite-vec. It provides
session-scoped memory with semantic search, optional FTS re-rank, and optional
local summaries.

## Features

- Local SQLite storage with WAL + tuned pragmas
- Vector search via sqlite-vec (fastembed on CPU)
- Hybrid retrieval (vector + FTS) with simple rerank (recency + priority)
- Session-scoped memory (session_id required)
- Optional entity extraction (tree-sitter) and knowledge graph
- Optional local summaries via GGUF (llama-cpp-python)
- Privacy filter + dedupe

## Install

From source:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[graph,summary]"
```

Dev extras:

```bash
pip install -e ".[dev]"
```

## Run

Development entrypoint (kept for compatibility):

```bash
python main.py
```

Package entrypoint:

```bash
python -m code_memory
```

## MCP Configuration (OpenCode)

Example `opencode.json`:

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "python",
      "args": ["-m", "code_memory"],
      "env": {
        "CODE_MEMORY_DB_DIR": "C:/path/to/your/workspace",
        "CODE_MEMORY_LOG_DIR": "C:/Users/you/.cache/code-memory/logs"
      }
    }
  }
}
```

If you install the package globally, you can use the console script:

```json
{
  "mcpServers": {
    "code-memory": {
      "command": "code-memory",
      "args": [],
      "env": {}
    }
  }
}
```

## Environment Variables

- `CODE_MEMORY_DB_PATH`: full path to the SQLite DB file
- `CODE_MEMORY_DB_DIR`: directory for the DB (default: current working dir)
- `CODE_MEMORY_WORKSPACE`: root for MCP resources (default: current working dir)
- `CODE_MEMORY_EMBED_MODEL`: embedding model name (default: BAAI/bge-small-en-v1.5)
- `CODE_MEMORY_EMBED_DIM`: embedding dimension (required for custom models)
- `CODE_MEMORY_MODEL_DIR`: cache directory for embeddings
- `CODE_MEMORY_LOG_DIR` / `CODE_MEMORY_LOG_FILE`: log output path
- `CODE_MEMORY_ENABLE_VEC` / `CODE_MEMORY_ENABLE_FTS` / `CODE_MEMORY_ENABLE_GRAPH`
- `CODE_MEMORY_TOP_K`, `CODE_MEMORY_TOP_P`, `CODE_MEMORY_OVERSAMPLE_K`
- `CODE_MEMORY_PRIORITY_WEIGHT`, `CODE_MEMORY_RECENCY_WEIGHT`, `CODE_MEMORY_FTS_BONUS`

Local summary (GGUF):

- `CODE_MEMORY_SUMMARY_MODEL`: full path to GGUF file (required to enable)
- `CODE_MEMORY_SUMMARY_CTX`, `CODE_MEMORY_SUMMARY_THREADS`
- `CODE_MEMORY_SUMMARY_MAX_TOKENS`, `CODE_MEMORY_SUMMARY_TEMPERATURE`, `CODE_MEMORY_SUMMARY_TOP_P`
- `CODE_MEMORY_SUMMARY_REPEAT_PENALTY`, `CODE_MEMORY_SUMMARY_GPU_LAYERS`
- `CODE_MEMORY_AUTO_INSTALL` (default 1) and `CODE_MEMORY_PIP_ARGS` for llama-cpp-python

## Tools

- `remember(content, session_id, kind, summary, tags, priority, metadata_json)`
- `search_memory(query, session_id, limit, top_p)`
- `list_recent(limit)`
- `list_entities(memory_id)`
- `upsert_entity(name, entity_type, observations_json, memory_id)`
- `add_relation(source, target, relation_type, memory_id)`
- `get_entity(name)`
- `get_context_graph(query, limit)`
- `maintenance(action, confirm, session_id, older_than_days)`
- `diagnostics()`
- `health()`

## Data Model

- `memories`: master table with content, summary, tags, session_id, priority
- `vec_memories`: sqlite-vec table with embeddings
- `memories_fts`: FTS5 index synced via triggers
- `entities`: extracted entities (NER/AST)
- Optional graph tables (`graph_entities`, `graph_observations`, `graph_relations`)

## Development

Smoke test:

```bash
python tests/test_memory.py
```

Pytest:

```bash
pytest
```

## License

TBD
