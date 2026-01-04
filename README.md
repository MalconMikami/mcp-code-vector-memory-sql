<div align="center">
  <img src="logo-256.png" width="160" height="160" alt="mcp-code-vector-memory-sql logo" />
</div>

# mcp-code-vector-memory-sql

Local MCP memory server for OpenCode/VS Code powered by SQLite + sqlite-vec.
It provides session-scoped memory with semantic search, optional FTS5 re-ranking,
and optional local summaries (GGUF).

Python module: `code_memory` · Console script: `code-memory`

Inspired by:
- `mcp-memory-libsql` (libSQL + vector search)
- `@modelcontextprotocol/server-memory` (knowledge-graph style memory)

## Table of contents

- [Why](#why)
- [Key features](#key-features)
- [Quick start](#quick-start)
- [MCP configuration](#mcp-configuration)
- [Configuration](#configuration)
- [Models (embeddings + local summaries)](#models-embeddings--local-summaries)
- [How it works](#how-it-works)
- [MCP tools](#mcp-tools)
- [Data model](#data-model)
- [Comparison](#comparison)
- [Docs](#docs)
- [Development](#development)
- [License](#license)

## Why

When you use an MCP client while coding, you often need memory that is:

- local-first (privacy by default)
- fast to query (semantic search)
- scoped to a session (avoid cross-project bleed)
- easy to run (SQLite, no services)

`mcp-code-vector-memory-sql` focuses on that workflow.

## Key features

- Session-scoped storage (`session_id` is required)
- Vector search with `sqlite-vec` + `fastembed` (CPU)
- Optional FTS5 for exact/fuzzy matching and re-ranking
- Optional entity extraction (regex + tree-sitter) and knowledge graph
- Optional local summarization via GGUF (`llama-cpp-python`)
- Sensitive-content filter + recent dedupe (hash window)

## Quick start

Requirements: Python 3.10+.

Install:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\\Scripts\\activate   # Windows

pip install -e .
```

Optional extras:

```bash
pip install -e ".[graph,summary]"
```

Run:

```bash
python -m code_memory
```

Legacy entrypoint (kept for compatibility):

```bash
python main.py
```

## MCP configuration

Example `opencode.json`:

```json
{
  "mcpServers": {
    "mcp-code-vector-memory-sql": {
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
    "mcp-code-vector-memory-sql": {
      "command": "code-memory",
      "args": [],
      "env": {}
    }
  }
}
```

## Configuration

Full reference (detailed explanations + more examples): `docs/CONFIGURATION.md`.

### Common env vars (most people only change these)

| Variable | What it controls |
| --- | --- |
| `CODE_MEMORY_DB_PATH` | Full path to the SQLite DB file (overrides `CODE_MEMORY_DB_DIR`) |
| `CODE_MEMORY_WORKSPACE` | Root folder for MCP `resource://workspace` and `resource://readme` |
| `CODE_MEMORY_EMBED_MODEL` | Embedding model name (fastembed) |
| `CODE_MEMORY_EMBED_DIM` | Embedding dimension (required for non-default models) |
| `CODE_MEMORY_ENABLE_VEC` | Enable/disable vector search |
| `CODE_MEMORY_ENABLE_FTS` | Enable/disable FTS5 search and re-ranking |
| `CODE_MEMORY_TOP_K` | Default max results returned by `search_memory` |
| `CODE_MEMORY_TOP_P` | Recency filter applied during re-ranking (0..1) |

### Example configs

Project-local DB + logs:

```json
{
  "CODE_MEMORY_DB_DIR": "C:/repo",
  "CODE_MEMORY_LOG_DIR": "C:/repo/.logs"
}
```

Disable the graph (default) and tune search a bit:

```json
{
  "CODE_MEMORY_ENABLE_GRAPH": "0",
  "CODE_MEMORY_TOP_K": "12",
  "CODE_MEMORY_TOP_P": "0.7",
  "CODE_MEMORY_OVERSAMPLE_K": "4"
}
```

## Models (embeddings + local summaries)

Detailed guide: `docs/MODELS.md`.

### Embedding models (fastembed)

The default is `BAAI/bge-small-en-v1.5` (fast, small, strong baseline).

Other popular options you can use with this server:
- `snowflake/snowflake-arctic-embed-s` (quality-focused, still small)
- `nomic-ai/nomic-embed-text-v1.5` (larger, better for long inputs)

Note: some models (e.g. `Qwen2.5-Embedding-0.6B`) are not supported by
`fastembed` in this repo today. They require a different embedding backend.

### Local summary models (GGUF)

Any GGUF model supported by `llama-cpp-python` can be used for local summaries,
including small code-focused models like `Qwen2.5-Coder-0.5B` (very fast).

## How it works

### `remember`

1. Resolve `session_id` (input, MCP context, or `CODE_MEMORY_SESSION_ID`)
2. Filter sensitive content + skip recent duplicates
3. Generate optional local summary and basic tags
4. Store in SQLite (+ vector index and FTS index if enabled)
5. Optionally extract entities and update the knowledge graph

### `search_memory`

1. Create an embedding for the query (if vector search is enabled)
2. Retrieve candidates (oversample) and merge optional FTS hits
3. Re-rank with recency/priority and optional FTS bonus
4. Apply `top_p` (recency filter) and return the top results

## MCP tools

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

## Data model

- `memories`: main table (content, summary, tags, session_id, priority, metadata)
- `vec_memories`: sqlite-vec table for embeddings
- `memories_fts`: FTS5 index synchronized via triggers (optional)
- `entities`: extracted entities per memory
- `graph_*`: graph entities/observations/relations (optional)

## Comparison

| Capability | mcp-code-vector-memory-sql | mcp-memory-libsql | @modelcontextprotocol/server-memory |
| --- | --- | --- | --- |
| Storage | SQLite | libSQL (SQLite compatible) | JSONL |
| Remote DB | No | Yes (libSQL/Turso) | No |
| Vector search | Yes (sqlite-vec) | Yes (libSQL vector) | No |
| FTS re-rank | Yes (FTS5) | Not documented | Not documented |
| Session scoping | Yes (`session_id`) | Not documented | Not documented |
| Knowledge graph | Optional | Yes | Yes |
| Local summaries | Optional (GGUF) | Not documented | Not documented |

Note: comparison is based on the published READMEs of those projects.

## Docs

- `docs/README.md` (index)
- `docs/CONFIGURATION.md` (full configuration reference)
- `docs/MODELS.md` (choosing models)
- `docs/ARCHITECTURE.md` (design + schema)
- `docs/OPERATIONS.md` (maintenance + troubleshooting)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

TBD
