<div align="center">
  <img src="logo-256.png" width="256" height="256" alt="mcp-code-vector-memory-sql logo" />
</div>

# mcp-code-vector-memory-sql

Local MCP memory server for OpenCode/VS Code, Cline, Roo Code, Claude Desktop, OpenHands, Aider, Cursor, Gider, Windsurf, PydanticAI, LangGraph, CrewA powered by SQLite + FTS5 + libSQL vector (optional).
It provides session-aware memory with semantic search, optional FTS5 re-ranking,
and optional local summaries (GGUF).


Inspired by:
- `mcp-memory-libsql` (libSQL + vector search)
- `@modelcontextprotocol/server-memory` (knowledge-graph style memory)

## Table of contents

- [Why](#why)
- [How it works](#how-it-works)
- [Hybrid search (vectors + FTS + graph)](#hybrid-search-vectors--fts--graph)
- [Key features](#key-features)
- [Install](#install)
- [MCP setup](#mcp-setup)
- [Configuration](#configuration)
- [Models (embeddings + local summaries)](#models-embeddings--local-summaries)
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

## How it works

## Hybrid search (vectors + FTS + graph)

`mcp-code-vector-memory-sql` combines multiple retrieval signals to get better "developer
memory" results:

- **Vector search**: semantic similarity (libSQL vector backend)
- **FTS5**: exact/fuzzy term matches, merged into results
- **Graph**: entity-centric lookup via `get_context_graph`

If you want to understand the full retrieval pipeline and ranking details, see
[docs/HYBRID_SEARCH.md](docs/HYBRID_SEARCH.md) and [docs/TUNING_GUIDE.md](docs/TUNING_GUIDE.md).

### `remember`

1. Resolve `session_id` (input, MCP context, or `CODE_MEMORY_SESSION_ID`)
2. Filter sensitive content + skip recent duplicates
3. Generate basic tags
4. Store in libSQL (vectors + FTS5)
5. Extract entities/relations and update the knowledge graph

### `search_memory`

1. Create an embedding for the query (if vector search is enabled)
2. Retrieve candidates (oversample) and merge optional FTS hits
3. Re-rank with recency/priority and optional FTS bonus
4. Apply `top_p` (recency filter) and return the top results

## Key features

- Session-aware storage (`session_id` is required for remember; boosts same-session search results)
- Vector search with `fastembed` (CPU) + libSQL vector (optional, when using `CODE_MEMORY_DB_URL`)
- FTS5 for exact/fuzzy matching and re-ranking
- Entity extraction (LLM + regex fallback) and knowledge graph
- Optional local NER via GGUF (`llama-cpp-python`)
- Sensitive-content filter + recent dedupe (hash window)

## Install

Requirements: Python 3.10+.

From source:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\\Scripts\\activate   # Windows

pip install -e .
```

Optional extras:

```bash
pip install -e ".[ner]"
```

Run manually (useful for debugging):

```bash
python -m code_memory
```

Legacy entrypoint (kept for compatibility):

```bash
python main.py
```

## MCP setup

Below are configuration examples for popular MCP clients.

### OpenCode / VS Code

Example `opencode.json`:

```json
{
  "mcpServers": {
    "mcp-code-vector-memory-sql": {
      "command": "python",
      "args": ["-m", "code_memory"],
      "env": {
        "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
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

### Cline configuration

Add this to your Cline MCP settings:

```json
{
  "mcpServers": {
    "mcp-code-vector-memory-sql": {
      "command": "python",
      "args": ["-m", "code_memory"],
      "env": {
        "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
        "CODE_MEMORY_LOG_DIR": "/path/to/logs"
      }
    }
  }
}
```

### Claude Desktop with WSL configuration

If you run Claude Desktop on Windows and want the server inside WSL, configure
Claude Desktop to call `wsl.exe` and start the Python module there.

Example:

```json
{
  "mcpServers": {
    "mcp-code-vector-memory-sql": {
      "command": "wsl.exe",
      "args": [
        "bash",
        "-lc",
        "cd /path/to/your/repo && source .venv/bin/activate && python -m code_memory"
      ],
      "env": {
        "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
        "CODE_MEMORY_LOG_DIR": "/path/to/logs"
      }
    }
  }
}
```

Notes:

- WSL uses Linux paths (for example `/home/you/...`), not `C:\\...`.
- If you do not use a venv in WSL, remove `source .venv/bin/activate` and ensure
  `python` can import `code_memory`.

## Configuration

Full reference (detailed explanations + more examples): [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Common env vars (most people only change these)

| Variable | What it controls |
| --- | --- |
| `CODE_MEMORY_WORKSPACE` | Root folder for MCP `resource://workspace` and `resource://readme` |
| `CODE_MEMORY_DB_URL` | Required libSQL URL (`libsql://...`) |
| `CODE_MEMORY_EMBED_MODEL` | Embedding model name (fastembed) |
| `CODE_MEMORY_EMBED_DIM` | Embedding dimension (required for non-default models) |
| Vector search | Always enabled |
| FTS5 | Always enabled |
| `CODE_MEMORY_TOP_K` | Default max results returned by `search_memory` |
| `CODE_MEMORY_TOP_P` | Recency filter applied during re-ranking (0..1) |

### Example configs

Project-local DB + logs:

```json
{
  "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
  "CODE_MEMORY_LOG_DIR": "C:/repo/.logs"
}
```

Tune search a bit:

## Models (embeddings + local NER)

Detailed guide: [docs/MODELS.md](docs/MODELS.md).

### Embedding models (fastembed)

The default is `BAAI/bge-small-en-v1.5` (fast, small, strong baseline).

Other popular options you can use with this server:
- `snowflake/snowflake-arctic-embed-s` (quality-focused, still small)
- `nomic-ai/nomic-embed-text-v1.5` (larger, better for long inputs)

Note: some models (e.g. `Qwen2.5-Embedding-0.6B`) are not supported by
`fastembed` in this repo today. They require a different embedding backend.

### Local NER models (GGUF)

Any GGUF model supported by `llama-cpp-python` can be used for local NER/entity extraction,
including small code-focused models like `Qwen2.5-Coder-0.5B` (very fast).

## MCP tools

Full reference (inputs/outputs/examples): [docs/API.md](docs/API.md).

- `remember(content, session_id, kind, tags, priority, metadata_json)`
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

- `entities`: both memory nodes (`entity_type='memory'`) and extracted world-fact entities
- `observations`: stored text (content + tags/metadata), linked to an entity
- `relations`: directed edges between entities, optionally linked to an observation
- `observations.embedding`: libSQL vector embedding column (best-effort; requires libSQL vector functions)
- `observations_fts`: FTS5 index synchronized via triggers (creates internal FTS5 tables)

## Comparison

| Capability | mcp-code-vector-memory-sql | mcp-memory-libsql | @modelcontextprotocol/server-memory |
| --- | --- | --- | --- |
| Storage | SQLite | libSQL (SQLite compatible) | JSONL |
| Remote DB | No | Yes (libSQL/Turso) | No |
| Vector search | Yes (libSQL vector) | Yes (libSQL vector) | No |
| FTS re-rank | Yes (FTS5) | Not documented | Not documented |
| Session scoping | Yes (`session_id`) | Not documented | Not documented |
| Knowledge graph | Yes | Yes | Yes |
| Local NER | Optional (GGUF) | Not documented | Not documented |

Note: comparison is based on the published READMEs of those projects.

## Docs
 
- [Docs index](docs/README.md)
- [MCP tools API](docs/API.md)
- [Configuration](docs/CONFIGURATION.md)
- [libSQL local backend](docs/LIBSQL_LOCAL.md)
- [Models](docs/MODELS.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Hybrid search](docs/HYBRID_SEARCH.md)
- [Tuning guide](docs/TUNING_GUIDE.md)
- [Operations](docs/OPERATIONS.md)

## Performance snapshot

Latest local benchmark artifacts are stored under `docs/benchmarks/`.

| Benchmark | Mean | Ops/sec |
|---|---:|---:|
| `insert` | 0.972 ms | 1029 |
| `embed` | 5.363 ms | 186 |
| `vector_search` | 7.763 ms | 129 |
| `hybrid_search` | 10.840 ms | 92 |
| `tags_fts` | 0.131 ms | 7605 |
| `tags_like` | 0.017 ms | 57367 |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

TBD
