# MCP tools API

This page documents the MCP tools exposed by `mcp-code-vector-memory-sql`.

Notes:

- `session_id` is required for memory operations and scopes storage/retrieval.
- Most tools return JSON objects (or arrays of objects).
- Some features depend on flags:
  - `CODE_MEMORY_ENABLE_VEC` (vector search)
  - `CODE_MEMORY_ENABLE_FTS` (FTS merge/re-rank)
  - `CODE_MEMORY_ENABLE_GRAPH` (graph tables and graph tools)

## `remember`

Store a memory row (and optionally index it for vectors/FTS, extract entities, and update the graph).

Inputs:

- `content` (string, required): the raw text to store
- `session_id` (string, required): session scope key
- `kind` (string, optional): category (example: `decision`, `bugfix`, `note`)
- `summary` (string, optional): short summary (if omitted, server may generate one if GGUF summaries are enabled)
- `tags` (string, optional): comma-separated tags (example: `auth,jwt,bugfix`)
- `priority` (int, optional): 1..5 (1 is most important, 5 is least important)
- `metadata_json` (string, optional): JSON object encoded as a string

Returns (on success):

- `id` (int): memory id
- echoes `summary`, `session_id`, `kind`, `tags`, `priority`

Returns (on skip):

- `{ "status": "skipped" }` (for sensitive content or recent duplicates)

Example:

```json
{
  "content": "Decision: keep embeddings CPU-only. Use sqlite-vec for local semantic search.",
  "session_id": "opencode:2026-01-04",
  "kind": "decision",
  "summary": "Chose CPU-only embeddings + sqlite-vec for local semantic memory.",
  "tags": "architecture,sqlite,embeddings",
  "priority": 1,
  "metadata_json": "{\"repo\":\"mcp-code-vector-memory-sql\",\"ticket\":\"MEM-12\"}"
}
```

## `search_memory`

Search memories for a `session_id` using vector similarity (if enabled) plus optional FTS merge and re-ranking.

Inputs:

- `query` (string, required)
- `session_id` (string, required)
- `limit` (int, optional): max results to return (default: `CODE_MEMORY_TOP_K`)
- `top_p` (float, optional): recency filter (0..1], default: `CODE_MEMORY_TOP_P`

Returns:

- array of memory objects; each item typically contains:
  - `id`, `session_id`, `kind`, `content`, `summary`, `tags`, `priority`, `metadata`, `created_at`
  - `score` (float): lower is better (ranking score)
  - `fts_hit` (bool): whether the item was also matched by FTS

Example:

```json
{
  "query": "sqlite-vec embedding dimension mismatch",
  "session_id": "opencode:2026-01-04",
  "limit": 8,
  "top_p": 0.7
}
```

## `list_recent`

List the most recent stored memories (not session-scoped by default).

Inputs:

- `limit` (int, optional): default 20

Returns:

- array of memory objects (similar shape to `search_memory` items, without ranking fields)

## `list_entities`

List extracted entities for a specific memory id.

Inputs:

- `memory_id` (int, required)

Returns:

- array of `{ "entity_type": string, "name": string, "source": string, "path": string|null }`

Notes:

- Entity extraction may be minimal when `tree-sitter` is not installed; regex extraction still runs.

## Graph tools (require `CODE_MEMORY_ENABLE_GRAPH=1`)

When graph mode is enabled, `remember` updates graph tables and the following tools can be used.

### `upsert_entity`

Create or update a graph entity and attach observations.

Inputs:

- `name` (string, required)
- `entity_type` (string, optional, default `entity`)
- `observations_json` (string, optional): JSON array of strings
- `memory_id` (int, optional): link observations to a memory id

Returns:

- `{ "status": "ok", "entity_id": int }`

### `add_relation`

Add a relation between two entities.

Inputs:

- `source` (string, required)
- `target` (string, required)
- `relation_type` (string, required): active-voice relation name (example: `depends_on`, `implements`)
- `memory_id` (int, optional)

Returns:

- `{ "status": "ok", "relation_id": int }`

### `get_entity`

Fetch a single graph entity, including observations and relations (shape depends on stored data).

Inputs:

- `name` (string, required)

Returns:

- object with entity details, observations, and relations

### `get_context_graph`

Return a context graph snapshot or search the graph.

Inputs:

- `query` (string, optional): when set, performs semantic search over graph entities (requires vectors)
- `limit` (int, optional): default 10

Returns:

- object containing entities and relations (shape depends on implementation and flags)

## `maintenance`

Manual maintenance operations (never runs automatically).

Inputs:

- `action` (string, required): one of
  - `vacuum`
  - `purge_all` (destructive)
  - `purge_session` (destructive, requires `session_id`)
  - `prune_older_than` (destructive, requires `older_than_days`)
- `confirm` (bool, optional): required for destructive actions (everything except `vacuum`)
- `session_id` (string, optional): used by `purge_session`
- `older_than_days` (int, optional): used by `prune_older_than`

Returns:

- `{ "status": "ok", ... }` with action-specific fields (for example `deleted`)

## `diagnostics`

Return environment and DB diagnostics (flags, defaults, table list, counts).

Inputs: none

Returns:

- object with:
  - `db_path`, embedding model info, feature flags
  - defaults (top_k/top_p/weights)
  - summary config (enabled/model/ctx/threads/etc)
  - tables and row counts

## `health`

Lightweight health check.

Inputs: none

Returns:

- `{ "status": "ok", ... }` including:
  - `db_path`, embedding dimension/model
  - feature flags
  - summary enablement
  - log file path (if configured)
