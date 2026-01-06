# MCP tools API

This page documents the MCP tools exposed by `mcp-code-vector-memory-sql`.

Notes:

- `session_id` is required for `remember`. For `search_memory`, providing `session_id` enables session-aware ranking
  (same-session results get a boost), but search still considers all sessions.
- Most tools return JSON objects (or arrays of objects).
- Vector / FTS / graph are always enabled.

## `remember`

Store a memory observation (and optionally index it for vectors/FTS, extract entities, and update relations).

Inputs:

- `content` (string, required): the raw text to store
- `session_id` (string, required): session scope key
- `kind` (string, optional): category (example: `decision`, `bugfix`, `note`)
- `tags` (string, optional): comma-separated tags (example: `auth,jwt,bugfix`)
- `priority` (int, optional): 1..5 (1 is most important, 5 is least important)
- `metadata_json` (string, optional): JSON object encoded as a string

Returns (on success):

- `id` (int): observation id
- echoes `session_id`, `kind`, `tags`, `priority`, `content_hash`

Returns (on skip):

- `{ "status": "skipped" }` (for sensitive content or recent duplicates)

Example:

```json
{
  "content": "Decision: keep embeddings CPU-only. Use vector search for local semantic recall.",
  "session_id": "opencode:2026-01-04",
  "kind": "decision",
  "tags": "architecture,sqlite,embeddings",
  "priority": 1,
  "metadata_json": "{\"repo\":\"mcp-code-vector-memory-sql\",\"ticket\":\"MEM-12\"}"
}
```

## `search_memory`

Search memories using vector similarity plus FTS merge and re-ranking.

Inputs:

- `query` (string, required)
- `session_id` (string, optional): used for session-aware ranking
- `limit` (int, optional): max results to return (default: `CODE_MEMORY_TOP_K`)
- `top_p` (float, optional): recency filter (0..1], default: `CODE_MEMORY_TOP_P`

Returns:

- array of memory objects; each item typically contains:
  - `id`, `session_id`, `kind`, `content`, `content_hash`, `tags`, `priority`, `metadata`, `created_at`
  - `score` (float): lower is better (ranking score)
  - `fts_hit` (bool): whether the item was also matched by FTS

Example:

```json
{
  "query": "vector embedding dimension mismatch",
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

- array of `{ "name": string, "entity_type": string, "relation_type": string }`

Notes:

- Entities are derived from the graph observations linked to the memory id (focus: "world facts" like technologies, services, errors, commands, paths).

## Graph tools

`remember` updates graph tables and the following tools can be used.

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

- object containing entities and relations (shape depends on stored data)

## `maintenance`

Manual maintenance operations (never runs automatically).

Inputs:

- `action` (string, required): one of
  - `vacuum`
  - `rebuild_graph` (destructive): rebuild `entities` + `relations` from stored observations
  - `purge_all` (destructive)
  - `purge_session` (destructive, requires `session_id`)
  - `prune_older_than` (destructive, requires `older_than_days`)
- `confirm` (bool, optional): required for destructive actions (everything except `vacuum`)
- `session_id` (string, optional): used by `purge_session`
- `older_than_days` (int, optional): used by `prune_older_than`

Returns:

- `{ "status": "ok", ... }` with action-specific fields (for example `deleted`)

## `diagnostics`

Return environment and DB diagnostics (defaults, table list, counts).

Inputs: none

Returns:

- object with:
  - `db_url`, embedding model info
  - defaults (top_k/top_p/weights)
  - NER config (enabled/model/ctx/threads/etc)
  - tables and row counts

## `health`

Lightweight health check.

Inputs: none

Returns:

- `{ "status": "ok", ... }` including:
  - `db_url`, embedding dimension/model
  - NER enablement
  - log file path (if configured)
