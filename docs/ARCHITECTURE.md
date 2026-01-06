# Architecture

## Components

- FastMCP: MCP server and tool/resource definitions
- MemoryStore: SQLite persistence layer
- fastembed: CPU embeddings
- Vector search: libSQL vector (libSQL backend)
- FTS5: optional full-text index used for merging matches and re-ranking
- Graph: entities, observations and relations (unified schema)
- NER (optional): local GGUF entity/relation extraction via llama.cpp

## `remember` flow

1. Resolve `session_id` (input, MCP context, or env fallback)
2. Filter sensitive content and skip recent duplicates (hash window)
3. Optionally extract entities/relations and basic tags
4. Store in `observations` (linked to a `memory` node in `entities`)
5. Create embedding in `observations.embedding` (if enabled)
6. Update `observations_fts` via triggers (if enabled)
7. Extract world-fact entities + relations and write to `entities`/`relations` (if enabled)

## `search_memory` flow

1. Embed the query (if vector search is enabled)
2. Query `observations.embedding` via vector distance and retrieve candidates (oversample)
3. Merge FTS hits (if enabled)
4. Re-rank by distance, priority, recency and optional FTS bonus
5. Apply `top_p` recency filtering and return the top results

## `get_context_graph` flow

- Without a query: returns a snapshot of the graph
- With a query: semantic search over graph entities (requires vectors)

## Schema (high level)

- `entities`: both memory nodes (`entity_type='memory'`) and extracted world-fact entities
- `observations`: stored text (content + tags/metadata), linked to an entity
- `relations`: directed edges between entities, optionally linked to an observation
- `observations.embedding`: embeddings for observations (libSQL vector backend only)
- `observations_fts`: FTS5 index for observations (creates internal FTS5 tables)

## MCP resources

- `resource://workspace`: read-only workspace browser
- `resource://readme`: project README file
