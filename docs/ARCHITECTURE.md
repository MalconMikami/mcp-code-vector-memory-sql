# Architecture

## Components

- FastMCP: MCP server and tool/resource definitions
- MemoryStore: SQLite persistence layer
- fastembed: CPU embeddings
- sqlite-vec: vector table for semantic search
- FTS5: optional full-text index used for merging matches and re-ranking
- Graph (optional): entities, observations and relations
- Summary (optional): local GGUF summaries via llama.cpp

## `remember` flow

1. Resolve `session_id` (input, MCP context, or env fallback)
2. Filter sensitive content and skip recent duplicates (hash window)
3. Optionally generate local summary and basic tags
4. Store in `memories` and create embedding in `vec_memories` (if enabled)
5. Update `memories_fts` via triggers (if enabled)
6. Extract entities and update the graph tables (if enabled)

## `search_memory` flow

1. Embed the query (if vector search is enabled)
2. Query `vec_memories` and retrieve candidates (oversample)
3. Merge FTS hits (if enabled)
4. Re-rank by distance, priority, recency and optional FTS bonus
5. Apply `top_p` recency filtering and return the top results

## `get_context_graph` flow

- Without a query: returns a snapshot of the graph
- With a query: semantic search over graph entities (requires vectors)

## Schema (high level)

- `memories`: content, summary, tags, priority, session_id, metadata, created_at
- `vec_memories`: embeddings
- `memories_fts`: FTS5 index of `memories`
- `entities`: extracted entities per memory row
- `graph_entities`, `graph_observations`, `graph_relations`: knowledge graph (optional)

## MCP resources

- `resource://workspace`: read-only workspace browser
- `resource://readme`: project README file
