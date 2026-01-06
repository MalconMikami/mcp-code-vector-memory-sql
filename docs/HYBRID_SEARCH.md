# Hybrid search (vectors + FTS + graph)

`mcp-code-vector-memory-sql` combines multiple retrieval signals to get better "developer
memory" results:

- **Vector search**: semantic similarity (libSQL vector backend)
- **FTS5**: exact/fuzzy term matches, merged into results
- **Graph**: entity-centric lookup via `get_context_graph`

This document explains the retrieval pipeline. For tuning, see
`docs/TUNING_GUIDE.md`.

## What happens during `search_memory`

At a high level, `search_memory(query, session_id, limit, top_p)` does:

1. **Vector retrieval** (if enabled)
   - Embeds the query with `fastembed`
   - Retrieves `limit * CODE_MEMORY_OVERSAMPLE_K` nearest neighbors
2. **FTS merge** (if enabled)
   - Runs an FTS5 lookup for the query
   - Marks vector hits that also match FTS, and adds extra FTS-only hits
3. **Fallback**
   - If nothing matched, return the most recent observations (as a last resort)
4. **Re-ranking**
   - Computes a score that blends:
     - vector distance (lower is better)
     - optional FTS bonus
     - priority (1..5; 1 is most important)
     - recency penalty
5. **Recency filter (`top_p`)**
   - Keeps only the newest `top_p` fraction of the candidate set
6. **Return**
   - Returns the top `limit` items after ranking

## Why merge FTS with vectors?

Vectors are great at "meaning", but code workflows also need exact matches:

- symbol names (`FooBarService`, `get_user_by_id`)
- file paths (`src/api/auth.py`)
- error messages and stack traces

FTS complements vectors by making sure exact tokens can still surface, and then
the combined ranking decides what should come first.

## Tuning

See `docs/TUNING_GUIDE.md`.

## Feature flags and behavior changes

This project does not expose feature flags for vector/FTS/graph; they are always enabled.

## Suggested defaults

These are starting points, not rules.

### General coding memory (balanced)

```json
{
  "CODE_MEMORY_TOP_K": "12",
  "CODE_MEMORY_TOP_P": "0.6",
  "CODE_MEMORY_OVERSAMPLE_K": "4"
}
```

### More semantic recall (keep older items)

```json
{
  "CODE_MEMORY_TOP_P": "1.0",
  "CODE_MEMORY_OVERSAMPLE_K": "6"
}
```

### More exact-match bias (names/paths/errors)

```json
{
  "CODE_MEMORY_FTS_BONUS": "0.2"
}
```
