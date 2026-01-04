# Hybrid search (vectors + FTS + optional graph)

`mcp-code-vector-memory-sql` combines multiple retrieval signals to get better "developer
memory" results:

- **Vector search**: semantic similarity (sqlite-vec)
- **FTS5** (optional): exact/fuzzy term matches, merged into results
- **Graph** (optional): entity-centric lookup via `get_context_graph`

This document explains the retrieval pipeline and how to tune it.

## What happens during `search_memory`

At a high level, `search_memory(query, session_id, limit, top_p)` does:

1. **Vector retrieval** (if enabled)
   - Embeds the query with `fastembed`
   - Retrieves `limit * CODE_MEMORY_OVERSAMPLE_K` nearest neighbors
2. **FTS merge** (if enabled)
   - Runs an FTS5 lookup for the query
   - Marks vector hits that also match FTS, and adds extra FTS-only hits
3. **Fallback**
   - If nothing matched, return the most recent memories for the session
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

## Tuning guide

All tuning is done via environment variables. Full reference:
`docs/CONFIGURATION.md`.

### Candidate set size: `CODE_MEMORY_TOP_K` + `CODE_MEMORY_OVERSAMPLE_K`

- `CODE_MEMORY_TOP_K` is the default `limit` when the client does not pass one
- `CODE_MEMORY_OVERSAMPLE_K` controls how many candidates you retrieve before
  re-ranking:
  - candidates = `limit * oversample_k`

Tradeoff:

- Higher oversample can improve recall (more candidates to re-rank)
- Too high increases CPU/DB work

### Recency filtering: `CODE_MEMORY_TOP_P`

After ranking, `top_p` keeps only the newest fraction of the candidate set.

Practical use:

- Set `top_p` closer to `1.0` if you want older-but-relevant memories to still
  appear
- Keep it lower (for example `0.5` to `0.7`) if you mostly care about "what just
  happened" in the current session

### Priority vs recency: `CODE_MEMORY_PRIORITY_WEIGHT` and `CODE_MEMORY_RECENCY_WEIGHT`

`mcp-code-vector-memory-sql` uses a 1..5 priority scale (1 is most important).

- Increase `CODE_MEMORY_PRIORITY_WEIGHT` if you want "high priority" memories to
  remain near the top even when older
- Increase `CODE_MEMORY_RECENCY_WEIGHT` if you want newer memories to dominate

### FTS preference: `CODE_MEMORY_FTS_BONUS`

FTS hits get a score bonus (lower is better).

- Increase `CODE_MEMORY_FTS_BONUS` if exact term matches should outrank purely
  semantic matches
- Keep it smaller if you find FTS "overpowers" semantic retrieval

## Feature flags and behavior changes

### Vectors disabled (`CODE_MEMORY_ENABLE_VEC=0`)

`search_memory` becomes:

- FTS lookup (if enabled), otherwise
- "most recent memories" fallback

This is useful if you want a lightweight setup and mostly rely on exact terms.

### FTS disabled (`CODE_MEMORY_ENABLE_FTS=0`)

`search_memory` becomes vector-only with priority/recency ranking.

This is useful if you want purely semantic retrieval and the simplest setup.

### Graph enabled (`CODE_MEMORY_ENABLE_GRAPH=1`)

The graph is updated during `remember` and can be queried via `get_context_graph`.
It is complementary to `search_memory` (entity-centric instead of memory-row
centric).

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
