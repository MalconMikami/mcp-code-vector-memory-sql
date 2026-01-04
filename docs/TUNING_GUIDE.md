# Tuning guide (memory retrieval)

This document focuses on tuning `search_memory` behavior via environment
variables.

Full configuration reference: `docs/CONFIGURATION.md`.

## How `search_memory` ranks results (mental model)

1. Vector retrieval fetches `limit * CODE_MEMORY_OVERSAMPLE_K` candidates (if enabled)
2. Optional FTS hits are merged into the candidate set
3. Candidates are re-ranked (lower score is better) using:
   - vector distance
   - optional FTS bonus
   - priority
   - recency
4. `top_p` keeps only the newest fraction of the candidate set
5. The final top `limit` items are returned

## Core knobs

### Candidate set size: `CODE_MEMORY_TOP_K` + `CODE_MEMORY_OVERSAMPLE_K`

- `CODE_MEMORY_TOP_K` controls the default `limit` when the client does not pass one
- `CODE_MEMORY_OVERSAMPLE_K` controls how many candidates are fetched before re-ranking:
  - candidates = `limit * oversample_k`

Tradeoff:

- Higher oversample can improve recall (more candidates to re-rank)
- Too high increases CPU/DB work

### Recency filtering: `CODE_MEMORY_TOP_P`

After ranking, `top_p` keeps only the newest fraction of candidates.

Practical use:

- Set `top_p` closer to `1.0` if you want older-but-relevant memories to still appear
- Keep it lower (for example `0.5` to `0.7`) if you mostly care about "what just happened"

### Priority vs recency: `CODE_MEMORY_PRIORITY_WEIGHT` and `CODE_MEMORY_RECENCY_WEIGHT`

`mcp-code-vector-memory-sql` uses a 1..5 priority scale (1 is most important).

- Increase `CODE_MEMORY_PRIORITY_WEIGHT` if you want high-priority memories to remain near the top even when older
- Increase `CODE_MEMORY_RECENCY_WEIGHT` if you want newer memories to dominate

### FTS preference: `CODE_MEMORY_FTS_BONUS`

FTS hits get a score bonus (lower is better).

- Increase `CODE_MEMORY_FTS_BONUS` if exact term matches should outrank purely semantic matches
- Keep it smaller if you find FTS "overpowers" semantic retrieval

## Feature-flag presets

### FTS-only mode (no vectors)

```json
{
  "CODE_MEMORY_ENABLE_VEC": "0",
  "CODE_MEMORY_ENABLE_FTS": "1"
}
```

### Vector-only mode (no FTS)

```json
{
  "CODE_MEMORY_ENABLE_VEC": "1",
  "CODE_MEMORY_ENABLE_FTS": "0"
}
```

## Suggested defaults

### Balanced

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
