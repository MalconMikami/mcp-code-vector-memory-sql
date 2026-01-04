# Memory techniques (how to store better context)

`mcp-code-vector-memory-sql` is a storage and retrieval layer. The quality of results depends a
lot on *what* you store and *how* you structure it.

This page collects practical patterns for MCP clients and coding assistants.

## What to store

Good "memories" tend to be:

- durable (still relevant later)
- specific (actionable and testable)
- compact (summarized, with details available when needed)

Examples of high-signal memories:

- architectural decisions and tradeoffs
- API contracts (request/response shapes, error handling)
- project conventions (lint rules, formatting, folder layout)
- recurring commands and local setup details
- known gotchas and debugging notes

Examples of low-signal memories:

- long logs without conclusions
- raw code dumps that will change tomorrow
- secrets, tokens, or personal data (these are filtered anyway)

## Use the fields intentionally

### `summary`

If you can provide a good one-liner summary, do it. Search and re-ranking often
benefit from concise descriptions.

If you do not provide a summary, `mcp-code-vector-memory-sql` may generate one locally (if
GGUF summarization is enabled).

### `tags`

Use tags to create lightweight facets you can search for later.

Examples:

- `bugfix,auth,jwt`
- `design,db,migrations`
- `perf,sqlite,fts`

### `kind`

Use `kind` for broad categories (for example `decision`, `bugfix`, `plan`,
`release`, `note`).

### `priority` (1..5)

Priority is part of ranking. Use it to keep important context visible:

- `1`: critical decisions, invariants, must-not-forget constraints
- `2`: important implementation notes, known pitfalls
- `3`: default
- `4-5`: low-signal breadcrumbs you may still want occasionally

### `metadata_json`

Use metadata for structured information you may want to parse later, such as:

- ticket IDs
- file paths touched
- hashes/versions
- URLs to related docs

## Chunking strategies

Vector search works best when stored text is "about one thing".

Instead of storing one huge memory, store a few smaller ones:

- one for the decision
- one for the implementation notes
- one for the validation steps

This improves retrieval precision and makes summaries more stable.

## Retrieval strategies

### Start broad, then narrow

1. Search with a broad query to find candidate memories
2. Refine the query with concrete terms (symbols, filenames, error text)

### Use recency filtering intentionally

If you want a "what did we just do" experience, keep `CODE_MEMORY_TOP_P` lower.
If you want stronger recall across the session history, increase it toward `1.0`.

## Local summaries: when to enable

Enable local summaries when:

- you want more compact memories without sending data to a cloud API
- you store longer notes and want an automatic one-liner summary

Keep it disabled when:

- you prefer to write summaries explicitly
- you want a lighter install and fewer dependencies

See `docs/MODELS.md` and `docs/CONFIGURATION.md` for GGUF setup.
