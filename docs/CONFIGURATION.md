# Configuration

This page documents all environment variables supported by `mcp-code-vector-memory-sql`, with
behavior notes and practical examples.

If you are looking for a shorter overview, see `README.md`.

## Configuration sources and precedence

`mcp-code-vector-memory-sql` can read configuration from:

1. Environment variables (highest priority)
2. A JSONC config file at `src/code-vector-memory.jsonc` (supports `//` comments)
3. Code defaults (lowest priority)

If a key exists in the JSON config file with value `null`, it is treated as "not set".

## Storage and workspace

### `CODE_MEMORY_WORKSPACE`

Root folder used by MCP resources:

- `resource://workspace` (directory browser)
- `resource://readme` (project README)

Default: current working directory (cwd).

Example (point resources at your repo, while keeping the DB elsewhere):

```json
{
  "CODE_MEMORY_WORKSPACE": "C:/repo",
  "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080"
}
```

## Embeddings (vector search)

Embeddings are produced by `fastembed` (CPU) and stored via libSQL vector functions
(`vector32`, `vector_distance_cos`, `libsql_vector_idx`) when using `CODE_MEMORY_DB_URL`.

### `CODE_MEMORY_EMBED_MODEL`

The embedding model name.

- Default: `BAAI/bge-small-en-v1.5`
- Must be supported by `fastembed` (see `docs/MODELS.md`)

Example:

```json
{
  "CODE_MEMORY_EMBED_MODEL": "snowflake/snowflake-arctic-embed-s"
}
```

### `CODE_MEMORY_EMBED_DIM`

The embedding vector dimension. This is critical because the embedding column is
created with a fixed size (for example `float[384]` or `float[768]`).

- Default: `384` (matches `BAAI/bge-small-en-v1.5`)
- Required when you change to a model with a different dimension
- If you change the embedding model/dimension, use a new DB file (or delete the
  old one) so the vector table can be created with the correct dimension

Safe switch example (new DB file + explicit dim):

```json
{
  "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
  "CODE_MEMORY_EMBED_MODEL": "nomic-ai/nomic-embed-text-v1.5",
  "CODE_MEMORY_EMBED_DIM": "768"
}
```

### `CODE_MEMORY_MODEL_DIR`

Directory used to cache embedding models.

This is mapped to `FASTEMBED_CACHE_PATH` and `HF_HOME` under the hood.

Default: `~/.cache/code-memory`.

Example:

```json
{
  "CODE_MEMORY_MODEL_DIR": "C:/Users/you/.cache/code-memory/models"
}
```

## Features

This project always enables:

- Vector search
- FTS5
- Entity/relation extraction (graph)

## Search parameters (ranking)

The search pipeline:

1. Vector search retrieves `limit * CODE_MEMORY_OVERSAMPLE_K` candidates
2. Optional FTS results are merged into the candidate set
3. Results are re-ranked (lower score is better) using:
   - vector distance
   - optional FTS bonus
   - priority
   - recency
4. A recency filter keeps only the newest `top_p` fraction of the candidate set

### `CODE_MEMORY_TOP_K`

Default `limit` used by `search_memory` when a client does not pass a `limit`.

- Default: `12`

### `CODE_MEMORY_TOP_P`

Recency filter used after ranking.

- Range: `(0, 1]`
- Default: `0.6`
- Example: if there are 20 candidates and `top_p=0.6`, the newest 12 candidates
  (60%) are kept before returning the top results.

### `CODE_MEMORY_OVERSAMPLE_K`

How many candidates are retrieved before re-ranking.

- Default: `4`
- Effective candidates = `limit * OVERSAMPLE_K`

Higher values can improve recall (more candidates to re-rank) at the cost of
CPU and DB work.

### `CODE_MEMORY_PRIORITY_WEIGHT`

How much `priority` affects ranking.

- Default: `0.15`
- Priority uses a 1..5 scale (1 is most important, 5 is least important)
- This weight increases the score by `priority_weight * (priority - 1)` so
  lower priorities are favored

Example:

```json
{
  "CODE_MEMORY_PRIORITY_WEIGHT": "0.25"
}
```

### `CODE_MEMORY_RECENCY_WEIGHT`

How much recency affects ranking.

- Default: `0.2`
- Older entries get a larger penalty (higher score)

Example:

```json
{
  "CODE_MEMORY_RECENCY_WEIGHT": "0.1"
}
```

### `CODE_MEMORY_FTS_BONUS`

Bonus applied when an item is matched by FTS.

- Default: `0.1`
- This is subtracted from the score (lower score is better), so FTS hits are
  slightly preferred when everything else is equal.

Example:

```json
{
  "CODE_MEMORY_FTS_BONUS": "0.2"
}
```

## Logging

### `CODE_MEMORY_LOG_LEVEL`

Default: `INFO`.

### `CODE_MEMORY_LOG_DIR` / `CODE_MEMORY_LOG_FILE`

Where logs are written.

Notes:

- `CODE_MEMORY_LOG_DIR` takes precedence over `CODE_MEMORY_LOG_FILE` if both are set.
- If you pass a directory path, `mcp-code-vector-memory-sql` creates a timestamped file inside it.
- If you pass a file path (with an extension), the parent directory is used and
  a timestamped file is still created (the filename you passed is not used).

Example:

```json
{
  "CODE_MEMORY_LOG_DIR": "C:/Users/you/.cache/code-memory/logs"
}
```

Example (file path - directory is used):

```json
{
  "CODE_MEMORY_LOG_FILE": "C:/Users/you/.cache/code-memory/logs/server.log"
}
```

## Local NER (GGUF)

Local entity/relation extraction is optional and requires a GGUF file on disk.

### `CODE_MEMORY_NER_MODEL`

Full path to a GGUF model file (or a Hugging Face repo id). If unset, local NER is disabled.

Behavior:

- If `CODE_MEMORY_NER_MODEL` is not set, local NER is disabled (regex fallback still runs).
- If `CODE_MEMORY_NER_MODEL` is set to a local `.gguf` path, that file is used.
- If `CODE_MEMORY_NER_MODEL` is set to a Hugging Face repo id (for example
  `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`), the server will download a suitable
  `.gguf` from that repo into `CODE_MEMORY_MODEL_DIR/gguf` (prefers `Q4_K_M` when available).

Example:

```json
{
  "CODE_MEMORY_NER_MODEL": "C:/models/qwen2.5-coder-0.5b-instruct.gguf"
}
```

Example (Hugging Face repo id):

```json
{
  "CODE_MEMORY_NER_MODEL": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
}
```

Tuning variables (optional):

- `CODE_MEMORY_NER_CTX`
- `CODE_MEMORY_NER_THREADS`
- `CODE_MEMORY_NER_MAX_TOKENS`
- `CODE_MEMORY_NER_TEMPERATURE`
- `CODE_MEMORY_NER_TOP_P`
- `CODE_MEMORY_NER_REPEAT_PENALTY`
- `CODE_MEMORY_NER_GPU_LAYERS`
- `CODE_MEMORY_NER_MAX_INPUT_CHARS`
- `CODE_MEMORY_NER_PROMPT`
- `CODE_MEMORY_AUTO_INSTALL` and `CODE_MEMORY_PIP_ARGS`

Auto-download controls:

- `CODE_MEMORY_NER_AUTO_DOWNLOAD` (default `1`)

## Session id fallback

### `CODE_MEMORY_SESSION_ID`

If your client does not pass `session_id`, the server tries to use
`CODE_MEMORY_SESSION_ID` as a fallback. Most MCP clients should pass `session_id`
explicitly.

## Copy/paste examples

### Minimal setup (defaults)

```json
{}
```

### Fast + small (default embedding) and keep only newer items

```json
{
  "CODE_MEMORY_TOP_K": "10",
  "CODE_MEMORY_TOP_P": "0.8"
}
```

### FTS-only mode (no vectors)

```json
{}
```
### `CODE_MEMORY_DB_URL` (required)

Connect to a libSQL server using the Python `libsql-client`.

- This project always uses a libSQL backend; this setting is required.
- Can point to a local `sqld` on localhost (recommended for “local server” mode) or a remote Turso database.

Examples:

```json
{ "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080" }
```

```json
{ "CODE_MEMORY_DB_URL": "libsql://your-db.turso.io" }
```

### `CODE_MEMORY_DB_AUTH_TOKEN`

Auth token used with `CODE_MEMORY_DB_URL` (same idea as `LIBSQL_AUTH_TOKEN`).

```json
{
  "CODE_MEMORY_DB_URL": "libsql://your-db.turso.io",
  "CODE_MEMORY_DB_AUTH_TOKEN": "..."
}
```
