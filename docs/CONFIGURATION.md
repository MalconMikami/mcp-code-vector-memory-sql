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

### `CODE_MEMORY_DB_PATH`

Full path to the SQLite database file (for example `C:/repo/code_memory.db`).

- If set, it overrides `CODE_MEMORY_DB_DIR`.
- If the parent directory does not exist, SQLite will fail to open the DB.
- Relative paths are resolved from the current working directory.

Example:

```json
{
  "CODE_MEMORY_DB_PATH": "C:/repo/.cache/code_memory.db"
}
```

### `CODE_MEMORY_DB_DIR`

Directory where the DB file will be created. The DB filename is always
`code_memory.db`.

- Default: current working directory (cwd)
- Ignored when `CODE_MEMORY_DB_PATH` is set

Example:

```json
{
  "CODE_MEMORY_DB_DIR": "C:/repo"
}
```

### `CODE_MEMORY_WORKSPACE`

Root folder used by MCP resources:

- `resource://workspace` (directory browser)
- `resource://readme` (project README)

Default: current working directory (cwd).

Example (point resources at your repo, while keeping the DB elsewhere):

```json
{
  "CODE_MEMORY_WORKSPACE": "C:/repo",
  "CODE_MEMORY_DB_PATH": "C:/Users/you/.cache/code-memory/code_memory.db"
}
```

## Embeddings (vector search)

Embeddings are produced by `fastembed` (CPU) and stored via `sqlite-vec`.

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

The embedding vector dimension. This is critical because the sqlite-vec table is
created with a fixed size (for example `float[384]` or `float[768]`).

- Default: `384` (matches `BAAI/bge-small-en-v1.5`)
- Required when you change to a model with a different dimension
- If you change the embedding model/dimension, use a new DB file (or delete the
  old one) so the vector table can be created with the correct dimension

Safe switch example (new DB file + explicit dim):

```json
{
  "CODE_MEMORY_DB_PATH": "C:/repo/.cache/code_memory_nomic.db",
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

## Feature flags

### `CODE_MEMORY_ENABLE_VEC`

Enable/disable vector search (sqlite-vec).

- Default: `1`
- When disabled, `search_memory` falls back to FTS (if enabled) and then to
  "recent memories" as a last resort.
- Any value other than `0`, `false`, or `no` (case-insensitive) is treated as
  enabled.

### `CODE_MEMORY_ENABLE_FTS`

Enable/disable FTS5.

- Default: `1`
- When enabled, FTS matches are merged into the candidate set and get a bonus
  during re-ranking (`CODE_MEMORY_FTS_BONUS`).
- Any value other than `0`, `false`, or `no` (case-insensitive) is treated as
  enabled.

### `CODE_MEMORY_ENABLE_GRAPH`

Enable/disable the knowledge graph tables and updates.

- Default: `0`
- When enabled, entity extraction runs on `remember` and the graph tables
  (`graph_entities`, `graph_observations`, `graph_relations`) are maintained.
- Any value other than `0`, `false`, or `no` (case-insensitive) is treated as
  enabled.

Example:

```json
{
  "CODE_MEMORY_ENABLE_GRAPH": "1"
}
```

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

## Local summarization (GGUF)

Local summaries are optional and require a GGUF file on disk.

### `CODE_MEMORY_SUMMARY_MODEL`

Full path to a GGUF model file. If unset, local summaries are disabled.

Behavior:

- If `CODE_MEMORY_SUMMARY_MODEL` is not set, local summaries are disabled.
- If `CODE_MEMORY_SUMMARY_MODEL` is set to a local `.gguf` path, that file is used.
- If `CODE_MEMORY_SUMMARY_MODEL` is set to a Hugging Face repo id (for example
  `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`), the server will download a suitable
  `.gguf` from that repo into `CODE_MEMORY_MODEL_DIR/gguf` (prefers `Q4_K_M` when available).

Example:

```json
{
  "CODE_MEMORY_SUMMARY_MODEL": "C:/models/qwen2.5-coder-0.5b-instruct.gguf"
}
```

Example (Hugging Face repo id):

```json
{
  "CODE_MEMORY_SUMMARY_MODEL": "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
}
```

The remaining summary-related variables tune llama.cpp runtime and generation:

- `CODE_MEMORY_SUMMARY_CTX`
- `CODE_MEMORY_SUMMARY_THREADS`
- `CODE_MEMORY_SUMMARY_MAX_TOKENS`
- `CODE_MEMORY_SUMMARY_TEMPERATURE`
- `CODE_MEMORY_SUMMARY_TOP_P`
- `CODE_MEMORY_SUMMARY_REPEAT_PENALTY`
- `CODE_MEMORY_SUMMARY_GPU_LAYERS`
- `CODE_MEMORY_SUMMARY_MAX_CHARS`
- `CODE_MEMORY_SUMMARY_PROMPT`
- `CODE_MEMORY_AUTO_INSTALL` and `CODE_MEMORY_PIP_ARGS`

Auto-download controls:

- `CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD` (default `1`)

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
{
  "CODE_MEMORY_ENABLE_VEC": "0",
  "CODE_MEMORY_ENABLE_FTS": "1"
}
```
