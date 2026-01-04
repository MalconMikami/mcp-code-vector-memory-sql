# Operations

## Logs and diagnostics

- `CODE_MEMORY_LOG_LEVEL` controls verbosity
- `CODE_MEMORY_LOG_DIR` or `CODE_MEMORY_LOG_FILE` choose where logs are written
- `diagnostics()` returns flags, defaults and table counts
- `health()` returns effective configuration and feature availability

## Maintenance

Tool: `maintenance(action, confirm, session_id, older_than_days)`

- `vacuum`: run `VACUUM` (no confirm required)
- `purge_all`: delete all memories (requires `confirm=true`)
- `purge_session`: delete all memories for a `session_id` (requires `confirm=true`)
- `prune_older_than`: delete memories older than N days (requires `confirm=true`)

## Privacy and dedupe

- Sensitive content is skipped by default (tokens, passwords, secrets)
- Recent duplicates are filtered by content hash

## Troubleshooting

- Embedding dimension mismatch: set `CODE_MEMORY_EMBED_DIM` correctly and use a new DB file
- Summaries not enabled: verify `CODE_MEMORY_SUMMARY_MODEL` path and llama-cpp-python install
- sqlite-vec load errors: ensure `sqlite-vec` is installed and compatible with your Python build
