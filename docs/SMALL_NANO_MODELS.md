# Small / nano GGUF models (local summaries)

`mcp-code-vector-memory-sql` can generate summaries locally when you set `CODE_MEMORY_SUMMARY_MODEL`
to a GGUF file. This is optional and runs via `llama-cpp-python`.

This page focuses on choosing and tuning *small* GGUF models for fast, local
summaries during development.

## Why small models?

Small models can be a great fit for summarization because:

- summaries are short (you do not need long-form generation)
- you care more about speed than creativity
- local-first privacy is preserved

## Suggested model: `Qwen2.5-Coder-0.5B`

`Qwen2.5-Coder-0.5B` is a good default when you want:

- very fast CPU inference
- code-aware text generation (good for commit-like summaries)
- low RAM requirements

Tradeoff: being very small, it can miss nuance compared to larger models.

## Tuning knobs in `mcp-code-vector-memory-sql`

See `docs/CONFIGURATION.md` for the full reference. The most useful knobs:

- `CODE_MEMORY_SUMMARY_CTX`: context size (higher uses more RAM)
- `CODE_MEMORY_SUMMARY_THREADS`: CPU parallelism
- `CODE_MEMORY_SUMMARY_MAX_TOKENS`: cap summary length
- `CODE_MEMORY_SUMMARY_TEMPERATURE` / `CODE_MEMORY_SUMMARY_TOP_P`: creativity vs determinism
- `CODE_MEMORY_SUMMARY_PROMPT`: custom prompt for your style

## Finding GGUF builds

Most GGUF builds are shared on the Hugging Face hub. Practical search queries:

- `Qwen2.5-Coder-0.5B GGUF`
- `Llama 3.2 3B GGUF`
- `Instruct GGUF Q4_K_M`

Always verify the model license and the publisher you trust.

## Practical recommendation

Start with:

- a small instruct-style GGUF model
- low temperature (`0.1` to `0.3`)
- a short, strict prompt (1-3 sentences)

Then iterate based on what you actually want to retrieve later.
