# Benchmarks

This page documents how we measure performance for `mcp-code-vector-memory-sql`, plus a snapshot from the latest run.

## How to run

Benchmarks are implemented with `pytest-benchmark` in `tests/benchmarks/`.

PowerShell:

```powershell
$env:RUN_BENCHMARKS=1
$ts = Get-Date -Format "yyyy-MM-dd-HH-mm"
python -m pytest -q tests/benchmarks --benchmark-only --benchmark-json=("docs/benchmarks/$ts.json")
```

Notes:
- Benchmarks are skipped by default. You must set `RUN_BENCHMARKS=1`.
- We store benchmark artifacts under `docs/benchmarks/` using timestamped filenames.

If you want to run without summary model auto-download:

```powershell
$env:RUN_BENCHMARKS=1
$env:CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD=0
$ts = Get-Date -Format "yyyy-MM-dd-HH-mm"
python -m pytest -q tests/benchmarks --benchmark-only --benchmark-json=("docs/benchmarks/$ts.json")
```

## Configuration used

Benchmarks use the same environment-based configuration as the MCP server. The benchmark suite configures a temporary DB and sets common flags.

Typical benchmark env (see `tests/benchmarks/test_store_benchmark.py`):
- `CODE_MEMORY_ENABLE_VEC=1` for vector benchmarks
- `CODE_MEMORY_ENABLE_FTS=1` for hybrid/FTS benchmarks
- `CODE_MEMORY_ENABLE_GRAPH=0` (graph is not benchmarked yet)
- `CODE_MEMORY_TOP_K`, `CODE_MEMORY_TOP_P`, `CODE_MEMORY_OVERSAMPLE_K` as needed

Embedding / summary:
- Embeddings: controlled by `CODE_MEMORY_EMBED_MODEL` and `CODE_MEMORY_EMBED_DIM`
- Summaries (GGUF): controlled by `CODE_MEMORY_SUMMARY_MODEL` and `CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD`

## Machine (latest run)

- OS: Windows 10 Pro (10.0.19045) 64-bit
- CPU: AMD Ryzen 5 5600X (6 cores / 12 threads), max clock ~3.7 GHz
- RAM: ~32 GB (TotalVisibleMemorySize reported by Windows)
- Python: 3.12.1

## Results (latest run)

Command:

```powershell
$env:RUN_BENCHMARKS=1
$ts = Get-Date -Format "yyyy-MM-dd-HH-mm"
python -m pytest -q tests/benchmarks --benchmark-only --benchmark-json=("docs/benchmarks/$ts.json")
```

Summary (mean time and derived ops/sec):

| Benchmark | Mean (ms) | Ops/sec |
|---|---:|---:|
| `test_benchmark_insert_ops` | 1.155 | 865.5 |
| `test_benchmark_embedding_ops` | 5.664 | 176.6 |
| `test_benchmark_vector_search_ops` | 7.988 | 125.2 |
| `test_benchmark_hybrid_search_ops` | 12.747 | 78.4 |
| `test_benchmark_tag_search_fts_ops` | 0.098 | 10230.4 |
| `test_benchmark_tag_search_like_ops` | 0.019 | 52355.1 |

Raw benchmark artifact:
- `docs/benchmarks/2026-01-04-22-03.json`

## Summary (GGUF) benchmarks

These benchmarks measure `llama-cpp-python` summary generation speed with the configured GGUF model.

We run three input sizes:
- small: short notes (a few lines)
- medium: several paragraphs
- large: many paragraphs (simulates long logs/docs)

To reduce variance, the benchmarks set:
- `CODE_MEMORY_SUMMARY_MAX_TOKENS=128`
- `CODE_MEMORY_SUMMARY_TEMPERATURE=0.0`

CPU-only:

```powershell
$env:RUN_BENCHMARKS=1
$ts = Get-Date -Format "yyyy-MM-dd-HH-mm"
python -m pytest -q tests/benchmarks -k "summary_gguf_cpu" -vv --benchmark-only --benchmark-json=("docs/benchmarks/$ts.json")
```

GPU (optional):
- Requires a `llama-cpp-python` build with GPU support working on your machine.
- Enable by setting `CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS>0`.

```powershell
$env:RUN_BENCHMARKS=1
$env:CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS=999
$ts = Get-Date -Format "yyyy-MM-dd-HH-mm"
python -m pytest -q tests/benchmarks -k "summary_gguf_gpu" -vv --benchmark-only --benchmark-json=("docs/benchmarks/$ts.json")
```

Tip: to run only non-summary store benchmarks, use `-k "not summary_gguf"`.
