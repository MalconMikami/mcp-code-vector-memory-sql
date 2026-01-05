import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")


def _import_server_with_env(tmp_path: Path):
    db_path = tmp_path / "bench.db"
    os.environ["CODE_MEMORY_DB_PATH"] = str(db_path)
    os.environ["CODE_MEMORY_ENABLE_VEC"] = "1"
    os.environ["CODE_MEMORY_ENABLE_FTS"] = "1"
    os.environ["CODE_MEMORY_ENABLE_GRAPH"] = "0"
    os.environ["CODE_MEMORY_LOG_LEVEL"] = "ERROR"
    os.environ["CODE_MEMORY_AUTO_INSTALL"] = "1"
    os.environ["CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD"] = "1"
    os.environ.setdefault("CODE_MEMORY_SUMMARY_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
    os.environ["CODE_MEMORY_CONFIG_PATH"] = str(tmp_path / "no-config.jsonc")

    if "code_memory.server" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("code_memory.server"))
    return importlib.import_module("code_memory.server")


def _benchmarks_enabled() -> bool:
    return os.getenv("RUN_BENCHMARKS", "0").lower() in ("1", "true", "yes")


def _summaries_enabled(s) -> bool:
    try:
        return bool(s._summary_enabled())
    except Exception:
        return False


def _import_summary_with_env(tmp_path: Path, *, gpu_layers: int):
    os.environ["CODE_MEMORY_DB_PATH"] = str(tmp_path / "bench.db")
    os.environ["CODE_MEMORY_ENABLE_VEC"] = "0"
    os.environ["CODE_MEMORY_ENABLE_FTS"] = "0"
    os.environ["CODE_MEMORY_ENABLE_GRAPH"] = "0"
    os.environ["CODE_MEMORY_LOG_LEVEL"] = "ERROR"
    os.environ["CODE_MEMORY_AUTO_INSTALL"] = "1"
    os.environ["CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD"] = "1"
    os.environ.setdefault("CODE_MEMORY_SUMMARY_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
    os.environ["CODE_MEMORY_SUMMARY_GPU_LAYERS"] = str(int(gpu_layers))
    os.environ.setdefault("CODE_MEMORY_SUMMARY_MAX_TOKENS", "128")
    os.environ.setdefault("CODE_MEMORY_SUMMARY_TEMPERATURE", "0.0")
    os.environ["CODE_MEMORY_CONFIG_PATH"] = str(tmp_path / "no-config.jsonc")

    mod = importlib.import_module("code_memory.summary")
    return importlib.reload(mod)


def _make_summary_payload(*, size: str) -> str:
    header = (
        "Summarize the following developer notes into 1-3 sentences. "
        "Focus on what changed, why, and what to do next.\n\n"
        "NOTES:\n"
    )
    if size == "small":
        body = (
            "- Changed ranking to prefer recent and high-priority items.\n"
            "- Added FTS merge.\n"
            "- Updated docs.\n"
        )
    elif size == "medium":
        paragraph = (
            "We refactored the memory MCP into modules, improved search reranking with FTS bonus and priority/recency, "
            "added a JSONC config file with env override precedence, and documented client setup and tuning. "
            "Benchmarks now use pytest-benchmark.\n"
        )
        body = paragraph * 8
    elif size == "large":
        paragraph = (
            "Context: The MCP server stores session-scoped memories in SQLite with optional sqlite-vec embeddings and "
            "optional FTS5 indexing. We tuned retrieval by oversampling candidates, applying an FTS bonus for exact hits, "
            "and reranking with priority and recency. We also added optional GGUF summaries and wrote a configuration "
            "template. Validation includes integration tests plus pytest-benchmark for inserts/search.\n"
        )
        body = paragraph * 40
    else:
        raise ValueError(f"unknown size: {size}")
    return header + body


@pytest.mark.benchmark
def test_benchmark_insert_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)

    # Warm the embedder outside measurements (model load/caching can be expensive).
    _ = s.store.embedder.embed("warmup")

    session_id = "bench-session"
    payload = "Benchmark insert payload. The quick brown fox jumps over the lazy dog."

    def _run():
        s.store.add(
            content=payload,
            session_id=session_id,
            kind="bench",
            summary=None,
            tags="bench,perf",
            priority=3,
            metadata={"k": "v"},
        )

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_embedding_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)

    # Warmup (model load)
    _ = s.store.embedder.embed("warmup")
    payload = "Embedding benchmark payload. The quick brown fox jumps over the lazy dog."

    def _run():
        s.store.embedder.embed(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_vector_search_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)
    _ = s.store.embedder.embed("warmup")

    session_id = "bench-session"
    for i in range(100):
        s.store.add(
            content=f"Record {i}: sqlite-vec and fastembed performance notes.",
            session_id=session_id,
            kind="bench",
            summary=None,
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    def _run():
        s.store.search(
            query="sqlite-vec performance",
            session_id=session_id,
            limit=10,
            top_p=s.DEFAULT_TOP_P,
            use_fts=False,
        )

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_hybrid_search_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)
    _ = s.store.embedder.embed("warmup")

    session_id = "bench-session"
    for i in range(100):
        s.store.add(
            content=f"Record {i}: error stacktrace in module_{i % 10}.py about sqlite-vec dimension mismatch.",
            session_id=session_id,
            kind="bench",
            summary=None,
            tags="bench,errors",
            priority=3,
            metadata={"i": i},
        )

    def _run():
        s.store.search(
            query="sqlite-vec dimension mismatch module_1.py",
            session_id=session_id,
            limit=10,
            top_p=s.DEFAULT_TOP_P,
            use_fts=True,
        )

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_tag_search_fts_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)
    _ = s.store.embedder.embed("warmup")

    if not s.ENABLE_FTS:
        pytest.skip("FTS disabled (CODE_MEMORY_ENABLE_FTS=0).")

    session_id = "bench-session"
    for i in range(200):
        s.store.add(
            content=f"Record {i}: tag-search benchmark.",
            session_id=session_id,
            kind="bench",
            summary=None,
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    conn = s._connect_db(load_vec=False)

    def _run():
        conn.execute(
            """
            SELECT m.id
            FROM memories_fts f
            JOIN memories m ON f.rowid = m.id
            WHERE memories_fts MATCH ?
            AND m.session_id = ?
            LIMIT 10
            """,
            ("tags:bench", session_id),
        ).fetchall()

    try:
        benchmark(_run)
    finally:
        conn.close()


@pytest.mark.benchmark
def test_benchmark_tag_search_like_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    s = _import_server_with_env(tmp_path)
    _ = s.store.embedder.embed("warmup")

    session_id = "bench-session"
    for i in range(200):
        s.store.add(
            content=f"Record {i}: tag-search benchmark.",
            session_id=session_id,
            kind="bench",
            summary=None,
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    conn = s._connect_db(load_vec=False)

    def _run():
        conn.execute(
            """
            SELECT id
            FROM memories
            WHERE session_id = ?
            AND tags LIKE ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (session_id, "%bench%"),
        ).fetchall()

    try:
        benchmark(_run)
    finally:
        conn.close()


@pytest.mark.benchmark
def test_benchmark_summary_gguf_cpu_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=0)
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="small")

    # Warmup (model load)
    _ = summary.generate_summary(payload)

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_summary_gguf_cpu_medium_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=0)
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="medium")
    _ = summary.generate_summary(payload)

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_summary_gguf_cpu_large_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=0)
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="large")
    _ = summary.generate_summary(payload)

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_summary_gguf_gpu_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    # Default to 999 to request "all layers" on supported GPU builds, but allow override.
    gpu_layers = int(os.getenv("CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS", "999"))
    if gpu_layers <= 0:
        pytest.skip("Set CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS>0 to run the GPU summary benchmark.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=gpu_layers)
    if not summary.gpu_offload_supported():
        pytest.skip("llama-cpp-python build does not support GPU offload (llama_supports_gpu_offload=false).")
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="small")

    # Warmup (may fail if llama-cpp has no GPU backend configured)
    try:
        _ = summary.generate_summary(payload)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU summary warmup failed (gpu_layers={gpu_layers}): {exc}")

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_summary_gguf_gpu_medium_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    gpu_layers = int(os.getenv("CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS", "999"))
    if gpu_layers <= 0:
        pytest.skip("Set CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS>0 to run the GPU summary benchmark.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=gpu_layers)
    if not summary.gpu_offload_supported():
        pytest.skip("llama-cpp-python build does not support GPU offload (llama_supports_gpu_offload=false).")
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="medium")
    try:
        _ = summary.generate_summary(payload)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU summary warmup failed (gpu_layers={gpu_layers}): {exc}")

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)


@pytest.mark.benchmark
def test_benchmark_summary_gguf_gpu_large_ops(benchmark, tmp_path: Path):
    if not _benchmarks_enabled():
        pytest.skip("Set RUN_BENCHMARKS=1 to run benchmarks.")

    gpu_layers = int(os.getenv("CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS", "999"))
    if gpu_layers <= 0:
        pytest.skip("Set CODE_MEMORY_BENCH_SUMMARY_GPU_LAYERS>0 to run the GPU summary benchmark.")

    summary = _import_summary_with_env(tmp_path, gpu_layers=gpu_layers)
    if not summary.gpu_offload_supported():
        pytest.skip("llama-cpp-python build does not support GPU offload (llama_supports_gpu_offload=false).")
    if not summary.summary_enabled():
        pytest.skip("Summaries disabled (set CODE_MEMORY_SUMMARY_MODEL or enable auto-download).")

    payload = _make_summary_payload(size="large")
    try:
        _ = summary.generate_summary(payload)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"GPU summary warmup failed (gpu_layers={gpu_layers}): {exc}")

    def _run():
        summary.generate_summary(payload)

    benchmark(_run)
