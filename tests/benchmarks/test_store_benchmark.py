import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("pytest_benchmark")


def _import_server_with_env(tmp_path: Path):
    db_path = tmp_path / "bench.db"
    os.environ["CODE_MEMORY_DB_URL"] = f"file:{db_path}"
    os.environ["CODE_MEMORY_DB_AUTH_TOKEN"] = ""
    os.environ["CODE_MEMORY_LOG_LEVEL"] = "ERROR"
    os.environ["CODE_MEMORY_AUTO_INSTALL"] = "1"
    os.environ.setdefault("CODE_MEMORY_NER_MODEL", "")
    os.environ.setdefault("CODE_MEMORY_NER_AUTO_DOWNLOAD", "0")
    os.environ["CODE_MEMORY_CONFIG_PATH"] = str(tmp_path / "no-config.jsonc")

    if "code_memory.server" in list(importlib.sys.modules.keys()):
        importlib.reload(importlib.import_module("code_memory.server"))
    return importlib.import_module("code_memory.server")


def _benchmarks_enabled() -> bool:
    return os.getenv("RUN_BENCHMARKS", "0").lower() in ("1", "true", "yes")


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
            content=f"Record {i}: vector search and embedding performance notes.",
            session_id=session_id,
            kind="bench",
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    def _run():
        s.store.search(
            query="vector search performance",
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
            content=f"Record {i}: error stacktrace in module_{i % 10}.py about embedding dimension mismatch.",
            session_id=session_id,
            kind="bench",
            tags="bench,errors",
            priority=3,
            metadata={"i": i},
        )

    def _run():
        s.store.search(
            query="embedding dimension mismatch module_1.py",
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

    session_id = "bench-session"
    for i in range(200):
        s.store.add(
            content=f"Record {i}: tag-search benchmark.",
            session_id=session_id,
            kind="bench",
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    conn = s._connect_db(load_vec=False)

    def _run():
        res = conn.execute(
            """
            SELECT o.id
            FROM observations_fts f
            JOIN observations o ON f.rowid = o.id
            WHERE observations_fts MATCH ?
            AND o.session_id = ?
            LIMIT 10
            """,
            ("tags:bench", session_id),
        )
        _ = res.rows

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
            tags="bench,perf",
            priority=3,
            metadata={"i": i},
        )

    conn = s._connect_db(load_vec=False)

    def _run():
        res = conn.execute(
            """
            SELECT id
            FROM observations
            WHERE session_id = ?
            AND tags LIKE ?
            ORDER BY created_at DESC
            LIMIT 10
            """,
            (session_id, "%bench%"),
        )
        _ = res.rows

    try:
        benchmark(_run)
    finally:
        conn.close()
