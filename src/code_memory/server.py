import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import logging

from fastembed import TextEmbedding
from fastmcp import Context, FastMCP
from fastmcp.resources import DirectoryResource, FileResource
import sqlite_vec

# Tree-sitter é opcional; se não estiver disponível, seguimos sem AST/NER.
try:
    from tree_sitter import Language, Parser
    from tree_sitter_languages import get_language, get_parser

    TREE_SITTER_AVAILABLE = True
    TREE_SITTER_PARSER = get_parser("python")
except Exception:  # pragma: no cover
    TREE_SITTER_AVAILABLE = False
    TREE_SITTER_PARSER = None

# LLM local opcional (llama-cpp-python)
LLAMA_CPP_AVAILABLE = False
Llama = None

sqlite3 = None

ROOT = Path(os.getenv("CODE_MEMORY_WORKSPACE", Path.cwd())).resolve()
DB_DIR = Path(os.getenv("CODE_MEMORY_DB_DIR", Path.cwd()))
DB_PATH = Path(os.getenv("CODE_MEMORY_DB_PATH", DB_DIR / "code_memory.db")).resolve()
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_MODEL_NAME = os.getenv("CODE_MEMORY_EMBED_MODEL", DEFAULT_EMBED_MODEL)
EMBED_DIM_RAW = os.getenv("CODE_MEMORY_EMBED_DIM")
EMBED_DIM_CONFIGURED = EMBED_DIM_RAW is not None
try:
    EMBED_DIM = int(EMBED_DIM_RAW) if EMBED_DIM_RAW else 384
except Exception:
    EMBED_DIM = 384
    EMBED_DIM_CONFIGURED = False
MODEL_CACHE_DIR = Path(os.getenv("CODE_MEMORY_MODEL_DIR", Path.home() / ".cache" / "code-memory"))
DEFAULT_TOP_K = int(os.getenv("CODE_MEMORY_TOP_K", "12"))
DEFAULT_TOP_P = float(os.getenv("CODE_MEMORY_TOP_P", "0.6"))
RECENCY_WEIGHT = float(os.getenv("CODE_MEMORY_RECENCY_WEIGHT", "0.2"))
PRIORITY_WEIGHT = float(os.getenv("CODE_MEMORY_PRIORITY_WEIGHT", "0.15"))
FTS_BONUS = float(os.getenv("CODE_MEMORY_FTS_BONUS", "0.1"))
OVERSAMPLE_K = int(os.getenv("CODE_MEMORY_OVERSAMPLE_K", "4"))
ENABLE_VEC = os.getenv("CODE_MEMORY_ENABLE_VEC", "1").lower() not in ("0", "false", "no")
ENABLE_FTS = os.getenv("CODE_MEMORY_ENABLE_FTS", "1").lower() not in ("0", "false", "no")
ENABLE_GRAPH = os.getenv("CODE_MEMORY_ENABLE_GRAPH", "0").lower() not in ("0", "false", "no")
SUMMARY_MODEL = os.getenv("CODE_MEMORY_SUMMARY_MODEL")  # local GGUF path
SUMMARY_CTX = int(os.getenv("CODE_MEMORY_SUMMARY_CTX", "2048"))
SUMMARY_THREADS = int(os.getenv("CODE_MEMORY_SUMMARY_THREADS", "4"))
SUMMARY_MAX_TOKENS = int(os.getenv("CODE_MEMORY_SUMMARY_MAX_TOKENS", "200"))
SUMMARY_TEMPERATURE = float(os.getenv("CODE_MEMORY_SUMMARY_TEMPERATURE", "0.2"))
SUMMARY_TOP_P = float(os.getenv("CODE_MEMORY_SUMMARY_TOP_P", "0.9"))
SUMMARY_REPEAT_PENALTY = float(os.getenv("CODE_MEMORY_SUMMARY_REPEAT_PENALTY", "1.05"))
SUMMARY_N_GPU_LAYERS = int(os.getenv("CODE_MEMORY_SUMMARY_GPU_LAYERS", "0"))
SUMMARY_MAX_CHARS = int(os.getenv("CODE_MEMORY_SUMMARY_MAX_CHARS", "300"))
SUMMARY_PROMPT = os.getenv("CODE_MEMORY_SUMMARY_PROMPT", "")
SUMMARY_AUTO_INSTALL = os.getenv("CODE_MEMORY_AUTO_INSTALL", "1").lower() not in ("0", "false", "no")
SUMMARY_PIP_ARGS = os.getenv("CODE_MEMORY_PIP_ARGS", "").strip()
LOG_LEVEL = os.getenv("CODE_MEMORY_LOG_LEVEL", "INFO").upper()
LOG_BASE = os.getenv("CODE_MEMORY_LOG_DIR") or os.getenv("CODE_MEMORY_LOG_FILE")

def _resolve_logfile(base: Optional[str]) -> Optional[Path]:
    if not base:
        return None
    base_path = Path(base)
    if base_path.suffix:
        log_dir = base_path.parent if base_path.parent != Path() else Path.cwd()
    else:
        log_dir = base_path
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return log_dir / f"code-memory-{stamp}.log"


LOG_FILE = _resolve_logfile(LOG_BASE)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=str(LOG_FILE) if LOG_FILE else None,
    filemode="a",
)
logger = logging.getLogger("code-memory")

_SUMMARY_LLM = None


def _in_venv() -> bool:
    return getattr(sys, "base_prefix", sys.prefix) != sys.prefix


def _install_llama_cpp() -> None:
    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python"]
    if SUMMARY_PIP_ARGS:
        cmd.extend(SUMMARY_PIP_ARGS.split())
    elif not _in_venv():
        cmd.append("--user")
    logger.info("Tentando instalar dependencia: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_llama_cpp() -> bool:
    global LLAMA_CPP_AVAILABLE, Llama
    if LLAMA_CPP_AVAILABLE and Llama is not None:
        return True
    try:
        from llama_cpp import Llama as _Llama

        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Import llama_cpp falhou: %s", exc)
        if not SUMMARY_AUTO_INSTALL:
            return False
    try:
        _install_llama_cpp()
        from llama_cpp import Llama as _Llama

        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Instalacao llama_cpp falhou: %s", exc)
        return False

# Instruções claras para o cliente MCP (OpenCode)
server = FastMCP(
    name="code-memory",
    instructions=(
        "Memory MCP server for OpenCode (VS Code compatible). Use this as the persistence layer for context "
        "and summaries.\n"
        "Tools:\n"
        "  - remember(content, session_id, kind, summary, tags, priority(1-5), metadata_json): "
        "    store context with vector + FTS + entities. Priority low=5, high=1.\n"
        "    session_id is required (current OpenCode session). Prefer adding summary to "
        "compress context. Sensitive content is filtered; duplicates are skipped.\n"
        "  - search_memory(query, session_id, limit): semantic + vector (sqlite-vec) with FTS re-rank; "
        "    session_id is required to scope results and use priority ordering.\n"
        "    Default retrieval: top_k=CODE_MEMORY_TOP_K and top_p=CODE_MEMORY_TOP_P (prefer newer).\n"
        "  - list_recent(limit): latest memories (with priority/session).\n"
        "  - list_entities(memory_id): functions/classes extracted for that memory.\n"
        "  - upsert_entity(name, entity_type, observations_json, memory_id): add entity + observations to graph.\n"
        "  - add_relation(source, target, relation_type, memory_id): add relation between entities.\n"
        "  - get_entity(name): fetch one entity with observations.\n"
        "  - get_context_graph(query, limit): returns entities + relations (semantic if query is set).\n"
        "  - health(): status, embedding dim, cache dir, logging info.\n"
        "Usage pattern (recommendation):\n"
        "  1) Before big actions, search_memory scoped by session_id to recall prior context.\n"
        "  2) After meaningful changes/decisions, call remember with a concise summary and priority=1-3 "
        "(higher = more important). Use priority=4-5 for low-signal logs.\n"
        "  3) Use tags/kind to classify (e.g., bugfix, design, refactor) and metadata_json for extra structure.\n"
        "Resources: resource://workspace, resource://readme.\n"
        "Env: CODE_MEMORY_EMBED_MODEL + CODE_MEMORY_EMBED_DIM for embeddings; "
        "CODE_MEMORY_MODEL_DIR for cache; CODE_MEMORY_LOG_DIR for logs; "
        "CODE_MEMORY_WORKSPACE to set the resource root; "
        "CODE_MEMORY_ENABLE_VEC/FTS/GRAPH to toggle features. "
        "Optional local summary: CODE_MEMORY_SUMMARY_MODEL (GGUF path). "
        "Auto-install llama-cpp-python via CODE_MEMORY_AUTO_INSTALL=1 (default)."
    ),
)


# ---------------------------
# Embedding and storage layer
# ---------------------------
def _lazy_sqlite():
    global sqlite3
    if sqlite3 is None:
        import sqlite3 as _sqlite3

        sqlite3 = _sqlite3
    return sqlite3


def _apply_pragmas(conn) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA page_size=8192")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA optimize")


def _connect_db(load_vec: bool = False):
    conn = _lazy_sqlite().connect(DB_PATH)
    _apply_pragmas(conn)
    if load_vec and ENABLE_VEC:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    return conn


class EmbeddingModel:
    """Wrapper for fastembed in CPU, with lazy loading."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME, embed_dim: Optional[int] = None):
        os.environ.setdefault("FASTEMBED_CACHE_PATH", str(MODEL_CACHE_DIR))
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
        self.model_name = model_name
        self._model: Optional[TextEmbedding] = None
        self._dim = embed_dim
        if not EMBED_DIM_CONFIGURED and model_name != DEFAULT_EMBED_MODEL:
            logger.warning(
                "CODE_MEMORY_EMBED_DIM not set for model %s; defaulting to %s.",
                model_name,
                self._dim,
            )
        if self._dim is None:
            logger.info("Embedding model configured: %s (cache: %s)", model_name, MODEL_CACHE_DIR)
        else:
            logger.info("Embedding model configured: %s (dim=%s cache: %s)", model_name, self._dim, MODEL_CACHE_DIR)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return int(self._dim or 0)

    def embed(self, text: str) -> List[float]:
        self._load()
        if not self._model:
            raise RuntimeError("Embedding model not available.")
        return list(self._model.embed([text]))[0]

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading embedding model: %s (cache: %s)", self.model_name, MODEL_CACHE_DIR)
        self._model = TextEmbedding(model_name=self.model_name)
        actual_dim = len(next(self._model.embed(["init"])))
        if self._dim is None:
            self._dim = actual_dim
        elif self._dim != actual_dim:
            raise ValueError(
                f"Embedding dim mismatch: configured={self._dim} actual={actual_dim}. "
                "Update CODE_MEMORY_EMBED_DIM or rebuild the DB."
            )
        logger.info("Embedding model ready: dim=%s", self._dim)


def _serialize_vector(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MemoryStore:
    def __init__(self, db_path: Path, embedder: Optional[EmbeddingModel] = None):
        self.db_path = db_path
        self.embedder = embedder or EmbeddingModel(embed_dim=EMBED_DIM)
        self._ensure_schema()
        logger.info("MemoryStore initialized at %s with embedder %s", db_path, EMBED_MODEL_NAME)

    def _ensure_schema(self) -> None:
        conn = _connect_db(load_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC))
        logger.info("Abrindo/checando schema SQLite em %s", self.db_path)
        try:
            # Tabela principal
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    kind TEXT,
                    content TEXT NOT NULL,
                    summary TEXT,
                    tags TEXT,
                    priority INTEGER DEFAULT 3,
                    metadata TEXT,
                    hash TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Vetor
            if ENABLE_VEC:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                        embedding float[{self.embedder.dim}]
                    )
                    """
                )
            # FTS5 para busca exata/fuzzy
            if ENABLE_FTS:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                        content,
                        summary,
                        tags,
                        metadata
                    )
                    """
                )
                conn.execute("DROP TRIGGER IF EXISTS memories_ai")
                conn.execute("DROP TRIGGER IF EXISTS memories_ad")
                conn.execute("DROP TRIGGER IF EXISTS memories_au")
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_fts(rowid, content, summary, tags, metadata)
                        VALUES (new.id, new.content, new.summary, new.tags, new.metadata);
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid = old.id;
                    END;
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                        DELETE FROM memories_fts WHERE rowid = old.id;
                        INSERT INTO memories_fts(rowid, content, summary, tags, metadata)
                        VALUES (new.id, new.content, new.summary, new.tags, new.metadata);
                    END;
                    """
                )
                try:
                    mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                    if mem_count and fts_count == 0:
                        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                except Exception:
                    pass
            # Entidades extraídas (NER/AST)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id INTEGER NOT NULL,
                    entity_type TEXT,
                    name TEXT,
                    source TEXT,
                    path TEXT,
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                )
                """
            )
            # Knowledge graph tables
            if ENABLE_GRAPH:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        entity_type TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_observations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        memory_id INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(entity_id) REFERENCES graph_entities(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS graph_relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id INTEGER NOT NULL,
                        target_id INTEGER NOT NULL,
                        relation_type TEXT NOT NULL,
                        memory_id INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(source_id) REFERENCES graph_entities(id) ON DELETE CASCADE,
                        FOREIGN KEY(target_id) REFERENCES graph_entities(id) ON DELETE CASCADE
                    )
                    """
                )
                if ENABLE_VEC:
                    conn.execute(
                        f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS vec_graph_entities USING vec0(
                            embedding float[{self.embedder.dim}]
                        )
                        """
                    )
            # Índice para hash dedupe
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_session_created ON memories(session_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC)")
            if ENABLE_GRAPH:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_entities_name ON graph_entities(name)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_obs_entity ON graph_observations(entity_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_source ON graph_relations(source_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_target ON graph_relations(target_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_obs_memory ON graph_observations(memory_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_rel_memory ON graph_relations(memory_id)")
            # Ajuste de schema em bancos antigos (add columns)
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3")
            except Exception:
                pass
            conn.commit()
            logger.info("Schema pronto: memories, vec=%s, fts=%s, entities, graph=%s", ENABLE_VEC, ENABLE_FTS, ENABLE_GRAPH)
        finally:
            conn.close()

    def add(
        self,
        content: str,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        summary: Optional[str] = None,
        tags: Optional[str] = None,
        priority: int = 3,
        metadata: Optional[dict] = None,
        ctx: Context | None = None,
    ) -> int:
        if not session_id:
            logger.error("session_id ausente; memoria nao sera salva.")
            return -1
        if _looks_sensitive(content):
            if ctx:
                ctx.warning("Skipping store: sensitive content detected.")
            logger.warning("Conteudo sensivel detectado, ignorando armazenamento.")
            return -1

        # Deduplicação simples: mesmo hash recente não grava novamente
        content_hash = _hash_content(content)
        if self._is_recent_duplicate(content_hash):
            if ctx:
                ctx.info("Skipping store: recent duplicate content.")
            logger.info("Conteudo duplicado recente, ignorando armazenamento.")
            return -1

        embedding = self.embedder.embed(content) if ENABLE_VEC else []
        summary = summary or _auto_summary(content)
        tags = tags or _auto_tags(content)
        prio = _clamp_priority(priority)

        entities = _extract_entities(content)

        conn = _connect_db(load_vec=ENABLE_VEC)
        try:
            cur = conn.execute(
                """
                INSERT INTO memories (session_id, kind, content, summary, tags, priority, metadata, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    content,
                    summary,
                    tags,
                    prio,
                    json.dumps(metadata or {}),
                    content_hash,
                ),
            )
            row_id = cur.lastrowid

            # Vetor
            if ENABLE_VEC:
                conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize_vector(embedding)),
                )
            # Entidades
            if entities:
                conn.executemany(
                    """
                    INSERT INTO entities (memory_id, entity_type, name, source, path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(row_id, e["type"], e["name"], e["source"], e.get("path")) for e in entities],
                )

            conn.commit()
            # Update knowledge graph with extracted entities
            if entities and ENABLE_GRAPH:
                obs_text = summary or content
                for e in entities:
                    self.upsert_graph_entity(
                        name=e["name"],
                        entity_type=e.get("type") or "entity",
                        observations=[obs_text],
                        memory_id=row_id,
                    )
            return row_id
        finally:
            logger.debug("Stored memory row_id=%s session_id=%s priority=%s tags=%s", row_id if 'row_id' in locals() else None, session_id, prio, tags)
            conn.close()

    def _is_recent_duplicate(self, content_hash: str, window_seconds: int = 300) -> bool:
        conn = _connect_db()
        try:
            row = conn.execute(
                """
                SELECT created_at FROM memories
                WHERE hash = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            if not row:
                return False
            ts = row[0]
            try:
                # SQLite DATETIME to epoch heuristic
                last_time = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                return False
            return (time.time() - last_time) < window_seconds
        finally:
            conn.close()

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        top_p: float = DEFAULT_TOP_P,
        use_fts: bool = True,
    ) -> List[dict]:
        q_vec = self.embedder.embed(query) if ENABLE_VEC else None
        effective_k = _effective_k(limit)
        conn = _connect_db(load_vec=ENABLE_VEC)
        try:
            if ENABLE_VEC:
                rows = conn.execute(
                    """
                    SELECT
                        m.id,
                        m.session_id,
                        m.kind,
                        m.content,
                        m.summary,
                        m.tags,
                        m.priority,
                        m.metadata,
                        m.created_at,
                        v.distance
                    FROM vec_memories v
                    JOIN memories m ON v.rowid = m.id
                    WHERE v.embedding MATCH ?
                    AND k = ?
                    AND (? IS NULL OR m.session_id = ?)
                    ORDER BY v.distance
                    LIMIT ?
                    """,
                    (_serialize_vector(q_vec), effective_k, session_id, session_id, effective_k),
                ).fetchall()
            else:
                rows = []
        finally:
            conn.close()

        results = [
            {
                "id": row[0],
                "session_id": row[1],
                "kind": row[2],
                "content": row[3],
                "summary": row[4],
                "tags": row[5],
                "priority": row[6],
                "metadata": json.loads(row[7] or "{}"),
                "created_at": row[8],
                "score": row[9],
                "fts_hit": False,
            }
            for row in rows
        ]

        # FTS re-rank (optional)
        if use_fts and ENABLE_FTS:
            fts_hits = self._fts_lookup(query, session_id, limit)
            by_id = {r["id"]: r for r in results}
            for hit in fts_hits:
                if hit["id"] in by_id:
                    by_id[hit["id"]]["fts_hit"] = True
                else:
                    hit["fts_hit"] = True
                    results.append(hit)

        if not results:
            results = self._recent_filtered(session_id, limit)

        results = _rerank_results(results, top_p)
        results = results[:limit]
        logger.debug(
            "Search query=%s session_id=%s limit=%s hits=%s",
            query,
            session_id,
            limit,
            len(results),
        )
        logger.debug("Search results sample: %s", results[:3])
        return results

    def _fts_lookup(self, query: str, session_id: Optional[str], limit: int) -> List[dict]:
        conn = _connect_db()
        try:
            rows = conn.execute(
                """
                SELECT m.id, m.session_id, m.kind, m.content, m.summary, m.tags, m.priority, m.metadata, m.created_at
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ?
                AND (? IS NULL OR m.session_id = ?)
                LIMIT ?
                """,
                (query, session_id, session_id, limit),
            ).fetchall()
        except Exception as exc:
            logger.warning("FTS query failed for %s: %s", query, exc)
            return []
        finally:
            conn.close()

        return [
            {
                "id": row[0],
                "session_id": row[1],
                "kind": row[2],
                "content": row[3],
                "summary": row[4],
                "tags": row[5],
                "priority": row[6],
                "metadata": json.loads(row[7] or "{}"),
                "created_at": row[8],
                "score": 0.0,
                "fts_hit": True,
            }
            for row in rows
        ]

    def recent(self, limit: int = 20) -> List[dict]:
        conn = _connect_db()
        try:
            rows = conn.execute(
                """
                SELECT id, session_id, kind, content, summary, tags, priority, metadata, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "id": row[0],
                "session_id": row[1],
                "kind": row[2],
                "content": row[3],
                "summary": row[4],
                "tags": row[5],
                "priority": row[6],
                "metadata": json.loads(row[7] or "{}"),
                "created_at": row[8],
            }
            for row in rows
        ]

    def _recent_filtered(
        self,
        session_id: Optional[str],
        limit: int,
    ) -> List[dict]:
        conn = _connect_db()
        try:
            rows = conn.execute(
                """
                SELECT id, session_id, kind, content, summary, tags, priority, metadata, created_at
                FROM memories
                WHERE (? IS NULL OR session_id = ?)
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, session_id, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "kind": row[2],
                "content": row[3],
                "summary": row[4],
                "tags": row[5],
                "priority": row[6],
                "metadata": json.loads(row[7] or "{}"),
                "created_at": row[8],
                "score": 1.0,
                "fts_hit": False,
            }
            for row in rows
        ]

    def list_entities(self, memory_id: int) -> List[dict]:
        conn = _connect_db()
        try:
            rows = conn.execute(
                """
                SELECT entity_type, name, source, path
                FROM entities
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchall()
        finally:
            conn.close()
        return [
            {"entity_type": row[0], "name": row[1], "source": row[2], "path": row[3]}
            for row in rows
        ]

    def upsert_graph_entity(
        self,
        name: str,
        entity_type: str,
        observations: List[str],
        memory_id: Optional[int] = None,
    ) -> int:
        if not ENABLE_GRAPH:
            return -1
        if not name:
            return -1
        conn = _connect_db(load_vec=ENABLE_VEC)
        try:
            row = conn.execute(
                "SELECT id FROM graph_entities WHERE name = ?",
                (name,),
            ).fetchone()
            if row:
                entity_id = row[0]
                conn.execute(
                    "UPDATE graph_entities SET entity_type = ? WHERE id = ?",
                    (entity_type, entity_id),
                )
            else:
                cur = conn.execute(
                    "INSERT INTO graph_entities (name, entity_type) VALUES (?, ?)",
                    (name, entity_type),
                )
                entity_id = cur.lastrowid

            for obs in observations:
                conn.execute(
                    """
                    INSERT INTO graph_observations (entity_id, content, memory_id)
                    VALUES (?, ?, ?)
                    """,
                    (entity_id, obs, memory_id),
                )

            embed_text = f"{name} " + " ".join(observations)
            if ENABLE_VEC:
                embedding = self.embedder.embed(embed_text)
                conn.execute("DELETE FROM vec_graph_entities WHERE rowid = ?", (entity_id,))
                conn.execute(
                    "INSERT INTO vec_graph_entities(rowid, embedding) VALUES (?, ?)",
                    (entity_id, _serialize_vector(embedding)),
                )
            conn.commit()
            return entity_id
        finally:
            conn.close()

    def add_graph_relation(
        self,
        source: str,
        target: str,
        relation_type: str,
        memory_id: Optional[int] = None,
    ) -> int:
        if not source or not target or not relation_type:
            return -1
        source_id = self.upsert_graph_entity(source, "entity", [], memory_id)
        target_id = self.upsert_graph_entity(target, "entity", [], memory_id)
        conn = _connect_db()
        try:
            cur = conn.execute(
                """
                INSERT INTO graph_relations (source_id, target_id, relation_type, memory_id)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, relation_type, memory_id),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def get_graph_entity(self, name: str) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        conn = _connect_db()
        try:
            row = conn.execute(
                "SELECT id, name, entity_type, created_at FROM graph_entities WHERE name = ?",
                (name,),
            ).fetchone()
            if not row:
                return {"status": "not_found"}
            entity_id = row[0]
            obs = conn.execute(
                "SELECT content FROM graph_observations WHERE entity_id = ? ORDER BY id DESC LIMIT 50",
                (entity_id,),
            ).fetchall()
            rel = conn.execute(
                """
                SELECT r.relation_type, s.name, t.name
                FROM graph_relations r
                JOIN graph_entities s ON r.source_id = s.id
                JOIN graph_entities t ON r.target_id = t.id
                WHERE r.source_id = ? OR r.target_id = ?
                """,
                (entity_id, entity_id),
            ).fetchall()
            return {
                "id": entity_id,
                "name": row[1],
                "entity_type": row[2],
                "created_at": row[3],
                "observations": [o[0] for o in obs],
                "relations": [
                    {"type": r[0], "source": r[1], "target": r[2]} for r in rel
                ],
            }
        finally:
            conn.close()

    def read_graph(self, limit: int = 10) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        conn = _connect_db()
        try:
            entities = conn.execute(
                """
                SELECT id, name, entity_type, created_at
                FROM graph_entities
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            entity_ids = [e[0] for e in entities]
            relations = []
            if entity_ids:
                placeholders = ",".join("?" for _ in entity_ids)
                relations = conn.execute(
                    f"""
                    SELECT r.relation_type, s.name, t.name
                    FROM graph_relations r
                    JOIN graph_entities s ON r.source_id = s.id
                    JOIN graph_entities t ON r.target_id = t.id
                    WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                    """,
                    (*entity_ids, *entity_ids),
                ).fetchall()
            return {
                "entities": [
                    {"id": e[0], "name": e[1], "entity_type": e[2], "created_at": e[3]}
                    for e in entities
                ],
                "relations": [
                    {"type": r[0], "source": r[1], "target": r[2]} for r in relations
                ],
            }
        finally:
            conn.close()

    def search_graph(self, query: str, limit: int = 5) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}

        rows = []
        if ENABLE_VEC:
            conn = _connect_db(load_vec=True)
            try:
                q_vec = self.embedder.embed(query)
                rows = conn.execute(
                    """
                    SELECT g.id, g.name, g.entity_type, g.created_at, v.distance
                    FROM vec_graph_entities v
                    JOIN graph_entities g ON v.rowid = g.id
                    WHERE v.embedding MATCH ?
                    AND k = ?
                    ORDER BY v.distance
                    LIMIT ?
                    """,
                    (_serialize_vector(q_vec), limit, limit),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            conn = _connect_db()
            try:
                rows = conn.execute(
                    """
                    SELECT id, name, entity_type, created_at, 0.0
                    FROM graph_entities
                    WHERE name LIKE ?
                    LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()
            finally:
                conn.close()

        entity_ids = [r[0] for r in rows]
        rel = []
        if entity_ids:
            conn = _connect_db()
            try:
                placeholders = ",".join("?" for _ in entity_ids)
                rel = conn.execute(
                    f"""
                    SELECT r.relation_type, s.name, t.name
                    FROM graph_relations r
                    JOIN graph_entities s ON r.source_id = s.id
                    JOIN graph_entities t ON r.target_id = t.id
                    WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                    """,
                    (*entity_ids, *entity_ids),
                ).fetchall()
            finally:
                conn.close()

        return {
            "entities": [
                {
                    "id": r[0],
                    "name": r[1],
                    "entity_type": r[2],
                    "created_at": r[3],
                    "score": r[4],
                }
                for r in rows
            ],
            "relations": [
                {"type": r[0], "source": r[1], "target": r[2]} for r in rel
            ],
        }


SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*[\\\"']?[A-Za-z0-9\\-_/]{16,})"),
    re.compile(r"(?i)(secret\s*[:=]\s*[\\\"']?[A-Za-z0-9\\-_/]{12,})"),
    re.compile(r"(?i)(password\s*[:=])"),
]


def _looks_sensitive(text: str) -> bool:
    return any(p.search(text) for p in SENSITIVE_PATTERNS)


def _auto_summary(content: str, max_len: int = 240) -> str:
    summary = _generate_summary(content)
    if summary:
        return summary[:max_len]
    first = content.strip().splitlines()[0] if content.strip() else ""
    if not first:
        return content[:max_len]
    if len(first) > max_len:
        return first[: max_len - 3] + "..."
    return first


def _auto_tags(content: str) -> str:
    keywords = ["bug", "error", "fix", "database", "api", "frontend", "backend", "ui", "test", "deploy"]
    lower = content.lower()
    found = [k for k in keywords if k in lower]
    return ",".join(found) if found else "general"


def _clamp_priority(value: int, min_v: int = 1, max_v: int = 5) -> int:
    try:
        return max(min_v, min(max_v, int(value)))
    except Exception:
        return 3


def _summary_enabled() -> bool:
    if not SUMMARY_MODEL:
        return False
    try:
        if not Path(SUMMARY_MODEL).exists():
            return False
    except Exception:
        return False
    return _ensure_llama_cpp()


def _summary_prompt() -> str:
    if SUMMARY_PROMPT:
        return SUMMARY_PROMPT
    return (
        "You are a concise summarizer for developer activity. "
        "Summarize the content into 1-3 sentences. "
        "Return plain text only."
    )


def _generate_summary(content: str) -> str:
    if not _summary_enabled():
        return ""
    try:
        llm = _get_summary_llm()
        if not llm:
            return ""
        prompt = _summary_prompt()
        full_prompt = f"{prompt}\n\nContent:\n{content}\n\nSummary:"
        logger.info("Resumo local via GGUF: %s", SUMMARY_MODEL)
        output = llm(
            full_prompt,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=SUMMARY_TEMPERATURE,
            top_p=SUMMARY_TOP_P,
            repeat_penalty=SUMMARY_REPEAT_PENALTY,
            stop=["\n\n"],
        )
        summary = output["choices"][0]["text"] if output and "choices" in output else ""
        return summary.strip()[:SUMMARY_MAX_CHARS]
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha ao gerar resumo local: %s", exc)
        return ""


def _get_summary_llm():
    global _SUMMARY_LLM
    if not _summary_enabled():
        return None
    if _SUMMARY_LLM is not None:
        return _SUMMARY_LLM
    try:
        logger.info("Carregando GGUF local: %s", SUMMARY_MODEL)
        _SUMMARY_LLM = Llama(
            model_path=SUMMARY_MODEL,
            n_ctx=SUMMARY_CTX,
            n_threads=SUMMARY_THREADS,
            n_gpu_layers=SUMMARY_N_GPU_LAYERS,
            verbose=False,
        )
        return _SUMMARY_LLM
    except Exception as exc:  # pragma: no cover
        logger.warning("Falha ao carregar GGUF local: %s", exc)
        return None


def _clamp_top_k(value: int) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return max(1, DEFAULT_TOP_K)


def _clamp_top_p(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return DEFAULT_TOP_P
    if v <= 0:
        return 1.0
    return min(1.0, v)


def _effective_k(limit: int) -> int:
    try:
        mult = max(1, int(OVERSAMPLE_K))
    except Exception:
        mult = 4
    return max(1, int(limit) * mult)


def _parse_timestamp(value: str) -> float:
    try:
        return time.mktime(time.strptime(value, "%Y-%m-%d %H:%M:%S"))
    except Exception:
        return 0.0


def _apply_recency_filter(results: List[dict], top_p: float) -> List[dict]:
    if not results:
        return results
    top_p = _clamp_top_p(top_p)
    if top_p >= 1.0:
        return results
    sorted_by_time = sorted(results, key=lambda r: _parse_timestamp(r.get("created_at", "")), reverse=True)
    keep = max(1, int(math.ceil(len(sorted_by_time) * top_p)))
    keep_ids = {r["id"] for r in sorted_by_time[:keep]}
    return [r for r in results if r["id"] in keep_ids]


def _resolve_session_id(session_id: Optional[str], ctx: Context | None) -> Optional[str]:
    if session_id:
        return session_id
    if ctx and getattr(ctx, "session_id", None):
        return ctx.session_id
    env_session = os.getenv("CODE_MEMORY_SESSION_ID")
    return env_session or None


def _rerank_results(results: List[dict], top_p: float) -> List[dict]:
    if not results:
        return results
    # Normalize recency
    times = [_parse_timestamp(r.get("created_at", "")) for r in results]
    now = max(times) if times else time.time()
    max_age = max((now - t) for t in times) if times else 0.0

    for r in results:
        base = r.get("score", 1.0) or 1.0
        if r.get("fts_hit"):
            base -= FTS_BONUS
        priority = r.get("priority") or 3
        base += PRIORITY_WEIGHT * (priority - 1)
        age = now - _parse_timestamp(r.get("created_at", ""))
        if max_age > 0:
            base += RECENCY_WEIGHT * (age / max_age)
        r["score"] = base

    results = _apply_recency_filter(results, top_p)
    results.sort(key=lambda x: x["score"])
    return results


def _extract_entities(content: str) -> List[dict]:
    entities: List[dict] = []
    # Heurística simples via regex
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\s*\(", content):
        name = match.group(1)
        entities.append({"type": "function_like", "name": name, "source": "regex", "path": None})
    # Tree-sitter opcional para nomes mais precisos (quando disponível)
    if TREE_SITTER_AVAILABLE:
        try:
            tree = TREE_SITTER_PARSER.parse(bytes(content, "utf-8"))
            root = tree.root_node
            for node in root.walk():
                pass  # placeholder to allow iteration
            # Coleta nomes de funções/classe simples
            entities.extend(_walk_tree_for_entities(tree, content))
        except Exception:
            pass
    return entities


def _walk_tree_for_entities(tree, source: str) -> List[dict]:
    # Focado em Python; extensível para outras linguagens se necessário.
    names = []
    cursor = tree.walk()
    visited = set()
    while True:
        node = cursor.node
        if node.id in visited:
            pass
        else:
            visited.add(node.id)
            if node.type in ("function_definition", "class_definition"):
                # child structure: (def|class) <name>
                for child in node.children:
                    if child.type == "identifier":
                        name = source[child.start_byte : child.end_byte]
                        names.append(
                            {
                                "type": "function" if node.type == "function_definition" else "class",
                                "name": name,
                                "source": "tree-sitter",
                                "path": None,
                            }
                        )
                        break
        if cursor.goto_first_child():
            continue
        while not cursor.goto_next_sibling():
            if not cursor.goto_parent():
                return names


store = MemoryStore(DB_PATH)


# -------------
# MCP tools
# -------------
@server.tool(
    description="Store memory with vector + FTS + entities. session_id is required. Optional: kind, summary, tags, priority, metadata_json."
)
def remember(
    content: str,
    session_id: Optional[str] = None,
    kind: Optional[str] = None,
    summary: Optional[str] = None,
    tags: Optional[str] = None,
    priority: int = 3,
    metadata_json: Optional[str] = None,
    ctx: Context | None = None,
) -> dict:
    try:
        resolved_session_id = _resolve_session_id(session_id, ctx)
        if not resolved_session_id:
            return {"status": "error", "error": "session_id is required", "tool": "remember"}
        metadata = json.loads(metadata_json) if metadata_json else None
        mem_id = store.add(
            content=content,
            session_id=resolved_session_id,
            kind=kind,
            summary=summary,
            tags=tags,
            priority=priority,
            metadata=metadata,
            ctx=ctx,
        )
        if mem_id == -1:
            return {"status": "skipped"}
        if ctx:
            ctx.info(f"Stored memory {mem_id}")
        logger.info("remember ok id=%s session_id=%s priority=%s", mem_id, resolved_session_id, priority)
        logger.debug("remember payload summary=%s tags=%s kind=%s metadata=%s", summary, tags, kind, metadata)
        return {
            "id": mem_id,
            "summary": summary,
            "session_id": resolved_session_id,
            "kind": kind,
            "tags": tags,
            "priority": priority,
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool remember falhou")
        return {"status": "error", "error": str(exc), "tool": "remember"}


@server.tool(description="Semantic + vector search over stored memories. Also re-ranks with FTS.")
def search_memory(
    query: str,
    session_id: Optional[str] = None,
    limit: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    ctx: Context | None = None,
) -> List[dict]:
    try:
        resolved_session_id = _resolve_session_id(session_id, ctx)
        if not resolved_session_id:
            return {"status": "error", "error": "session_id is required", "tool": "search_memory"}
        limit = _clamp_top_k(limit)
        top_p = _clamp_top_p(top_p)
        return store.search(
            query=query,
            session_id=resolved_session_id,
            limit=limit,
            top_p=top_p,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool search_memory falhou")
        return {"status": "error", "error": str(exc), "tool": "search_memory"}


@server.tool(description="List most recent memory entries for debugging or inspection.")
def list_recent(limit: int = 20) -> List[dict]:
    try:
        return store.recent(limit=limit)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_recent falhou")
        return {"status": "error", "error": str(exc), "tool": "list_recent"}


@server.tool(description="List entities extracted for a given memory id.")
def list_entities(memory_id: int) -> List[dict]:
    try:
        return store.list_entities(memory_id)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_entities falhou")
        return {"status": "error", "error": str(exc), "tool": "list_entities"}


@server.tool(description="Upsert a graph entity with observations (JSON list).")
def upsert_entity(
    name: str,
    entity_type: str = "entity",
    observations_json: Optional[str] = None,
    memory_id: Optional[int] = None,
) -> dict:
    try:
        observations = json.loads(observations_json) if observations_json else []
        if not isinstance(observations, list):
            observations = [str(observations)]
        entity_id = store.upsert_graph_entity(name, entity_type, observations, memory_id)
        return {"status": "ok", "entity_id": entity_id}
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool upsert_entity falhou")
        return {"status": "error", "error": str(exc), "tool": "upsert_entity"}


@server.tool(description="Add a relation between two entities in the graph.")
def add_relation(
    source: str,
    target: str,
    relation_type: str,
    memory_id: Optional[int] = None,
) -> dict:
    try:
        rel_id = store.add_graph_relation(source, target, relation_type, memory_id)
        return {"status": "ok", "relation_id": rel_id}
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool add_relation falhou")
        return {"status": "error", "error": str(exc), "tool": "add_relation"}


@server.tool(description="Get a graph entity with observations and relations.")
def get_entity(name: str) -> dict:
    try:
        return store.get_graph_entity(name)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_entity falhou")
        return {"status": "error", "error": str(exc), "tool": "get_entity"}


@server.tool(description="Return a context graph (semantic if query is provided).")
def get_context_graph(query: Optional[str] = None, limit: int = 10) -> dict:
    try:
        if query:
            return store.search_graph(query, limit=limit)
        return store.read_graph(limit=limit)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_context_graph falhou")
        return {"status": "error", "error": str(exc), "tool": "get_context_graph"}


@server.tool(description="Manual maintenance: vacuum/prune/purge. Never runs automatically.")
def maintenance(
    action: str,
    confirm: bool = False,
    session_id: Optional[str] = None,
    older_than_days: Optional[int] = None,
) -> dict:
    try:
        if action != "vacuum" and not confirm:
            return {"status": "error", "error": "confirm=true required for destructive actions"}
        conn = _connect_db(load_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC))
        try:
            if action == "vacuum":
                conn.execute("VACUUM")
                return {"status": "ok", "action": "vacuum"}

            ids = []
            if action == "purge_all":
                ids = [row[0] for row in conn.execute("SELECT id FROM memories").fetchall()]
            elif action == "purge_session" and session_id:
                ids = [
                    row[0]
                    for row in conn.execute("SELECT id FROM memories WHERE session_id = ?", (session_id,)).fetchall()
                ]
            elif action == "prune_older_than" and older_than_days is not None:
                cutoff = time.time() - (older_than_days * 86400)
                rows = conn.execute(
                    "SELECT id, created_at FROM memories"
                ).fetchall()
                for mid, created_at in rows:
                    ts = _parse_timestamp(created_at)
                    if ts and ts < cutoff:
                        ids.append(mid)
            else:
                return {"status": "error", "error": "invalid action or missing parameter"}

            if not ids:
                return {"status": "ok", "deleted": 0}

            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM entities WHERE memory_id IN ({placeholders})", ids)
            if ENABLE_VEC:
                conn.execute(f"DELETE FROM vec_memories WHERE rowid IN ({placeholders})", ids)
            if ENABLE_GRAPH:
                conn.execute(f"DELETE FROM graph_observations WHERE memory_id IN ({placeholders})", ids)
                conn.execute(f"DELETE FROM graph_relations WHERE memory_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            conn.commit()
            return {"status": "ok", "deleted": len(ids)}
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool maintenance falhou")
        return {"status": "error", "error": str(exc), "tool": "maintenance"}


@server.tool(description="Diagnostics for environment, DB, and feature flags.")
def diagnostics() -> dict:
    try:
        conn = _connect_db(load_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC))
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            counts = {}
            for t in ("memories", "entities", "graph_entities", "graph_observations", "graph_relations"):
                if t in tables:
                    counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        finally:
            conn.close()
        return {
            "cwd": str(Path.cwd()),
            "db_path": str(DB_PATH),
            "model": EMBED_MODEL_NAME,
            "model_cache_dir": str(MODEL_CACHE_DIR),
            "flags": {"vec": ENABLE_VEC, "fts": ENABLE_FTS, "graph": ENABLE_GRAPH},
            "defaults": {
                "top_k": DEFAULT_TOP_K,
                "top_p": DEFAULT_TOP_P,
                "recency_weight": RECENCY_WEIGHT,
                "priority_weight": PRIORITY_WEIGHT,
                "fts_bonus": FTS_BONUS,
                "oversample_k": OVERSAMPLE_K,
            },
            "summary": {
                "enabled": _summary_enabled(),
                "model": SUMMARY_MODEL,
                "ctx": SUMMARY_CTX,
                "threads": SUMMARY_THREADS,
                "max_tokens": SUMMARY_MAX_TOKENS,
                "temperature": SUMMARY_TEMPERATURE,
                "auto_install": SUMMARY_AUTO_INSTALL,
            },
            "tables": tables,
            "counts": counts,
        }
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool diagnostics falhou")
        return {"status": "error", "error": str(exc), "tool": "diagnostics"}


@server.tool(description="Health check for the code-memory server.")
def health() -> dict:
    try:
        info = {
            "status": "ok",
            "db_path": str(DB_PATH),
            "embedding_dim": store.embedder.dim,
            "model": EMBED_MODEL_NAME,
            "model_cache_dir": str(MODEL_CACHE_DIR),
            "tree_sitter": TREE_SITTER_AVAILABLE,
            "default_top_k": DEFAULT_TOP_K,
            "default_top_p": DEFAULT_TOP_P,
            "recency_weight": RECENCY_WEIGHT,
            "priority_weight": PRIORITY_WEIGHT,
            "fts_bonus": FTS_BONUS,
            "oversample_k": OVERSAMPLE_K,
            "flags": {"vec": ENABLE_VEC, "fts": ENABLE_FTS, "graph": ENABLE_GRAPH},
            "summary": {
                "enabled": _summary_enabled(),
                "model": SUMMARY_MODEL,
                "ctx": SUMMARY_CTX,
                "threads": SUMMARY_THREADS,
                "max_tokens": SUMMARY_MAX_TOKENS,
                "temperature": SUMMARY_TEMPERATURE,
                "auto_install": SUMMARY_AUTO_INSTALL,
            },
            "log_level": LOG_LEVEL,
            "log_file": str(LOG_FILE) if LOG_FILE else None,
        }
        logger.info("health check: %s", info)
        return info
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool health falhou")
        return {"status": "error", "error": str(exc), "tool": "health"}


# -------------
# MCP resources
# -------------
@server.resource(
    "resource://workspace",
    description="Browse the project workspace (read-only).",
)
def workspace_resource() -> DirectoryResource:
    return DirectoryResource(
        uri="file://workspace",
        name="workspace",
        path=ROOT,
        recursive=True,
    )


@server.resource("resource://readme", description="Project README if present.")
def readme_resource() -> FileResource:
    path = ROOT / "README.md"
    if not path.exists():
        raise FileNotFoundError("README.md not found in workspace.")
    return FileResource(uri="file://readme", name="README", path=path)


def main() -> None:
    try:
        server.run()
    except Exception as exc:
        logger.exception("Falha ao iniciar/rodar o servidor: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
