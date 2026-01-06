from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from fastmcp import Context

from .config import (
    DB_AUTH_TOKEN,
    DB_BACKEND,
    DB_URL,
    DEFAULT_TOP_P,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    logger,
)
from .db import connect_db
from .embeddings import EmbeddingModel
from .entities import extract_graph
from .rerank import effective_k, rerank_results
from .security import hash_content, looks_sensitive

SCHEMA_VERSION = 3


def _merge_origin(existing: str | None, incoming: str | None) -> str:
    e = str(existing or "").strip().lower()
    i = str(incoming or "").strip().lower()
    if i == "agent":
        return "agent"
    if e:
        return e
    return i or "memory"


def clamp_priority(value: int, min_v: int = 1, max_v: int = 5) -> int:
    try:
        return max(min_v, min(max_v, int(value)))
    except Exception:
        return 3


def auto_tags(content: str) -> str:
    keywords = ["bug", "error", "fix", "database", "api", "frontend", "backend", "ui", "test", "deploy"]
    lower = (content or "").lower()
    found = [k for k in keywords if k in lower]
    return ",".join(found) if found else "general"


class MemoryStore:
    """
    Unified schema: entities / observations / relations (+ optional observations_fts and vector index).

    - Each `remember()` creates a "memory node" in `entities` (entity_type='memory') and stores the raw text in
      `observations` linked to that node.
    - Extracted world-fact entities are stored in `entities` and linked via `relations` (type='mentions').
    - Additional extracted relations connect entities and point back to the observation_id.
    """

    def __init__(self, db_url: str = DB_URL, embedder: Optional[EmbeddingModel] = None):
        self.db_url = db_url
        self.embedder = embedder or EmbeddingModel(embed_dim=EMBED_DIM)
        self._ensure_schema()
        logger.info("MemoryStore initialized at %s with embedder %s", db_url, EMBED_MODEL_NAME)

    def _db(self, *, load_vec: bool = False):
        return connect_db(
            enable_vec=True,
            load_vec=load_vec,
            db_url=self.db_url,
            db_auth_token=DB_AUTH_TOKEN or None,
        )

    def _execute(self, conn, sql: str, params: Optional[tuple | list] = None):
        if DB_BACKEND == "sqlite":
            return conn.execute(sql, params or ())
        return conn.execute(sql, list(params or []))

    def _fetchone(self, conn, sql: str, params: Optional[tuple | list] = None):
        res = self._execute(conn, sql, params)
        if DB_BACKEND == "sqlite":
            return res.fetchone()
        return res.rows[0] if res.rows else None

    def _fetchall(self, conn, sql: str, params: Optional[tuple | list] = None):
        res = self._execute(conn, sql, params)
        if DB_BACKEND == "sqlite":
            return res.fetchall()
        return res.rows

    def _lastrowid(self, result) -> int:
        if DB_BACKEND == "sqlite":
            return int(result.lastrowid)
        return int(getattr(result, "last_insert_rowid", 0) or 0)

    def _has_function(self, conn, name: str) -> bool:
        try:
            # Works on modern SQLite builds and libSQL.
            row = self._fetchone(conn, "SELECT 1 FROM pragma_function_list WHERE name = ? LIMIT 1", (name,))
            return bool(row)
        except Exception:
            # Fall back to optimistic execution (we'll catch failures where used).
            return False

    def _ensure_schema(self) -> None:
        conn = self._db(load_vec=True)
        logger.info("Opening/checking DB schema at %s (backend=%s)", self.db_url, DB_BACKEND)
        try:
            try:
                uv = self._fetchone(conn, "PRAGMA user_version")
                user_version = int(uv[0]) if uv else 0
            except Exception:
                user_version = 0

            if user_version != SCHEMA_VERSION:
                logger.warning(
                    "Schema version mismatch (have=%s want=%s). Resetting DB schema (data will be lost).",
                    user_version,
                    SCHEMA_VERSION,
                )
                self._execute(conn, "DROP TRIGGER IF EXISTS observations_ai")
                self._execute(conn, "DROP TRIGGER IF EXISTS observations_ad")
                self._execute(conn, "DROP TRIGGER IF EXISTS observations_au")
                self._execute(conn, "DROP TABLE IF EXISTS observations_fts")
                self._execute(conn, "DROP TABLE IF EXISTS relations")
                self._execute(conn, "DROP TABLE IF EXISTS observations")
                self._execute(conn, "DROP TABLE IF EXISTS entities")
                self._execute(conn, f"PRAGMA user_version = {SCHEMA_VERSION}")

            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    origin TEXT NOT NULL DEFAULT 'memory',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id INTEGER NOT NULL,
                    session_id TEXT,
                    kind TEXT,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    tags TEXT,
                    priority INTEGER DEFAULT 3,
                    metadata TEXT,
                    embedding BLOB,
                    embedding_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
                )
                """
            )
            self._execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    observation_id INTEGER,
                    origin TEXT NOT NULL DEFAULT 'memory',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(source_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES entities(id) ON DELETE CASCADE,
                    FOREIGN KEY(observation_id) REFERENCES observations(id) ON DELETE SET NULL
                )
                """
            )

            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_entities_origin ON entities(origin)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_obs_entity ON observations(entity_id)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_obs_session ON observations(session_id)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_obs_hash ON observations(content_hash)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_obs_created ON observations(created_at DESC)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_rel_obs ON relations(observation_id)")
            self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_rel_origin ON relations(origin)")
            # Dedupe within a session: one row per content hash (while allowing session_id NULL for graph-only notes).
            self._execute(
                conn,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_obs_session_hash ON observations(session_id, content_hash) WHERE session_id IS NOT NULL",
            )
            # Dedupe relations (ignore origin differences).
            self._execute(
                conn,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_dedupe ON relations(source_id, target_id, relation_type, IFNULL(observation_id, -1))",
            )

            # Vector index (best-effort): only available on libSQL servers with vector enabled.
            if self._has_function(conn, "libsql_vector_idx"):
                try:
                    self._execute(conn, "CREATE INDEX IF NOT EXISTS idx_obs_embedding ON observations(libsql_vector_idx(embedding))")
                except Exception as exc:
                    logger.warning("Failed to create libsql_vector_idx index: %s", exc)

            self._execute(
                conn,
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
                    content,
                    tags,
                    metadata,
                    content='observations',
                    content_rowid='id'
                )
                """
            )
            self._execute(conn, "DROP TRIGGER IF EXISTS observations_ai")
            self._execute(conn, "DROP TRIGGER IF EXISTS observations_ad")
            self._execute(conn, "DROP TRIGGER IF EXISTS observations_au")
            self._execute(
                conn,
                """
                CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
                    INSERT INTO observations_fts(rowid, content, tags, metadata)
                    VALUES (new.id, new.content, new.tags, new.metadata);
                END;
                """
            )
            self._execute(
                conn,
                """
                CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
                    DELETE FROM observations_fts WHERE rowid = old.id;
                END;
                """
            )
            self._execute(
                conn,
                """
                CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE ON observations BEGIN
                    DELETE FROM observations_fts WHERE rowid = old.id;
                    INSERT INTO observations_fts(rowid, content, tags, metadata)
                    VALUES (new.id, new.content, new.tags, new.metadata);
                END;
                """
            )

            if DB_BACKEND == "sqlite":
                conn.commit()
            logger.info("Schema ready: vec=on fts=on graph=on")
        finally:
            conn.close()

    def _upsert_entity(self, conn, *, name: str, entity_type: str, origin: str = "memory") -> int:
        if not name:
            return -1
        row = self._fetchone(conn, "SELECT id, entity_type, origin FROM entities WHERE name = ?", (name,))
        if row:
            entity_id = int(row[0])
            existing_entity_type = str(row[1] or "").strip()
            existing_origin = str(row[2] or "").strip()
            desired_type = entity_type or existing_entity_type or "other"
            desired_origin = _merge_origin(existing_origin, origin)
            if desired_type != existing_entity_type or desired_origin != existing_origin:
                self._execute(
                    conn,
                    "UPDATE entities SET entity_type = ?, origin = ? WHERE id = ?",
                    (desired_type, desired_origin, entity_id),
                )
            return entity_id
        cur = self._execute(
            conn,
            "INSERT INTO entities (name, entity_type, origin) VALUES (?, ?, ?)",
            (name, entity_type or "other", origin or "memory"),
        )
        return self._lastrowid(cur)

    def _insert_relation(
        self,
        conn,
        *,
        source_id: int,
        target_id: int,
        relation_type: str,
        observation_id: Optional[int],
        origin: str = "memory",
    ) -> None:
        if source_id <= 0 or target_id <= 0 or not relation_type:
            return
        try:
            exists = self._fetchone(
                conn,
                """
                SELECT 1 FROM relations
                WHERE source_id = ? AND target_id = ? AND relation_type = ?
                  AND (observation_id IS ? OR observation_id = ?)
                LIMIT 1
                """,
                (source_id, target_id, relation_type, observation_id, observation_id),
            )
            if exists:
                try:
                    row = self._fetchone(
                        conn,
                        """
                        SELECT id, origin
                        FROM relations
                        WHERE source_id = ? AND target_id = ? AND relation_type = ?
                          AND (observation_id IS ? OR observation_id = ?)
                        LIMIT 1
                        """,
                        (source_id, target_id, relation_type, observation_id, observation_id),
                    )
                    if row:
                        rel_id = int(row[0])
                        existing_origin = str(row[1] or "").strip()
                        desired_origin = _merge_origin(existing_origin, origin)
                        if desired_origin != existing_origin:
                            self._execute(conn, "UPDATE relations SET origin = ? WHERE id = ?", (desired_origin, rel_id))
                except Exception:
                    pass
                return
        except Exception:
            pass
        self._execute(
            conn,
            """
            INSERT OR IGNORE INTO relations (source_id, target_id, relation_type, observation_id, origin)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, target_id, relation_type, observation_id, origin or "memory"),
        )

    def insert_memory(
        self,
        *,
        content: str,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[str] = None,
        priority: int = 3,
        metadata: Optional[dict] = None,
        ctx: Context | None = None,
    ) -> int:
        if not session_id:
            logger.error("Missing session_id; memory will not be stored.")
            return -1
        if looks_sensitive(content):
            if ctx:
                ctx.warning("Skipping store: sensitive content detected.")
            logger.warning("Sensitive content detected; skipping store.")
            return -1

        content_hash = hash_content(content)
        memory_node_name = f"mem:{session_id}:{content_hash[:12]}"

        tags_value = str(tags or "").strip()
        prio = clamp_priority(priority)
        meta = metadata or {}

        conn = self._db(load_vec=True)
        try:
            # Dedupe: if this exact content hash already exists in the same session, skip.
            try:
                exists = self._fetchone(
                    conn,
                    "SELECT 1 FROM observations WHERE session_id = ? AND content_hash = ? LIMIT 1",
                    (session_id, content_hash),
                )
                if exists:
                    return -1
            except Exception:
                pass

            memory_entity_id = self._upsert_entity(conn, name=memory_node_name, entity_type="memory", origin="memory")
            if memory_entity_id <= 0:
                return -1

            cur = self._execute(
                conn,
                """
                INSERT INTO observations (entity_id, session_id, kind, content, content_hash, tags, priority, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_entity_id,
                    session_id,
                    kind,
                    content,
                    content_hash,
                    tags_value or None,
                    prio,
                    json.dumps(meta),
                ),
            )
            observation_id = self._lastrowid(cur)

            emb_text = f"{tags_value}\n\n{content}".strip() if tags_value else str(content or "").strip()
            embedding = self.embedder.embed(emb_text)
            # Always store JSON for portable fallback search.
            try:
                self._execute(
                    conn,
                    "UPDATE observations SET embedding_json = ? WHERE id = ?",
                    (json.dumps(list(map(float, embedding))), observation_id),
                )
            except Exception:
                pass

            # Best-effort native vector (libSQL vector servers)
            try:
                conn.execute(
                    "UPDATE observations SET embedding = vector32(?) WHERE id = ?",
                    (json.dumps(list(map(float, embedding))), observation_id),
                )
            except Exception:
                pass

            if DB_BACKEND == "sqlite":
                conn.commit()
            return observation_id
        finally:
            conn.close()

    # Backwards-compatible alias
    def add(
        self,
        *,
        content: str,
        session_id: Optional[str] = None,
        kind: Optional[str] = None,
        tags: Optional[str] = None,
        priority: int = 3,
        metadata: Optional[dict] = None,
        ctx: Context | None = None,
    ) -> int:
        return self.insert_memory(
            content=content,
            session_id=session_id,
            kind=kind,
            tags=tags,
            priority=priority,
            metadata=metadata,
            ctx=ctx,
        )

    def complete_memory_fields(self, observation_id: int, *, content: str) -> None:
        if observation_id <= 0:
            return
        conn = self._db()
        try:
            row = self._fetchone(conn, "SELECT tags FROM observations WHERE id = ?", (observation_id,))
            if not row:
                return
            existing_tags = str(row[0] or "").strip()
            if existing_tags:
                return
            tags_value = auto_tags(content)
            if tags_value:
                self._execute(conn, "UPDATE observations SET tags = ? WHERE id = ?", (tags_value, observation_id))
        finally:
            conn.close()

    def process_memory_ner(self, observation_id: int, *, content: str, metadata: Optional[dict] = None) -> None:
        if observation_id <= 0:
            return
        meta = metadata or {}
        conn = self._db(load_vec=False)
        try:
            row = self._fetchone(conn, "SELECT entity_id FROM observations WHERE id = ?", (observation_id,))
            if not row:
                return
            memory_entity_id = int(row[0])

            graph = extract_graph(content, path=str(meta.get("path") or "") or None, use_llm=True)

            seen_mentions: set[tuple[int, int]] = set()
            for e in graph.get("entities", []) or []:
                name = str(e.get("name") or "").strip()
                et = str(e.get("type") or "other").strip()
                if not name:
                    continue
                ent_id = self._upsert_entity(conn, name=name, entity_type=et or "other", origin="memory")
                if ent_id > 0 and (memory_entity_id, ent_id) not in seen_mentions:
                    seen_mentions.add((memory_entity_id, ent_id))
                    self._insert_relation(
                        conn,
                        source_id=memory_entity_id,
                        target_id=ent_id,
                        relation_type="mentions",
                        observation_id=observation_id,
                        origin="memory",
                    )

            seen_rel: set[tuple[int, int, str]] = set()
            for r in graph.get("relations", []) or []:
                src_name = str(r.get("source") or "").strip()
                tgt_name = str(r.get("target") or "").strip()
                rt = str(r.get("type") or "related_to").strip()
                if not src_name or not tgt_name or not rt:
                    continue
                src_id = self._upsert_entity(conn, name=src_name, entity_type="other", origin="memory")
                tgt_id = self._upsert_entity(conn, name=tgt_name, entity_type="other", origin="memory")
                if src_id <= 0 or tgt_id <= 0:
                    continue
                key = (src_id, tgt_id, rt)
                if key in seen_rel:
                    continue
                seen_rel.add(key)
                self._insert_relation(
                    conn,
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rt,
                    observation_id=observation_id,
                    origin="memory",
                )
        finally:
            conn.close()

    def search(
        self,
        *,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        top_p: float = DEFAULT_TOP_P,
        use_fts: bool = True,
    ) -> List[dict]:
        k = effective_k(limit)
        results: List[dict] = []

        q_vec = self.embedder.embed(query)

        # Try native vector distance first (best-effort); fall back to Python cosine over embedding_json.
        conn = self._db(load_vec=False)
        rows = []
        try:
            if self._has_function(conn, "vector_distance_cos") and self._has_function(conn, "vector32"):
                rows = self._fetchall(
                    conn,
                    """
                    SELECT
                        o.id,
                        o.session_id,
                        o.kind,
                        o.content,
                        o.content_hash,
                        o.tags,
                        o.priority,
                        o.metadata,
                        o.created_at,
                        e.name,
                        e.entity_type,
                        vector_distance_cos(o.embedding, vector32(?)) AS distance,
                        o.embedding_json
                    FROM observations o
                    JOIN entities e ON o.entity_id = e.id
                    WHERE o.embedding IS NOT NULL
                    ORDER BY distance ASC
                    LIMIT ?
                    """,
                    (json.dumps(list(map(float, q_vec))), k),
                )
        except Exception:
            rows = []
        finally:
            conn.close()

        if rows:
            for row in rows:
                results.append(
                    {
                        "id": int(row[0]),
                        "session_id": row[1],
                        "kind": row[2],
                        "content": row[3],
                        "content_hash": row[4],
                        "tags": row[5],
                        "priority": row[6],
                        "metadata": json.loads(row[7] or "{}"),
                        "created_at": row[8],
                        "entity_name": row[9],
                        "entity_type": row[10],
                        "score": float(row[11]),
                        "fts_hit": False,
                    }
                )

        # Always merge FTS results
        fts_hits = self._fts_lookup(query, limit=k)
        by_id = {r["id"]: r for r in results}
        for hit in fts_hits:
            if hit["id"] in by_id:
                by_id[hit["id"]]["fts_hit"] = True
            else:
                hit["fts_hit"] = True
                results.append(hit)

        if session_id:
            # Keep all results, but allow session-aware reranking to prefer matching session.
            pass

        if not results:
            results = self._python_vector_search(query=query, q_vec=q_vec, limit=k)

        if not results:
            results = self._recent(limit=limit)

        results = rerank_results(results, top_p, session_id=session_id)[:limit]
        return results

    def _python_vector_search(self, *, query: str, q_vec: List[float], limit: int) -> List[dict]:
        # Candidate selection: merge FTS hits with recent rows, then compute cosine distance in Python.
        candidates: List[dict] = []

        candidates.extend(self._fts_lookup(query, limit=limit))

        if len(candidates) < limit:
            by_id = {c["id"] for c in candidates}
            for r in self._recent(limit=limit * 2):
                if r["id"] not in by_id:
                    candidates.append(r)
                if len(candidates) >= limit * 2:
                    break

        if not candidates or not q_vec:
            return candidates[:limit]

        import numpy as np

        q = np.asarray(q_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0

        conn = self._db()
        try:
            ids = [c["id"] for c in candidates]
            placeholders = ",".join("?" for _ in ids)
            rows = self._fetchall(
                conn,
                f"SELECT id, embedding_json FROM observations WHERE id IN ({placeholders})",
                ids,
            )
        finally:
            conn.close()

        emb_by_id: dict[int, List[float]] = {}
        for oid, emb_json in rows:
            try:
                emb_by_id[int(oid)] = list(map(float, json.loads(emb_json or "[]")))
            except Exception:
                emb_by_id[int(oid)] = []

        for c in candidates:
            v = emb_by_id.get(int(c["id"])) or []
            if not v:
                c["score"] = float("inf")
                continue
            vec = np.asarray(v, dtype=np.float32)
            vn = float(np.linalg.norm(vec)) or 1.0
            cos = float(np.dot(q, vec) / (qn * vn))
            # Convert similarity to distance-like score (lower is better)
            c["score"] = 1.0 - cos
        candidates.sort(key=lambda x: x.get("score", float("inf")))
        return candidates[:limit]

    def _fts_lookup(self, query: str, limit: int) -> List[dict]:
        conn = self._db()
        try:
            rows = self._fetchall(
                conn,
                """
                SELECT
                    o.id,
                    o.session_id,
                    o.kind,
                    o.content,
                    o.content_hash,
                    o.tags,
                    o.priority,
                    o.metadata,
                    o.created_at,
                    e.name,
                    e.entity_type
                FROM observations_fts f
                JOIN observations o ON f.rowid = o.id
                JOIN entities e ON o.entity_id = e.id
                WHERE observations_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            )
        except Exception as exc:
            logger.warning("FTS query failed for %s: %s", query, exc)
            return []
        finally:
            conn.close()

        out: List[dict] = []
        for row in rows:
            out.append(
                {
                    "id": int(row[0]),
                    "session_id": row[1],
                    "kind": row[2],
                    "content": row[3],
                    "content_hash": row[4],
                    "tags": row[5],
                    "priority": row[6],
                    "metadata": json.loads(row[7] or "{}"),
                    "created_at": row[8],
                    "entity_name": row[9],
                    "entity_type": row[10],
                    "score": 0.0,
                    "fts_hit": True,
                }
            )
        return out

    def _recent(self, *, limit: int) -> List[dict]:
        conn = self._db()
        try:
            rows = self._fetchall(
                conn,
                """
                SELECT
                    o.id, o.session_id, o.kind, o.content, o.content_hash, o.tags, o.priority, o.metadata, o.created_at,
                    e.name, e.entity_type
                FROM observations o
                JOIN entities e ON o.entity_id = e.id
                ORDER BY o.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        finally:
            conn.close()

        return [
            {
                "id": int(r[0]),
                "session_id": r[1],
                "kind": r[2],
                "content": r[3],
                "content_hash": r[4],
                "tags": r[5],
                "priority": r[6],
                "metadata": json.loads(r[7] or "{}"),
                "created_at": r[8],
                "entity_name": r[9],
                "entity_type": r[10],
                "score": 1.0,
                "fts_hit": False,
            }
            for r in rows
        ]

    def recent(self, *, limit: int = 20) -> List[dict]:
        return self._recent(limit=limit)[:limit]

    def get_observation(self, observation_id: int) -> Optional[dict]:
        if observation_id <= 0:
            return None
        conn = self._db()
        try:
            row = self._fetchone(
                conn,
                """
                SELECT
                    o.id,
                    o.session_id,
                    o.kind,
                    o.content,
                    o.content_hash,
                    o.tags,
                    o.priority,
                    o.metadata,
                    o.created_at,
                    e.name,
                    e.entity_type
                FROM observations o
                JOIN entities e ON o.entity_id = e.id
                WHERE o.id = ?
                """,
                (observation_id,),
            )
        finally:
            conn.close()

        if not row:
            return None
        return {
            "id": int(row[0]),
            "session_id": row[1],
            "kind": row[2],
            "content": row[3],
            "content_hash": row[4],
            "tags": row[5],
            "priority": row[6],
            "metadata": json.loads(row[7] or "{}"),
            "created_at": row[8],
            "entity_name": row[9],
            "entity_type": row[10],
        }

    def list_entities(self, observation_id: int) -> List[dict]:
        conn = self._db()
        try:
            row = self._fetchone(conn, "SELECT entity_id FROM observations WHERE id = ?", (observation_id,))
            if not row:
                return []
            memory_entity_id = int(row[0])
            rows = self._fetchall(
                conn,
                """
                SELECT e.name, e.entity_type, r.relation_type
                FROM relations r
                JOIN entities e ON e.id = r.target_id
                WHERE r.source_id = ?
                  AND r.observation_id = ?
                ORDER BY e.name
                """,
                (memory_entity_id, observation_id),
            )
        finally:
            conn.close()
        return [{"name": r[0], "entity_type": r[1], "relation_type": r[2]} for r in rows]

    # Graph-style helpers used by server tools
    def upsert_graph_entity(
        self,
        name: str,
        entity_type: str,
        observations: List[str],
        memory_id: Optional[int] = None,
    ) -> int:
        conn = self._db(load_vec=True)
        try:
            entity_id = self._upsert_entity(conn, name=name, entity_type=entity_type or "other", origin="agent")
            for obs in observations or []:
                if not obs:
                    continue
                cur = self._execute(
                    conn,
                    """
                    INSERT INTO observations (entity_id, session_id, kind, content, content_hash, tags, priority, metadata)
                    VALUES (?, NULL, 'graph', ?, ?, NULL, 3, ?)
                    """,
                    (entity_id, obs, hash_content(obs), json.dumps({"source": "graph"})),
                )
                obs_id = self._lastrowid(cur)
                embedding = self.embedder.embed(obs)
                try:
                    self._execute(
                        conn,
                        "UPDATE observations SET embedding_json = ? WHERE id = ?",
                        (json.dumps(list(map(float, embedding))), obs_id),
                    )
                except Exception:
                    pass
                try:
                    conn.execute(
                        "UPDATE observations SET embedding = vector32(?) WHERE id = ?",
                        (json.dumps(list(map(float, embedding))), obs_id),
                    )
                except Exception:
                    pass
            if DB_BACKEND == "sqlite":
                conn.commit()
            return entity_id
        finally:
            conn.close()

    def add_graph_relation(self, source: str, target: str, relation_type: str, memory_id: Optional[int] = None) -> int:
        if not source or not target or not relation_type:
            return -1
        conn = self._db()
        try:
            src_id = self._upsert_entity(conn, name=source, entity_type="other", origin="agent")
            tgt_id = self._upsert_entity(conn, name=target, entity_type="other", origin="agent")
            self._insert_relation(
                conn,
                source_id=src_id,
                target_id=tgt_id,
                relation_type=relation_type,
                observation_id=memory_id,
                origin="agent",
            )
            if DB_BACKEND == "sqlite":
                conn.commit()
            return 1
        finally:
            conn.close()

    def get_graph_entity(self, name: str) -> dict:
        conn = self._db()
        try:
            row = self._fetchone(conn, "SELECT id, name, entity_type, created_at FROM entities WHERE name = ?", (name,))
            if not row:
                return {"status": "not_found"}
            entity_id = int(row[0])
            obs = self._fetchall(
                conn,
                "SELECT content FROM observations WHERE entity_id = ? ORDER BY id DESC LIMIT 50",
                (entity_id,),
            )
            rel = self._fetchall(
                conn,
                """
                SELECT r.relation_type, s.name, t.name
                FROM relations r
                JOIN entities s ON r.source_id = s.id
                JOIN entities t ON r.target_id = t.id
                WHERE r.source_id = ? OR r.target_id = ?
                """,
                (entity_id, entity_id),
            )
            return {
                "id": entity_id,
                "name": row[1],
                "entity_type": row[2],
                "created_at": row[3],
                "observations": [o[0] for o in obs],
                "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in rel],
            }
        finally:
            conn.close()

    def read_graph(self, limit: int = 10) -> dict:
        conn = self._db()
        try:
            entities = self._fetchall(
                conn,
                """
                SELECT id, name, entity_type, created_at
                FROM entities
                WHERE entity_type != 'memory'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            entity_ids = [e[0] for e in entities]
            relations = []
            if entity_ids:
                placeholders = ",".join("?" for _ in entity_ids)
                relations = self._fetchall(
                    conn,
                    f"""
                    SELECT r.relation_type, s.name, t.name
                    FROM relations r
                    JOIN entities s ON r.source_id = s.id
                    JOIN entities t ON r.target_id = t.id
                    WHERE r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders})
                    """,
                    (*entity_ids, *entity_ids),
                )
            return {
                "entities": [{"id": e[0], "name": e[1], "entity_type": e[2], "created_at": e[3]} for e in entities],
                "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in relations],
            }
        finally:
            conn.close()

    def search_graph(self, query: str, limit: int = 5) -> dict:
        # Simplest approach: search observations, then group by entity.
        hits = self.search(query=query, session_id=None, limit=max(20, limit * 5), top_p=1.0, use_fts=True)
        by_entity = {}
        for h in hits:
            name = h.get("entity_name")
            et = h.get("entity_type")
            if not name:
                continue
            if et == "memory":
                continue
            if name not in by_entity:
                by_entity[name] = {"name": name, "entity_type": et, "score": h.get("score", 0.0)}
            else:
                by_entity[name]["score"] = min(by_entity[name]["score"], h.get("score", 0.0))
        entities = list(by_entity.values())[:limit]

        # Fetch relations between returned entities (best effort)
        rel = []
        if entities:
            conn = self._db()
            try:
                names = [e["name"] for e in entities]
                placeholders = ",".join("?" for _ in names)
                rel = self._fetchall(
                    conn,
                    f"""
                    SELECT r.relation_type, s.name, t.name
                    FROM relations r
                    JOIN entities s ON r.source_id = s.id
                    JOIN entities t ON r.target_id = t.id
                    WHERE s.name IN ({placeholders}) OR t.name IN ({placeholders})
                    LIMIT 200
                    """,
                    (*names, *names),
                )
            finally:
                conn.close()

        return {"entities": entities, "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in rel]}
