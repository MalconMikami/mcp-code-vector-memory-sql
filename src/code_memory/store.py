from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

from fastmcp import Context

from .config import (
    DB_PATH,
    DEFAULT_TOP_P,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    ENABLE_FTS,
    ENABLE_GRAPH,
    ENABLE_VEC,
    logger,
)
from .db import connect_db
from .embeddings import EmbeddingModel, serialize_vector
from .entities import extract_entities
from .rerank import effective_k, rerank_results
from .security import hash_content, looks_sensitive
from .summary import auto_summary


def clamp_priority(value: int, min_v: int = 1, max_v: int = 5) -> int:
    try:
        return max(min_v, min(max_v, int(value)))
    except Exception:
        return 3


def auto_tags(content: str) -> str:
    keywords = ["bug", "error", "fix", "database", "api", "frontend", "backend", "ui", "test", "deploy"]
    lower = content.lower()
    found = [k for k in keywords if k in lower]
    return ",".join(found) if found else "general"


class MemoryStore:
    def __init__(self, db_path: Path = DB_PATH, embedder: Optional[EmbeddingModel] = None):
        self.db_path = db_path
        self.embedder = embedder or EmbeddingModel(embed_dim=EMBED_DIM)
        self._ensure_schema()
        logger.info("MemoryStore initialized at %s with embedder %s", db_path, EMBED_MODEL_NAME)

    def _db(self, *, load_vec: bool = False):
        return connect_db(db_path=self.db_path, enable_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC), load_vec=load_vec)

    def _ensure_schema(self) -> None:
        conn = self._db(load_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC))
        logger.info("Opening/checking SQLite schema at %s", self.db_path)
        try:
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

            if ENABLE_VEC:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
                        embedding float[{self.embedder.dim}]
                    )
                    """
                )

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

            try:
                conn.execute("ALTER TABLE memories ADD COLUMN session_id TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE memories ADD COLUMN priority INTEGER DEFAULT 3")
            except Exception:
                pass

            conn.commit()
            logger.info("Schema ready: vec=%s fts=%s graph=%s", ENABLE_VEC, ENABLE_FTS, ENABLE_GRAPH)
        finally:
            conn.close()

    def add(
        self,
        *,
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
            logger.error("Missing session_id; memory will not be stored.")
            return -1
        if looks_sensitive(content):
            if ctx:
                ctx.warning("Skipping store: sensitive content detected.")
            logger.warning("Sensitive content detected; skipping store.")
            return -1

        content_hash = hash_content(content)
        if self._is_recent_duplicate(content_hash):
            if ctx:
                ctx.info("Skipping store: recent duplicate content.")
            logger.info("Recent duplicate content; skipping store.")
            return -1

        embedding = self.embedder.embed(content) if ENABLE_VEC else []
        summary_value = summary or auto_summary(content)
        tags_value = tags or auto_tags(content)
        prio = clamp_priority(priority)

        entities = extract_entities(content)

        conn = self._db(load_vec=ENABLE_VEC)
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
                    summary_value,
                    tags_value,
                    prio,
                    json.dumps(metadata or {}),
                    content_hash,
                ),
            )
            row_id = cur.lastrowid

            if ENABLE_VEC:
                conn.execute(
                    "INSERT INTO vec_memories(rowid, embedding) VALUES (?, ?)",
                    (row_id, serialize_vector(embedding)),
                )

            if entities:
                conn.executemany(
                    """
                    INSERT INTO entities (memory_id, entity_type, name, source, path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(row_id, e["type"], e["name"], e["source"], e.get("path")) for e in entities],
                )

            conn.commit()

            if entities and ENABLE_GRAPH:
                obs_text = summary_value or content
                for e in entities:
                    self.upsert_graph_entity(
                        name=e["name"],
                        entity_type=e.get("type") or "entity",
                        observations=[obs_text],
                        memory_id=row_id,
                    )

            return int(row_id)
        finally:
            logger.debug(
                "Stored memory row_id=%s session_id=%s priority=%s tags=%s",
                row_id if "row_id" in locals() else None,
                session_id,
                prio,
                tags_value,
            )
            conn.close()

    def _is_recent_duplicate(self, content_hash: str, window_seconds: int = 300) -> bool:
        conn = self._db()
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
                last_time = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
            except Exception:
                return False
            return (time.time() - last_time) < window_seconds
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
        q_vec = self.embedder.embed(query) if ENABLE_VEC else None
        k = effective_k(limit)
        conn = self._db(load_vec=ENABLE_VEC)
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
                    (serialize_vector(q_vec), k, session_id, session_id, k),
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

        results = rerank_results(results, top_p)[:limit]
        logger.debug("Search query=%s session_id=%s limit=%s hits=%s", query, session_id, limit, len(results))
        logger.debug("Search results sample: %s", results[:3])
        return results

    def _fts_lookup(self, query: str, session_id: Optional[str], limit: int) -> List[dict]:
        conn = self._db()
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

    def recent(self, *, limit: int = 20) -> List[dict]:
        conn = self._db()
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

    def _recent_filtered(self, session_id: Optional[str], limit: int) -> List[dict]:
        conn = self._db()
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
        conn = self._db()
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
        return [{"entity_type": row[0], "name": row[1], "source": row[2], "path": row[3]} for row in rows]

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
        conn = self._db(load_vec=ENABLE_VEC)
        try:
            row = conn.execute("SELECT id FROM graph_entities WHERE name = ?", (name,)).fetchone()
            if row:
                entity_id = row[0]
                conn.execute("UPDATE graph_entities SET entity_type = ? WHERE id = ?", (entity_type, entity_id))
            else:
                cur = conn.execute("INSERT INTO graph_entities (name, entity_type) VALUES (?, ?)", (name, entity_type))
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
                    (entity_id, serialize_vector(embedding)),
                )
            conn.commit()
            return int(entity_id)
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
        conn = self._db()
        try:
            cur = conn.execute(
                """
                INSERT INTO graph_relations (source_id, target_id, relation_type, memory_id)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, relation_type, memory_id),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def get_graph_entity(self, name: str) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        conn = self._db()
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
                "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in rel],
            }
        finally:
            conn.close()

    def read_graph(self, limit: int = 10) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        conn = self._db()
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
                "entities": [{"id": e[0], "name": e[1], "entity_type": e[2], "created_at": e[3]} for e in entities],
                "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in relations],
            }
        finally:
            conn.close()

    def search_graph(self, query: str, limit: int = 5) -> dict:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}

        rows = []
        if ENABLE_VEC:
            conn = self._db(load_vec=True)
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
                    (serialize_vector(q_vec), limit, limit),
                ).fetchall()
            finally:
                conn.close()

        if not rows:
            conn = self._db()
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
            conn = self._db()
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
                {"id": r[0], "name": r[1], "entity_type": r[2], "created_at": r[3], "score": r[4]} for r in rows
            ],
            "relations": [{"type": r[0], "source": r[1], "target": r[2]} for r in rel],
        }

