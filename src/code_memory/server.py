import json
import os
import time
from pathlib import Path
from typing import List, Optional

from fastmcp import Context, FastMCP

from .config import (
    DB_PATH,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    EMBED_MODEL_NAME,
    ENABLE_FTS,
    ENABLE_GRAPH,
    ENABLE_VEC,
    FTS_BONUS,
    LOG_FILE,
    LOG_LEVEL,
    MODEL_CACHE_DIR,
    OVERSAMPLE_K,
    PRIORITY_WEIGHT,
    RECENCY_WEIGHT,
    ROOT,
    SUMMARY_AUTO_INSTALL,
    SUMMARY_CTX,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TEMPERATURE,
    SUMMARY_THREADS,
    logger,
)
from .db import connect_db
from .entities import TREE_SITTER_AVAILABLE
from .embeddings import EmbeddingModel
from .rerank import apply_recency_filter, clamp_top_k, clamp_top_p, parse_timestamp
from .security import hash_content, looks_sensitive
from .store import MemoryStore
from . import summary as _summary
from .config import EMBED_DIM, get_setting


# ---------------------------------------------------------------------------
# Backwards-compatible re-exports (tests + external integrations may import these)
# ---------------------------------------------------------------------------
EmbeddingModel = EmbeddingModel
MemoryStore = MemoryStore
EMBED_DIM = EMBED_DIM


def _looks_sensitive(text: str) -> bool:
    return looks_sensitive(text)


def _hash_content(content: str) -> str:
    return hash_content(content)


def _clamp_priority(value: int, min_v: int = 1, max_v: int = 5) -> int:
    return _store_clamp_priority(value, min_v=min_v, max_v=max_v)


def _apply_recency_filter(results: List[dict], top_p: float) -> List[dict]:
    return apply_recency_filter(results, top_p)


def _clamp_top_k(value: int) -> int:
    return clamp_top_k(value)


def _clamp_top_p(value: float) -> float:
    return clamp_top_p(value)


def _connect_db(load_vec: bool = False):
    return connect_db(db_path=DB_PATH, enable_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC), load_vec=load_vec)


# Summary model compatibility: keep server.SUMMARY_MODEL as a mutable variable used by tests/tools.
SUMMARY_MODEL = os.getenv("CODE_MEMORY_SUMMARY_MODEL", "").strip()


def _download_gguf_from_repo(repo_id: str) -> str:
    return _summary.download_gguf_from_repo(repo_id)


def _looks_like_hf_repo_id(value: str) -> bool:
    return _summary.looks_like_hf_repo_id(value)


def _resolve_summary_model_path() -> str:
    configured = str(get_setting("CODE_MEMORY_SUMMARY_MODEL", "") or "").strip()
    if configured:
        if Path(configured).exists():
            return configured
        if _looks_like_hf_repo_id(configured):
            if os.getenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "1").lower() in ("0", "false", "no"):
                return ""
            return _download_gguf_from_repo(configured)
        return configured
    return ""


def _ensure_summary_model_path() -> str:
    global SUMMARY_MODEL
    if SUMMARY_MODEL:
        try:
            if Path(SUMMARY_MODEL).exists():
                _summary.SUMMARY_MODEL = SUMMARY_MODEL
                return SUMMARY_MODEL
        except Exception:
            pass
        if _looks_like_hf_repo_id(SUMMARY_MODEL):
            resolved = _download_gguf_from_repo(SUMMARY_MODEL)
            if resolved:
                SUMMARY_MODEL = resolved
                _summary.SUMMARY_MODEL = resolved
            return SUMMARY_MODEL
        _summary.SUMMARY_MODEL = SUMMARY_MODEL
        return SUMMARY_MODEL

    resolved = _resolve_summary_model_path()
    if resolved:
        SUMMARY_MODEL = resolved
        _summary.SUMMARY_MODEL = resolved
        return SUMMARY_MODEL

    return ""


def _summary_enabled() -> bool:
    _summary.SUMMARY_MODEL = SUMMARY_MODEL
    return _summary.summary_enabled()


from .store import clamp_priority as _store_clamp_priority  # placed after to avoid circular formatting


server = FastMCP(
    name="mcp-code-vector-memory-sql",
    instructions=(
        "You have access to a fast local memory MCP server. Use it aggressively.\n"
        "\n"
        "Most important rules:\n"
        "  1) ALWAYS use memory at the start of a session.\n"
        "     - As soon as a new session starts (or when you are unsure), call search_memory to recall context.\n"
        "     - This is fast and can be called frequently.\n"
        "  2) ALWAYS scope memory by session_id.\n"
        "     - You MUST pass the session_id from the client/tool you are currently using.\n"
        "     - This server isolates memories per session_id; do not mix sessions.\n"
        "  3) ALWAYS respect priority.\n"
        "     - Priority is 1..5 where 1 = highest importance and 5 = lowest.\n"
        "     - If the user explicitly says something is important, store it with priority=1 or 2.\n"
        "     - Use priority=3 for normal notes, and 4-5 for low-signal breadcrumbs.\n"
        "\n"
        "What to do at initialization (recommended):\n"
        "  - Call search_memory(query=\"\", session_id=..., limit=..., top_p=...) to pull recent high-signal context.\n"
        "  - If you have a concrete goal, set query to that goal (e.g., \"current task\", \"architecture decision\").\n"
        "\n"
        "When to store memories:\n"
        "  - After decisions, constraints, and user preferences.\n"
        "  - After completing meaningful work (so future you can pick up quickly).\n"
        "  - After discovering important facts (commands, file paths, error causes, fixes).\n"
        "\n"
        "Basic concepts:\n"
        "  - memories: stored text entries (optionally summarized), searchable.\n"
        "  - tags: comma-separated labels to filter/organize (e.g., \"bugfix,db,perf\").\n"
        "  - kind: a short category (e.g., \"decision\", \"bugfix\", \"plan\", \"note\").\n"
        "  - entities/graph: optional structured extraction and relations to improve recall.\n"
        "\n"
        "Tools you can call (high level):\n"
        "  - search_memory(query, session_id, limit, top_p): retrieve relevant memories for THIS session.\n"
        "  - remember(content, session_id, kind, summary, tags, priority, metadata_json): store new memory for THIS session.\n"
        "  - list_recent(limit): inspect recent memories (debug/inspection).\n"
        "  - list_entities(memory_id): view extracted entities for a memory.\n"
        "  - get_context_graph(query, limit): fetch/search the context graph.\n"
        "  - health(), diagnostics(): check server status/config.\n"
        "\n"
        "Session scoping reminder:\n"
        "  - For EVERY remember/search call, include the correct session_id for the session you are working in.\n"
        "\n"
        "Notes:\n"
        "  - Sensitive content may be skipped.\n"
        "  - Recent duplicates may be skipped.\n"
        "  - This system is designed to be queried often; do not hesitate to call search_memory multiple times."
    ),
)


store = MemoryStore(DB_PATH)


def _resolve_session_id(session_id: Optional[str], ctx: Context | None) -> Optional[str]:
    if session_id:
        return session_id
    if ctx and getattr(ctx, "session_id", None):
        return ctx.session_id
    env_session = os.getenv("CODE_MEMORY_SESSION_ID")
    return env_session or None


# -------------
# MCP tools
# -------------
@server.tool(
    description=(
        "Store a new memory entry (optionally: vector embedding, FTS indexing, entity extraction, and graph observations).\n"
        "\n"
        "When to call:\n"
        "- After the user states durable facts (preferences, constraints, decisions, reminders).\n"
        "- After you discover non-obvious project context (paths, commands, setup details, gotchas).\n"
        "- After you complete work that will be useful later (what changed, why, and how to verify).\n"
        "\n"
        "Required:\n"
        "- session_id: Always pass the current session id so memories can be searched per-session.\n"
        "  (If omitted, the server will try ctx.session_id or CODE_MEMORY_SESSION_ID.)\n"
        "\n"
        "Parameters:\n"
        "- content (str): The memory text to store. Keep it short and specific.\n"
        "- session_id (str): Session scope for retrieval. Required.\n"
        "- kind (str|None): Optional category. Examples: 'decision', 'preference', 'todo', 'context', 'bug'.\n"
        "- summary (str|None): Optional human-readable summary. If omitted, the server auto-generates one.\n"
        "- tags (str|None): Optional comma-separated tags. If omitted, the server auto-generates tags.\n"
        "  Examples: 'setup,windows,sqlite', 'rag,search,tuning'.\n"
        "- priority (int): 1..5 where 1 = most important. Affects reranking during search.\n"
        "- metadata_json (str|None): JSON object as a string. Stored as metadata.\n"
        "  Example: '{\"repo\":\"mcp-code-vector-memory-sql\",\"path\":\"src/code_memory/server.py\"}'.\n"
        "\n"
        "Return (success):\n"
        "{\n"
        "  \"id\": 123,\n"
        "  \"summary\": \"...\",\n"
        "  \"session_id\": \"...\",\n"
        "  \"kind\": \"decision\",\n"
        "  \"tags\": \"setup,windows\",\n"
        "  \"priority\": 2\n"
        "}\n"
        "\n"
        "Return (skipped):\n"
        "- {\"status\":\"skipped\"} when content looks sensitive or the content is a recent duplicate.\n"
        "  (If session_id is missing, this tool returns an error instead.)\n"
        "\n"
        "Return (error):\n"
        "- {\"status\":\"error\",\"error\":\"...\",\"tool\":\"remember\"}\n"
    )
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
        logger.exception("Tool remember failed")
        return {"status": "error", "error": str(exc), "tool": "remember"}


@server.tool(
    description=(
        "Hybrid search over stored memories (semantic vector search + optional FTS re-rank).\n"
        "\n"
        "Required:\n"
        "- session_id: Always pass the current session id to search only within the current session.\n"
        "  (If omitted, the server will try ctx.session_id or CODE_MEMORY_SESSION_ID.)\n"
        "\n"
        "Parameters:\n"
        "- query (str): Natural language or keyword query. Also used for FTS matching.\n"
        "- session_id (str): Session scope for retrieval. Required.\n"
        "- limit (int): Number of results to return (top_k). Default comes from CODE_MEMORY_TOP_K.\n"
        "  Internally the server oversamples candidates by CODE_MEMORY_OVERSAMPLE_K before reranking.\n"
        "- top_p (float): Recency window for reranking (0 < top_p <= 1). Default from CODE_MEMORY_TOP_P.\n"
        "  top_p=1.0 keeps all candidates; top_p=0.6 keeps only the newest 60% of candidates (by created_at)\n"
        "  before final sorting. Use this to bias toward recent context.\n"
        "\n"
        "How results are ranked:\n"
        "- Starts from vector distance (lower is better).\n"
        "- Applies an FTS bonus when there is an FTS hit.\n"
        "- Applies a priority penalty (priority 1 is favored over priority 5).\n"
        "- Applies a recency penalty (older is ranked lower).\n"
        "\n"
        "Return: list[dict] sorted by score (lower is better). Each item contains:\n"
        "- id, session_id, kind, content, summary, tags, priority, metadata (object), created_at\n"
        "- score (float): final reranked score (lower is better)\n"
        "- fts_hit (bool): whether the item matched via FTS\n"
        "\n"
        "Example return:\n"
        "[\n"
        "  {\n"
        "    \"id\": 101,\n"
        "    \"session_id\": \"s-123\",\n"
        "    \"kind\": \"decision\",\n"
        "    \"summary\": \"Use SQLite vec0 for embeddings...\",\n"
        "    \"tags\": \"sqlite,vec,rag\",\n"
        "    \"priority\": 2,\n"
        "    \"score\": 0.42,\n"
        "    \"fts_hit\": true\n"
        "  }\n"
        "]\n"
        "\n"
        "Tuning tips:\n"
        "- Broad recall: limit=20..50, top_p=1.0.\n"
        "- Recent-only context (avoid stale matches): limit=10..20, top_p=0.3..0.7.\n"
        "- Exact identifiers/tags: include the exact token(s) in query (FTS searches content/summary/tags/metadata).\n"
        "\n"
        "Return (error):\n"
        "- {\"status\":\"error\",\"error\":\"...\",\"tool\":\"search_memory\"}\n"
    )
)
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
        limit = clamp_top_k(limit)
        top_p = clamp_top_p(top_p)
        return store.search(query=query, session_id=resolved_session_id, limit=limit, top_p=top_p)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool search_memory failed")
        return {"status": "error", "error": str(exc), "tool": "search_memory"}


@server.tool(
    description=(
        "List the most recent memory entries (across all sessions).\n"
        "\n"
        "Use cases:\n"
        "- Debugging / inspection.\n"
        "- Checking what has been stored recently.\n"
        "\n"
        "Parameters:\n"
        "- limit (int): Max number of rows to return (default: 20).\n"
        "\n"
        "Return: list[dict] ordered by created_at desc. Each item contains:\n"
        "- id, session_id, kind, content, summary, tags, priority, metadata (object), created_at\n"
        "\n"
        "Example return:\n"
        "[{\"id\":1,\"session_id\":\"s-123\",\"summary\":\"...\",\"tags\":\"general\",\"priority\":3,\"created_at\":\"2026-01-04 12:34:56\"}]\n"
    )
)
def list_recent(limit: int = 20) -> List[dict]:
    try:
        return store.recent(limit=limit)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_recent failed")
        return {"status": "error", "error": str(exc), "tool": "list_recent"}


@server.tool(
    description=(
        "List entities extracted from a single memory entry.\n"
        "\n"
        "Entity extraction is best-effort and may use regex and/or tree-sitter (when available).\n"
        "\n"
        "Parameters:\n"
        "- memory_id (int): Memory id returned by remember().\n"
        "\n"
        "Return: list[dict] where each item contains:\n"
        "- entity_type (str), name (str), source (str), path (str|None)\n"
        "\n"
        "Example return:\n"
        "[{\"entity_type\":\"function\",\"name\":\"search_memory\",\"source\":\"tree-sitter\",\"path\":null}]\n"
    )
)
def list_entities(memory_id: int) -> List[dict]:
    try:
        return store.list_entities(memory_id)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_entities failed")
        return {"status": "error", "error": str(exc), "tool": "list_entities"}


@server.tool(
    description=(
        "Upsert (create or update) a knowledge-graph entity and attach observations.\n"
        "\n"
        "Notes:\n"
        "- Requires CODE_MEMORY_ENABLE_GRAPH=1. If disabled, returns status 'disabled'.\n"
        "- Observations should be short sentences, facts, or summaries.\n"
        "\n"
        "Parameters:\n"
        "- name (str): Entity name (unique).\n"
        "- entity_type (str): Entity type/category (default: 'entity'). Examples: 'function', 'class', 'service'.\n"
        "- observations_json (str|None): JSON list of strings.\n"
        "  Example: '[\"This function performs hybrid search.\",\"Used in the MCP tool layer.\"]'\n"
        "- memory_id (int|None): Optional memory id to link the observation to a specific memory entry.\n"
        "\n"
        "Return:\n"
        "- {\"status\":\"ok\",\"entity_id\": 42}\n"
        "- {\"status\":\"disabled\"} if the graph feature is disabled.\n"
    )
)
def upsert_entity(
    name: str,
    entity_type: str = "entity",
    observations_json: Optional[str] = None,
    memory_id: Optional[int] = None,
) -> dict:
    try:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        observations = json.loads(observations_json) if observations_json else []
        if not isinstance(observations, list):
            observations = [str(observations)]
        entity_id = store.upsert_graph_entity(name, entity_type, observations, memory_id)
        return {"status": "ok", "entity_id": entity_id}
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool upsert_entity failed")
        return {"status": "error", "error": str(exc), "tool": "upsert_entity"}


@server.tool(
    description=(
        "Add a directed relation between two knowledge-graph entities.\n"
        "\n"
        "Notes:\n"
        "- Requires CODE_MEMORY_ENABLE_GRAPH=1. If disabled, returns status 'disabled'.\n"
        "- If the entities do not exist, they are created automatically.\n"
        "\n"
        "Parameters:\n"
        "- source (str): Source entity name.\n"
        "- target (str): Target entity name.\n"
        "- relation_type (str): Relation label. Examples: 'calls', 'depends_on', 'implements', 'uses'.\n"
        "- memory_id (int|None): Optional memory id to link the relation to a memory entry.\n"
        "\n"
        "Return:\n"
        "- {\"status\":\"ok\",\"relation_id\": 7}\n"
        "- {\"status\":\"disabled\"} if the graph feature is disabled.\n"
    )
)
def add_relation(
    source: str,
    target: str,
    relation_type: str,
    memory_id: Optional[int] = None,
) -> dict:
    try:
        if not ENABLE_GRAPH:
            return {"status": "disabled"}
        rel_id = store.add_graph_relation(source, target, relation_type, memory_id)
        return {"status": "ok", "relation_id": rel_id}
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool add_relation failed")
        return {"status": "error", "error": str(exc), "tool": "add_relation"}


@server.tool(
    description=(
        "Fetch a knowledge-graph entity by name, including observations and relations.\n"
        "\n"
        "Notes:\n"
        "- Requires CODE_MEMORY_ENABLE_GRAPH=1. If disabled, returns {\"status\":\"disabled\"}.\n"
        "- If the entity does not exist, returns {\"status\":\"not_found\"}.\n"
        "\n"
        "Parameters:\n"
        "- name (str): Entity name.\n"
        "\n"
        "Return (found):\n"
        "{\n"
        "  \"id\": 42,\n"
        "  \"name\": \"search_memory\",\n"
        "  \"entity_type\": \"function\",\n"
        "  \"created_at\": \"2026-01-04 12:34:56\",\n"
        "  \"observations\": [\"...\"],\n"
        "  \"relations\": [{\"type\":\"calls\",\"source\":\"A\",\"target\":\"B\"}]\n"
        "}\n"
        "\n"
        "Return (disabled/not found):\n"
        "- {\"status\":\"disabled\"} or {\"status\":\"not_found\"}\n"
    )
)
def get_entity(name: str) -> dict:
    try:
        return store.get_graph_entity(name)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_entity failed")
        return {"status": "error", "error": str(exc), "tool": "get_entity"}


@server.tool(
    description=(
        "Return a context graph snapshot (entities + relations).\n"
        "\n"
        "If query is provided, performs a semantic search over graph entities (uses embeddings when enabled).\n"
        "If query is omitted, returns the most recent entities in the graph.\n"
        "\n"
        "Notes:\n"
        "- Requires CODE_MEMORY_ENABLE_GRAPH=1. If disabled, returns {\"status\":\"disabled\"}.\n"
        "\n"
        "Parameters:\n"
        "- query (str|None): Optional semantic query to select relevant entities.\n"
        "- limit (int): Maximum number of entities to return (default: 10).\n"
        "\n"
        "Return:\n"
        "{\n"
        "  \"entities\": [{\"id\": 1, \"name\": \"X\", \"entity_type\": \"...\", \"created_at\": \"...\", \"score\": 0.12}],\n"
        "  \"relations\": [{\"type\": \"depends_on\", \"source\": \"X\", \"target\": \"Y\"}]\n"
        "}\n"
        "Note: 'score' is only present when query is provided (semantic search).\n"
    )
)
def get_context_graph(query: Optional[str] = None, limit: int = 10) -> dict:
    try:
        if query:
            return store.search_graph(query, limit=limit)
        return store.read_graph(limit=limit)
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_context_graph failed")
        return {"status": "error", "error": str(exc), "tool": "get_context_graph"}


@server.tool(
    description=(
        "Manual maintenance operations on the SQLite database.\n"
        "\n"
        "Safety:\n"
        "- Destructive actions require confirm=true.\n"
        "- Nothing runs automatically.\n"
        "\n"
        "Parameters:\n"
        "- action (str): One of:\n"
        "  - 'vacuum' (non-destructive): run SQLite VACUUM\n"
        "  - 'purge_all' (destructive): delete all memories\n"
        "  - 'purge_session' (destructive): delete memories for a specific session_id\n"
        "  - 'prune_older_than' (destructive): delete memories older than older_than_days\n"
        "- confirm (bool): Required for destructive actions (any action except 'vacuum').\n"
        "- session_id (str|None): Required when action='purge_session'.\n"
        "- older_than_days (int|None): Required when action='prune_older_than'.\n"
        "\n"
        "Return:\n"
        "- {\"status\":\"ok\",\"action\":\"vacuum\"}\n"
        "- {\"status\":\"ok\",\"deleted\": N}\n"
        "- {\"status\":\"error\",\"error\":\"...\"}\n"
    )
)
def maintenance(
    action: str,
    confirm: bool = False,
    session_id: Optional[str] = None,
    older_than_days: Optional[int] = None,
) -> dict:
    try:
        if action != "vacuum" and not confirm:
            return {"status": "error", "error": "confirm=true required for destructive actions"}

        conn = connect_db(db_path=DB_PATH, enable_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC), load_vec=ENABLE_VEC)
        try:
            if action == "vacuum":
                conn.execute("VACUUM")
                return {"status": "ok", "action": "vacuum"}

            ids = []
            if action == "purge_all":
                ids = [row[0] for row in conn.execute("SELECT id FROM memories").fetchall()]
            elif action == "purge_session" and session_id:
                ids = [row[0] for row in conn.execute("SELECT id FROM memories WHERE session_id = ?", (session_id,)).fetchall()]
            elif action == "prune_older_than" and older_than_days is not None:
                cutoff = time.time() - (older_than_days * 86400)
                rows = conn.execute("SELECT id, created_at FROM memories").fetchall()
                for mid, created_at in rows:
                    ts = parse_timestamp(created_at)
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
        logger.exception("Tool maintenance failed")
        return {"status": "error", "error": str(exc), "tool": "maintenance"}


@server.tool(
    description=(
        "Diagnostics for environment, DB, and feature flags.\n"
        "\n"
        "Use cases:\n"
        "- Confirm which features are enabled (vec/fts/graph).\n"
        "- Confirm embedding model/dim, cache paths, and defaults.\n"
        "- Inspect table presence and approximate row counts.\n"
        "\n"
        "Return: a dict containing cwd, db_path, embedding config, flags, defaults, summary config, tables, and counts.\n"
    )
)
def diagnostics() -> dict:
    try:
        conn = connect_db(db_path=DB_PATH, enable_vec=ENABLE_VEC or (ENABLE_GRAPH and ENABLE_VEC), load_vec=ENABLE_VEC)
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
        logger.exception("Tool diagnostics failed")
        return {"status": "error", "error": str(exc), "tool": "diagnostics"}


@server.tool(
    description=(
        "Health check for the server.\n"
        "\n"
        "Return: a dict with status='ok' and key runtime configuration (db path, embedding model/dim, feature flags,\n"
        "default search parameters, and summary configuration).\n"
    )
)
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
            "workspace": str(ROOT),
        }
        logger.info("health check: %s", info)
        return info
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool health failed")
        return {"status": "error", "error": str(exc), "tool": "health"}


def main() -> None:
    try:
        server.run()
    except Exception as exc:
        logger.exception("Server startup/run failed: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
