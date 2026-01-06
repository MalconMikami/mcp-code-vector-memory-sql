import json
import os
import time
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastmcp import Context, FastMCP

from .config import (
    DB_BACKEND,
    DB_AUTH_TOKEN,
    DB_URL,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    EMBED_MODEL_NAME,
    CROSS_SESSION_PENALTY,
    FTS_BONUS,
    LOG_FILE,
    LOG_LEVEL,
    MODEL_CACHE_DIR,
    NER_AUTO_INSTALL,
    NER_CTX,
    NER_MAX_TOKENS,
    NER_TEMPERATURE,
    NER_THREADS,
    OVERSAMPLE_K,
    PRIORITY_WEIGHT,
    RECENCY_WEIGHT,
    ROOT,
    SESSION_BONUS,
    logger,
)
from .db import connect_db
from .entities import TREE_SITTER_AVAILABLE
from .embeddings import EmbeddingModel
from .rerank import apply_recency_filter, clamp_top_k, clamp_top_p, parse_timestamp
from .security import hash_content, looks_sensitive
from .store import MemoryStore
from . import ner as _ner
from .config import EMBED_DIM, get_setting


# Tool I/O logging (for tracing client requests/responses)
def _tool_io_logging_enabled() -> bool:
    # Requested behavior: when log level is INFO, log every tool call input/output.
    return str(LOG_LEVEL).upper() == "INFO"


def _truncate_for_log(value: object, max_chars: int) -> str:
    try:
        s = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        s = str(value)
    if max_chars > 0 and len(s) > max_chars:
        return s[:max_chars] + "...(truncated)"
    return s


def _log_tool_io(tool_name: str, direction: str, payload: object) -> None:
    if not _tool_io_logging_enabled():
        return
    try:
        max_chars = int(os.getenv("CODE_MEMORY_LOG_TOOL_MAX_CHARS", "20000"))
    except Exception:
        max_chars = 20000
    logger.info("[tool:%s] %s %s", tool_name, direction, _truncate_for_log(payload, max_chars=max_chars))


def _db_execute(conn, sql: str, params: Optional[tuple | list] = None):
    if DB_BACKEND == "sqlite":
        return conn.execute(sql, params or ())
    return conn.execute(sql, list(params or []))


def _db_fetchone(conn, sql: str, params: Optional[tuple | list] = None):
    res = _db_execute(conn, sql, params)
    if DB_BACKEND == "sqlite":
        return res.fetchone()
    return res.rows[0] if res.rows else None


def _db_fetchall(conn, sql: str, params: Optional[tuple | list] = None):
    res = _db_execute(conn, sql, params)
    if DB_BACKEND == "sqlite":
        return res.fetchall()
    return res.rows


def _db_lastrowid(result) -> int:
    if DB_BACKEND == "sqlite":
        return int(result.lastrowid)
    return int(getattr(result, "last_insert_rowid", 0) or 0)


def _db_commit(conn) -> None:
    if DB_BACKEND == "sqlite":
        conn.commit()


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
    return connect_db(
        enable_vec=True,
        load_vec=load_vec,
        db_url=DB_URL,
        db_auth_token=DB_AUTH_TOKEN or None,
    )


def _ner_enabled() -> bool:
    return _ner.ner_enabled()


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
        "  2) ALWAYS provide session_id when you can.\n"
        "     - You SHOULD pass the session_id from the client/tool you are currently using.\n"
        "     - The server boosts same-session results during search (session-aware ranking).\n"
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
        "  - memories: stored observations, searchable.\n"
        "  - tags: comma-separated labels to filter/organize (e.g., \"bugfix,db,perf\").\n"
        "  - kind: a short category (e.g., \"decision\", \"bugfix\", \"plan\", \"note\").\n"
        "  - entities/relations: structured world-fact extraction to improve recall.\n"
        "\n"
        "Tools you can call (high level):\n"
        "  - search_memory(query, session_id, limit, top_p): retrieve relevant memories (same-session results get a boost).\n"
        "  - remember(content, session_id, kind, tags, priority, metadata_json): store new memory for the given session.\n"
        "  - list_recent(limit): inspect recent memories (debug/inspection).\n"
        "  - list_entities(memory_id): view extracted entities for a memory.\n"
        "  - get_context_graph(query, limit): fetch/search the context graph.\n"
        "  - health(), diagnostics(): check server status/config.\n"
        "\n"
        "Session reminder:\n"
        "  - For EVERY remember call, include the correct session_id.\n"
        "  - For search_memory, include session_id when available so the server can boost same-session results.\n"
        "\n"
        "Notes:\n"
        "  - Sensitive content may be skipped.\n"
        "  - Recent duplicates may be skipped.\n"
        "  - This system is designed to be queried often; do not hesitate to call search_memory multiple times."
    ),
)


store = MemoryStore()

_BG_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("CODE_MEMORY_BG_WORKERS", "1") or "1")),
    thread_name_prefix="code-memory-bg",
)


def _shutdown_bg_executor() -> None:
    try:
        _BG_EXECUTOR.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


atexit.register(_shutdown_bg_executor)


def _resolve_session_id(session_id: Optional[str], ctx: Context | None) -> Optional[str]:
    if session_id:
        return session_id
    if ctx and getattr(ctx, "session_id", None):
        return ctx.session_id
    env_session = os.getenv("CODE_MEMORY_SESSION_ID")
    return env_session or None


def _schedule_post_insert(observation_id: int, *, content: str, metadata: Optional[dict]) -> None:
    if observation_id <= 0:
        return
    meta = metadata or {}

    def _run_completion():
        try:
            store.complete_memory_fields(observation_id, content=content)
        except Exception:
            logger.exception("bg.complete_memory_fields failed observation_id=%s", observation_id)

    def _run_ner():
        try:
            store.process_memory_ner(observation_id, content=content, metadata=meta)
        except Exception:
            logger.exception("bg.process_memory_ner failed observation_id=%s", observation_id)

    try:
        _BG_EXECUTOR.submit(_run_completion)
        _BG_EXECUTOR.submit(_run_ner)
    except Exception:
        logger.exception("Failed to schedule background tasks observation_id=%s", observation_id)


# -------------
# MCP tools
# -------------
def _insert_memory_impl(
    *,
    tool_name: str,
    content: str,
    session_id: Optional[str],
    kind: Optional[str],
    tags: Optional[str],
    priority: int,
    metadata_json: Optional[str],
    ctx: Context | None,
) -> dict:
    _log_tool_io(
        tool_name,
        "in",
        {
            "content": content,
            "session_id": session_id,
            "kind": kind,
            "tags": tags,
            "priority": priority,
            "metadata_json": metadata_json,
            "ctx_session_id": getattr(ctx, "session_id", None) if ctx else None,
        },
    )
    resolved_session_id = _resolve_session_id(session_id, ctx)
    if not resolved_session_id:
        out = {"status": "error", "error": "session_id is required", "tool": tool_name}
        _log_tool_io(tool_name, "out", out)
        return out

    metadata = json.loads(metadata_json) if metadata_json else None
    mem_id = store.insert_memory(
        content=content,
        session_id=resolved_session_id,
        kind=kind,
        tags=tags,
        priority=priority,
        metadata=metadata,
        ctx=ctx,
    )
    if mem_id == -1:
        out = {"status": "skipped"}
        _log_tool_io(tool_name, "out", out)
        return out

    _schedule_post_insert(mem_id, content=content, metadata=metadata)

    stored = store.get_observation(mem_id)
    stored_tags = stored.get("tags") if stored else tags
    stored_kind = stored.get("kind") if stored else kind
    stored_priority = stored.get("priority") if stored else priority
    stored_hash = stored.get("content_hash") if stored else hash_content(content)
    if ctx:
        ctx.info(f"Stored memory {mem_id}")
    logger.info("%s ok id=%s session_id=%s priority=%s", tool_name, mem_id, resolved_session_id, priority)
    out = {
        "id": mem_id,
        "session_id": resolved_session_id,
        "kind": stored_kind,
        "tags": stored_tags,
        "priority": stored_priority,
        "content_hash": stored_hash,
    }
    _log_tool_io(tool_name, "out", out)
    return out


@server.tool(
    description=(
        "Insert a memory entry.\n"
        "\n"
        "This is the preferred API for agents.\n"
        "\n"
        "Important behavior:\n"
        "- Writes the memory row immediately (fast).\n"
        "- Runs post-processing asynchronously (slow):\n"
        "  - entity/relation extraction (NER) and graph upserts\n"
        "  - tag completion when tags are missing\n"
        "\n"
        "Parameters:\n"
        "- content (str): the memory text to store.\n"
        "- session_id (str): required session scope.\n"
        "- kind (str|None): optional category (decision/bugfix/note/etc).\n"
        "- tags (str|None): comma-separated tags; recommended when you already know them.\n"
        "- priority (int): 1..5 (1 = highest importance).\n"
        "- metadata_json (str|None): JSON object encoded as a string.\n"
        "\n"
        "Examples:\n"
        "1) Minimal:\n"
        "{ \"content\": \"We decided to use libSQL-only.\", \"session_id\": \"ses_...\" }\n"
        "\n"
        "2) With tags + metadata:\n"
        "{\n"
        "  \"content\": \"Fix: set CODE_MEMORY_DB_URL to start the server.\",\n"
        "  \"session_id\": \"ses_...\",\n"
        "  \"kind\": \"bugfix\",\n"
        "  \"tags\": \"setup,libsql,windows\",\n"
        "  \"priority\": 2,\n"
        "  \"metadata_json\": \"{\\\"path\\\":\\\"src/code_memory/server.py\\\",\\\"ticket\\\":\\\"MEM-12\\\"}\"\n"
        "}\n"
        "\n"
        "Return (success): {\"id\":..., \"session_id\":..., \"kind\":..., \"tags\":..., \"priority\":..., \"content_hash\":...}\n"
        "Return (skipped): {\"status\":\"skipped\"}\n"
        "Return (error): {\"status\":\"error\",...}\n"
    )
)
def insert_memory(
    content: str,
    session_id: Optional[str] = None,
    kind: Optional[str] = None,
    tags: Optional[str] = None,
    priority: int = 3,
    metadata_json: Optional[str] = None,
    ctx: Context | None = None,
) -> dict:
    try:
        return _insert_memory_impl(
            tool_name="insert_memory",
            content=content,
            session_id=session_id,
            kind=kind,
            tags=tags,
            priority=priority,
            metadata_json=metadata_json,
            ctx=ctx,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool insert_memory failed")
        out = {"status": "error", "error": str(exc), "tool": "insert_memory"}
        _log_tool_io("insert_memory", "out", out)
        return out


@server.tool(
    description=(
        "Store a new memory entry. Deprecated: use insert_memory.\n"
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
        "- tags (str|None): Optional comma-separated tags. If omitted, the server auto-generates tags.\n"
        "  Examples: 'setup,windows,sqlite', 'rag,search,tuning'.\n"
        "- priority (int): 1..5 where 1 = most important. Affects reranking during search.\n"
        "- metadata_json (str|None): JSON object as a string. Stored as metadata.\n"
        "  Example: '{\"repo\":\"mcp-code-vector-memory-sql\",\"path\":\"src/code_memory/server.py\"}'.\n"
        "\n"
        "Return (success):\n"
        "{\n"
        "  \"id\": 123,\n"
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
    tags: Optional[str] = None,
    priority: int = 3,
    metadata_json: Optional[str] = None,
    ctx: Context | None = None,
) -> dict:
    try:
        logger.warning("Tool remember is deprecated; use insert_memory instead.")
        return _insert_memory_impl(
            tool_name="remember",
            content=content,
            session_id=session_id,
            kind=kind,
            tags=tags,
            priority=priority,
            metadata_json=metadata_json,
            ctx=ctx,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool remember failed")
        out = {"status": "error", "error": str(exc), "tool": "remember"}
        _log_tool_io("remember", "out", out)
        return out


@server.tool(
    description=(
        "Hybrid search over stored memories (semantic vector search + optional FTS re-rank).\n"
        "\n"
        "Session-aware ranking:\n"
        "- If session_id is provided (argument, ctx.session_id, or CODE_MEMORY_SESSION_ID), results from the same\n"
        "  session get a score bonus.\n"
        "\n"
        "Parameters:\n"
        "- query (str): Natural language or keyword query. Also used for FTS matching.\n"
        "- session_id (str|None): Optional session id for ranking boost.\n"
        "- limit (int): Number of results to return (top_k). Default comes from CODE_MEMORY_TOP_K.\n"
        "  Internally the server oversamples candidates by CODE_MEMORY_OVERSAMPLE_K before reranking.\n"
        "- top_p (float): Recency window for reranking (0 < top_p <= 1). Default from CODE_MEMORY_TOP_P.\n"
        "  top_p=1.0 keeps all candidates; top_p=0.6 keeps only the newest 60% of candidates (by created_at)\n"
        "  before final sorting. Use this to bias toward recent context.\n"
        "\n"
        "How results are ranked:\n"
        "- Starts from vector distance (lower is better).\n"
        "- Applies an FTS bonus when there is an FTS hit.\n"
        "- Applies a session bonus for same-session results.\n"
        "- Applies a priority penalty (priority 1 is favored over priority 5).\n"
        "- Applies a recency penalty (older is ranked lower).\n"
        "\n"
        "Return: list[dict] sorted by score (lower is better). Each item contains:\n"
        "- id, session_id, kind, content, content_hash, tags, priority, metadata (object), created_at\n"
        "- score (float): final reranked score (lower is better)\n"
        "- fts_hit (bool): whether the item matched via FTS\n"
        "\n"
        "Example return:\n"
        "[\n"
        "  {\n"
        "    \"id\": 101,\n"
        "    \"session_id\": \"s-123\",\n"
        "    \"kind\": \"decision\",\n"
        "    \"content_hash\": \"...\",\n"
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
        "- Exact identifiers/tags: include the exact token(s) in query (FTS searches content/tags/metadata).\n"
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
        _log_tool_io(
            "search_memory",
            "in",
            {
                "query": query,
                "session_id": session_id,
                "limit": limit,
                "top_p": top_p,
                "ctx_session_id": getattr(ctx, "session_id", None) if ctx else None,
            },
        )
        resolved_session_id = _resolve_session_id(session_id, ctx)
        limit = clamp_top_k(limit)
        top_p = clamp_top_p(top_p)
        out = store.search(query=query, session_id=resolved_session_id, limit=limit, top_p=top_p)
        _log_tool_io("search_memory", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool search_memory failed")
        out = {"status": "error", "error": str(exc), "tool": "search_memory"}
        _log_tool_io("search_memory", "out", out)
        return out


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
        "- id, session_id, kind, content, content_hash, tags, priority, metadata (object), created_at\n"
        "\n"
        "Example return:\n"
        "[{\"id\":1,\"session_id\":\"s-123\",\"content_hash\":\"...\",\"tags\":\"general\",\"priority\":3,\"created_at\":\"2026-01-04 12:34:56\"}]\n"
    )
)
def list_recent(limit: int = 20) -> List[dict]:
    try:
        _log_tool_io("list_recent", "in", {"limit": limit})
        out = store.recent(limit=limit)
        _log_tool_io("list_recent", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_recent failed")
        out = {"status": "error", "error": str(exc), "tool": "list_recent"}
        _log_tool_io("list_recent", "out", out)
        return out


@server.tool(
    description=(
        "List entities extracted from a single memory entry.\n"
        "\n"
        "Entity extraction is best-effort (LLM + regex fallback).\n"
        "\n"
        "Parameters:\n"
        "- memory_id (int): Memory id returned by remember().\n"
        "\n"
        "Return: list[dict] where each item contains:\n"
        "- name (str), entity_type (str), relation_type (str)\n"
        "\n"
        "Example return:\n"
        "[{\"name\":\"PostgreSQL\",\"entity_type\":\"technology\",\"relation_type\":\"mentions\"}]\n"
    )
)
def list_entities(memory_id: int) -> List[dict]:
    try:
        _log_tool_io("list_entities", "in", {"memory_id": memory_id})
        out = store.list_entities(memory_id)
        _log_tool_io("list_entities", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool list_entities failed")
        out = {"status": "error", "error": str(exc), "tool": "list_entities"}
        _log_tool_io("list_entities", "out", out)
        return out


@server.tool(
    description=(
        "Upsert (create or update) a knowledge-graph entity and attach observations.\n"
        "\n"
        "Notes:\n"
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
        _log_tool_io(
            "upsert_entity",
            "in",
            {
                "name": name,
                "entity_type": entity_type,
                "observations_json": observations_json,
                "memory_id": memory_id,
            },
        )
        observations = json.loads(observations_json) if observations_json else []
        if not isinstance(observations, list):
            observations = [str(observations)]
        entity_id = store.upsert_graph_entity(name, entity_type, observations, memory_id)
        out = {"status": "ok", "entity_id": entity_id}
        _log_tool_io("upsert_entity", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool upsert_entity failed")
        out = {"status": "error", "error": str(exc), "tool": "upsert_entity"}
        _log_tool_io("upsert_entity", "out", out)
        return out


@server.tool(
    description=(
        "Add a directed relation between two knowledge-graph entities.\n"
        "\n"
        "Notes:\n"
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
        _log_tool_io(
            "add_relation",
            "in",
            {
                "source": source,
                "target": target,
                "relation_type": relation_type,
                "memory_id": memory_id,
            },
        )
        rel_id = store.add_graph_relation(source, target, relation_type, memory_id)
        out = {"status": "ok", "relation_id": rel_id}
        _log_tool_io("add_relation", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool add_relation failed")
        out = {"status": "error", "error": str(exc), "tool": "add_relation"}
        _log_tool_io("add_relation", "out", out)
        return out


@server.tool(
    description=(
        "Fetch a knowledge-graph entity by name, including observations and relations.\n"
        "\n"
        "Notes:\n"
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
        _log_tool_io("get_entity", "in", {"name": name})
        out = store.get_graph_entity(name)
        _log_tool_io("get_entity", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_entity failed")
        out = {"status": "error", "error": str(exc), "tool": "get_entity"}
        _log_tool_io("get_entity", "out", out)
        return out


@server.tool(
    description=(
        "Return a context graph snapshot (entities + relations).\n"
        "\n"
        "If query is provided, performs a semantic search over graph entities (uses embeddings when enabled).\n"
        "If query is omitted, returns the most recent entities in the graph.\n"
        "\n"
        "Notes:\n"
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
        _log_tool_io("get_context_graph", "in", {"query": query, "limit": limit})
        if query:
            out = store.search_graph(query, limit=limit)
        else:
            out = store.read_graph(limit=limit)
        _log_tool_io("get_context_graph", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool get_context_graph failed")
        out = {"status": "error", "error": str(exc), "tool": "get_context_graph"}
        _log_tool_io("get_context_graph", "out", out)
        return out


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
        "  - 'rebuild_graph' (destructive): rebuild entities + relations from stored observations\n"
        "  - 'purge_all' (destructive): delete all observations\n"
        "  - 'purge_session' (destructive): delete observations for a specific session_id\n"
        "  - 'prune_older_than' (destructive): delete observations older than older_than_days\n"
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
        _log_tool_io(
            "maintenance",
            "in",
            {
                "action": action,
                "confirm": confirm,
                "session_id": session_id,
                "older_than_days": older_than_days,
            },
        )
        if action != "vacuum" and not confirm:
            out = {"status": "error", "error": "confirm=true required for destructive actions"}
            _log_tool_io("maintenance", "out", out)
            return out

        conn = _connect_db(load_vec=False)
        try:
            if action == "vacuum":
                conn.execute("VACUUM")
                out = {"status": "ok", "action": "vacuum"}
                _log_tool_io("maintenance", "out", out)
                return out
            if action == "rebuild_graph":
                # Unified strategy: keep existing memory observations, rebuild entities+relations from them.
                rows = _db_fetchall(
                    conn,
                    """
                    SELECT o.id, o.content, o.metadata, o.entity_id
                    FROM observations o
                    JOIN entities e ON o.entity_id = e.id
                    WHERE e.entity_type = 'memory'
                    ORDER BY o.id
                    """
                )

                conn.execute("DELETE FROM relations")
                conn.execute("DELETE FROM entities WHERE entity_type != 'memory'")
                _db_commit(conn)

                rebuilt_entities = 0
                rebuilt_relations = 0
                for obs_id, content, metadata, mem_entity_id in rows:
                    try:
                        meta = json.loads(metadata or "{}") if isinstance(metadata, str) else (metadata or {})
                    except Exception:
                        meta = {}
                    entity_path = meta.get("path") or meta.get("file_path") or meta.get("filepath")
                    try:
                        entity_path = str(entity_path) if entity_path else None
                    except Exception:
                        entity_path = None

                    graph = extract_graph(content or "", path=entity_path, use_llm=False)

                    for e in graph.get("entities", []) or []:
                        name = str(e.get("name") or "").strip()
                        et = str(e.get("type") or "other").strip()
                        if not name:
                            continue
                        row = _db_fetchone(conn, "SELECT id FROM entities WHERE name = ?", (name,))
                        if row:
                            ent_id = int(row[0])
                            conn.execute("UPDATE entities SET entity_type = ? WHERE id = ?", (et or "other", ent_id))
                        else:
                            cur = conn.execute("INSERT INTO entities (name, entity_type) VALUES (?, ?)", (name, et or "other"))
                            ent_id = _db_lastrowid(cur)
                        rebuilt_entities += 1
                        conn.execute(
                            "INSERT INTO relations (source_id, target_id, relation_type, observation_id) VALUES (?, ?, 'mentions', ?)",
                            (int(mem_entity_id), ent_id, int(obs_id)),
                        )
                        rebuilt_relations += 1

                    for r in graph.get("relations", []) or []:
                        src_name = str(r.get("source") or "").strip()
                        tgt_name = str(r.get("target") or "").strip()
                        rel_type = str(r.get("type") or "related_to").strip()
                        if not src_name or not tgt_name or not rel_type:
                            continue
                        src_row = _db_fetchone(conn, "SELECT id FROM entities WHERE name = ?", (src_name,))
                        if not src_row:
                            cur = conn.execute("INSERT INTO entities (name, entity_type) VALUES (?, 'other')", (src_name,))
                            src_id = _db_lastrowid(cur)
                        else:
                            src_id = int(src_row[0])
                        tgt_row = _db_fetchone(conn, "SELECT id FROM entities WHERE name = ?", (tgt_name,))
                        if not tgt_row:
                            cur = conn.execute("INSERT INTO entities (name, entity_type) VALUES (?, 'other')", (tgt_name,))
                            tgt_id = _db_lastrowid(cur)
                        else:
                            tgt_id = int(tgt_row[0])
                        conn.execute(
                            "INSERT INTO relations (source_id, target_id, relation_type, observation_id) VALUES (?, ?, ?, ?)",
                            (src_id, tgt_id, rel_type, int(obs_id)),
                        )
                        rebuilt_relations += 1

                _db_commit(conn)
                out = {
                    "status": "ok",
                    "action": "rebuild_graph",
                    "observations": len(rows),
                    "entities": rebuilt_entities,
                    "relations": rebuilt_relations,
                }
                _log_tool_io("maintenance", "out", out)
                return out

            ids = []
            if action == "purge_all":
                ids = [row[0] for row in _db_fetchall(conn, "SELECT id FROM observations")]
            elif action == "purge_session" and session_id:
                ids = [row[0] for row in _db_fetchall(conn, "SELECT id FROM observations WHERE session_id = ?", (session_id,))]
            elif action == "prune_older_than" and older_than_days is not None:
                cutoff = time.time() - (older_than_days * 86400)
                rows = _db_fetchall(conn, "SELECT id, created_at FROM observations")
                for mid, created_at in rows:
                    ts = parse_timestamp(created_at)
                    if ts and ts < cutoff:
                        ids.append(mid)
            else:
                return {"status": "error", "error": "invalid action or missing parameter"}

            if not ids:
                out = {"status": "ok", "deleted": 0}
                _log_tool_io("maintenance", "out", out)
                return out

            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM relations WHERE observation_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", ids)
            _db_commit(conn)
            out = {"status": "ok", "deleted": len(ids)}
            _log_tool_io("maintenance", "out", out)
            return out
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool maintenance failed")
        out = {"status": "error", "error": str(exc), "tool": "maintenance"}
        _log_tool_io("maintenance", "out", out)
        return out


@server.tool(
    description=(
        "Diagnostics for environment and DB.\n"
        "\n"
        "Use cases:\n"
        "- Confirm DB backend/url and defaults.\n"
        "- Confirm embedding model/dim, cache paths, and defaults.\n"
        "- Inspect table presence and approximate row counts.\n"
        "\n"
        "Return: a dict containing cwd, db config, embedding config, defaults, NER config, tables, and counts.\n"
    )
)
def diagnostics() -> dict:
    try:
        _log_tool_io("diagnostics", "in", {})
        conn = _connect_db(load_vec=False)
        try:
            tables = [r[0] for r in _db_fetchall(conn, "SELECT name FROM sqlite_master WHERE type='table'")]
            counts = {}
            for t in ("entities", "observations", "relations", "observations_fts"):
                if t in tables:
                    counts[t] = _db_fetchone(conn, f"SELECT COUNT(*) FROM {t}")[0]
        finally:
            conn.close()
        out = {
            "cwd": str(Path.cwd()),
            "db_backend": DB_BACKEND,
            "db_url": DB_URL,
            "model": EMBED_MODEL_NAME,
            "model_cache_dir": str(MODEL_CACHE_DIR),
            "defaults": {
                "top_k": DEFAULT_TOP_K,
                "top_p": DEFAULT_TOP_P,
                "recency_weight": RECENCY_WEIGHT,
                "priority_weight": PRIORITY_WEIGHT,
                "fts_bonus": FTS_BONUS,
                "oversample_k": OVERSAMPLE_K,
                "session_bonus": SESSION_BONUS,
                "cross_session_penalty": CROSS_SESSION_PENALTY,
            },
            "ner": {
                "enabled": _ner_enabled(),
                "model": str(get_setting("CODE_MEMORY_NER_MODEL", "") or ""),
                "ctx": NER_CTX,
                "threads": NER_THREADS,
                "max_tokens": NER_MAX_TOKENS,
                "temperature": NER_TEMPERATURE,
                "auto_install": NER_AUTO_INSTALL,
                "auto_download": os.getenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "1"),
            },
            "tables": tables,
            "counts": counts,
        }
        _log_tool_io("diagnostics", "out", out)
        return out
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool diagnostics failed")
        out = {"status": "error", "error": str(exc), "tool": "diagnostics"}
        _log_tool_io("diagnostics", "out", out)
        return out


@server.tool(
    description=(
        "Health check for the server.\n"
        "\n"
        "Return: a dict with status='ok' and key runtime configuration (db path, embedding model/dim,\n"
        "default search parameters, and NER configuration).\n"
    )
)
def health() -> dict:
    try:
        _log_tool_io("health", "in", {})
        info = {
            "status": "ok",
            "db_url": DB_URL,
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
            "session_bonus": SESSION_BONUS,
            "cross_session_penalty": CROSS_SESSION_PENALTY,
            "ner": {
                "enabled": _ner_enabled(),
                "model": str(get_setting("CODE_MEMORY_NER_MODEL", "") or ""),
                "ctx": NER_CTX,
                "threads": NER_THREADS,
                "max_tokens": NER_MAX_TOKENS,
                "temperature": NER_TEMPERATURE,
                "auto_install": NER_AUTO_INSTALL,
                "auto_download": os.getenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "1"),
            },
            "log_level": LOG_LEVEL,
            "log_file": str(LOG_FILE) if LOG_FILE else None,
            "workspace": str(ROOT),
        }
        logger.info("health check: %s", info)
        _log_tool_io("health", "out", info)
        return info
    except Exception as exc:  # pragma: no cover
        logger.exception("Tool health failed")
        out = {"status": "error", "error": str(exc), "tool": "health"}
        _log_tool_io("health", "out", out)
        return out


def main() -> None:
    try:
        server.run()
    except Exception as exc:
        logger.exception("Server startup/run failed: %s", exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
