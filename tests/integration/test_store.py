from pathlib import Path
import json

from tests.factories import build_memories, make_memory


def _load_samples() -> list[dict]:
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_memories.json"
    return json.loads(data_path.read_text(encoding="utf-8-sig"))


def test_add_and_search_fts(server_module):
    store = server_module.store
    sample = _load_samples()[0]
    mem_id = store.add(
        content=sample["content"],
        session_id=sample["session_id"],
        kind=sample["kind"],
        tags=sample["tags"],
        priority=sample["priority"],
        metadata=sample["metadata"],
        ctx=None,
    )
    assert mem_id != -1

    results = store.search(query="timeout", session_id=sample["session_id"], limit=5)
    assert any("timeout" in r["content"].lower() for r in results)


def test_session_scope(server_module):
    store = server_module.store
    store.add(**make_memory(1, session_id="S1", keyword="alpha").__dict__, ctx=None)
    store.add(**make_memory(2, session_id="S2", keyword="alpha").__dict__, ctx=None)

    results = store.search(query="alpha", session_id="S1", limit=5)
    assert results
    assert results[0]["session_id"] == "S1"
    assert any(r["session_id"] == "S2" for r in results)


def test_dedupe_recent(server_module):
    store = server_module.store
    content = "Same content for dedupe"
    first = store.add(content=content, session_id="S1", kind="note", tags="", priority=3, metadata=None)
    second = store.add(content=content, session_id="S1", kind="note", tags="", priority=3, metadata=None)
    assert first != -1
    assert second == -1


def test_fts_trigger_delete(server_module):
    store = server_module.store
    mem_id = store.add(**make_memory(3, session_id="S1", keyword="beta").__dict__, ctx=None)
    assert mem_id != -1

    conn = server_module._connect_db()
    try:
        fts_before = conn.execute("SELECT COUNT(*) FROM observations_fts").rows[0][0]
        conn.execute("DELETE FROM observations WHERE id = ?", (mem_id,))
        fts_after = conn.execute("SELECT COUNT(*) FROM observations_fts").rows[0][0]
    finally:
        conn.close()

    assert fts_after == fts_before - 1


def test_recent_fallback(server_module):
    store = server_module.store
    for mem in build_memories(count=3, session_id="S1"):
        store.add(**mem.__dict__, ctx=None)

    results = store.search(query="no_hit_token", session_id="S1", limit=5)
    assert len(results) >= 1
