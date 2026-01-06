def test_remember_requires_session_id(server_module):
    result = server_module.remember.fn(content="no session")
    assert result["status"] == "error"
    assert "session_id" in result["error"]


def test_remember_and_search_tools(server_module):
    result = server_module.remember.fn(
        content="Store this note about search keyword zebra",
        session_id="S1",
        kind="note",
        tags="test",
        priority=2,
        metadata_json=None,
        ctx=None,
    )
    assert result.get("id")

    hits = server_module.search_memory.fn(query="zebra", session_id="S1", limit=5, top_p=1.0, ctx=None)
    assert isinstance(hits, list)
    assert any("zebra" in h["content"].lower() for h in hits)


def test_maintenance_confirm_required(server_module):
    res = server_module.maintenance.fn(action="purge_all", confirm=False)
    assert res["status"] == "error"
