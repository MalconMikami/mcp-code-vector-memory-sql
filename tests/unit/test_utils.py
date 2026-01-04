from datetime import datetime, timedelta


def test_looks_sensitive(server_module):
    assert server_module._looks_sensitive("api_key=abcd1234abcd1234") is True
    assert server_module._looks_sensitive("secret=abcdefghijkl") is True
    assert server_module._looks_sensitive("password=123456") is True
    assert server_module._looks_sensitive("all good here") is False


def test_clamp_priority(server_module):
    assert server_module._clamp_priority(1) == 1
    assert server_module._clamp_priority(5) == 5
    assert server_module._clamp_priority(0) == 1
    assert server_module._clamp_priority(10) == 5
    assert server_module._clamp_priority("3") == 3


def test_clamp_top_k_and_p(server_module):
    assert server_module._clamp_top_k(0) == 1
    assert server_module._clamp_top_k(3) == 3
    assert server_module._clamp_top_p(0) == 1.0
    assert server_module._clamp_top_p(0.5) == 0.5
    assert server_module._clamp_top_p(5.0) == 1.0


def test_apply_recency_filter(server_module):
    base = datetime(2024, 1, 1, 12, 0, 0)
    items = []
    for i in range(4):
        stamp = (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        items.append({"id": i + 1, "created_at": stamp})

    filtered = server_module._apply_recency_filter(items, top_p=0.5)
    assert len(filtered) == 2
    ids = {r["id"] for r in filtered}
    assert ids == {3, 4}


def test_hash_content_is_stable(server_module):
    h1 = server_module._hash_content("same content")
    h2 = server_module._hash_content("same content")
    assert h1 == h2
