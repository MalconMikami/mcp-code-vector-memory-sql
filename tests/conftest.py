from pathlib import Path
import importlib
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def server_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE_MEMORY_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("CODE_MEMORY_DB_DIR", str(tmp_path))
    monkeypatch.setenv("CODE_MEMORY_ENABLE_VEC", "0")
    monkeypatch.setenv("CODE_MEMORY_ENABLE_FTS", "1")
    monkeypatch.setenv("CODE_MEMORY_ENABLE_GRAPH", "0")
    monkeypatch.setenv("CODE_MEMORY_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("CODE_MEMORY_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("CODE_MEMORY_EMBED_DIM", "384")
    monkeypatch.setenv("CODE_MEMORY_AUTO_INSTALL", "0")

    mod = sys.modules.get("code_memory.server")
    if isinstance(mod, types.ModuleType):
        server = importlib.reload(mod)
    else:
        server = importlib.import_module("code_memory.server")

    server.store = server.MemoryStore(server.DB_PATH, embedder=server.EmbeddingModel(embed_dim=server.EMBED_DIM))
    return server
