import importlib
import os
import sys
from pathlib import Path


def _import_server(tmp_path: Path):
    os.environ["CODE_MEMORY_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["CODE_MEMORY_DB_DIR"] = str(tmp_path)
    os.environ["CODE_MEMORY_ENABLE_VEC"] = "0"
    os.environ["CODE_MEMORY_ENABLE_FTS"] = "0"
    os.environ["CODE_MEMORY_ENABLE_GRAPH"] = "0"
    os.environ["CODE_MEMORY_LOG_LEVEL"] = "ERROR"
    os.environ["CODE_MEMORY_WORKSPACE"] = str(tmp_path)
    os.environ["CODE_MEMORY_EMBED_DIM"] = "384"
    os.environ["CODE_MEMORY_AUTO_INSTALL"] = "0"
    os.environ["CODE_MEMORY_CONFIG_PATH"] = str(tmp_path / "no-config.jsonc")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    if "code_memory.server" in sys.modules:
        return importlib.reload(sys.modules["code_memory.server"])
    return importlib.import_module("code_memory.server")


def test_summary_model_uses_explicit_env_path(tmp_path, monkeypatch):
    server = _import_server(tmp_path)

    monkeypatch.setenv("CODE_MEMORY_SUMMARY_MODEL", str(tmp_path / "model.gguf"))
    server.SUMMARY_MODEL = os.getenv("CODE_MEMORY_SUMMARY_MODEL", "").strip()

    monkeypatch.setattr(server, "_download_gguf_from_repo", lambda repo_id: str(tmp_path / "downloaded.gguf"))
    resolved = server._ensure_summary_model_path()

    assert resolved == str(tmp_path / "model.gguf")


def test_summary_model_auto_download_is_lazy(tmp_path, monkeypatch):
    server = _import_server(tmp_path)

    monkeypatch.delenv("CODE_MEMORY_SUMMARY_MODEL", raising=False)
    monkeypatch.setenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "1")
    server.SUMMARY_MODEL = ""

    called = {"download": 0}

    def _fake_download(repo_id: str):
        called["download"] += 1
        return str(tmp_path / "downloaded.gguf")

    monkeypatch.setattr(server, "_download_gguf_from_repo", _fake_download)

    # Importing the module should not trigger downloads (lazy behavior).
    assert called["download"] == 0

    resolved = server._ensure_summary_model_path()
    # Not configured -> disabled (no auto-discovery).
    assert resolved == ""
    assert server.SUMMARY_MODEL == ""
    assert called["download"] == 0


def test_summary_model_auto_download_can_be_disabled(tmp_path, monkeypatch):
    server = _import_server(tmp_path)

    monkeypatch.delenv("CODE_MEMORY_SUMMARY_MODEL", raising=False)
    monkeypatch.setenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "0")
    server.SUMMARY_MODEL = ""

    monkeypatch.setattr(server, "_download_gguf_from_repo", lambda repo_id: (_ for _ in ()).throw(RuntimeError("should not run")))

    resolved = server._ensure_summary_model_path()
    assert resolved == ""


def test_summary_model_repo_id_downloads(tmp_path, monkeypatch):
    server = _import_server(tmp_path)

    monkeypatch.setenv("CODE_MEMORY_SUMMARY_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
    monkeypatch.setenv("CODE_MEMORY_SUMMARY_AUTO_DOWNLOAD", "1")
    server.SUMMARY_MODEL = os.getenv("CODE_MEMORY_SUMMARY_MODEL", "").strip()

    monkeypatch.setattr(server, "_download_gguf_from_repo", lambda repo_id: str(tmp_path / "downloaded.gguf"))
    resolved = server._ensure_summary_model_path()

    assert resolved == str(tmp_path / "downloaded.gguf")
