import importlib
import os
import sys
from pathlib import Path


def _import_ner(tmp_path: Path):
    os.environ["CODE_MEMORY_DB_URL"] = f"file:{tmp_path / 'test.db'}"
    os.environ["CODE_MEMORY_DB_AUTH_TOKEN"] = ""
    os.environ["CODE_MEMORY_LOG_LEVEL"] = "ERROR"
    os.environ["CODE_MEMORY_WORKSPACE"] = str(tmp_path)
    os.environ["CODE_MEMORY_EMBED_DIM"] = "384"
    os.environ["CODE_MEMORY_AUTO_INSTALL"] = "0"
    os.environ["CODE_MEMORY_CONFIG_PATH"] = str(tmp_path / "no-config.jsonc")

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    if "code_memory.ner" in sys.modules:
        return importlib.reload(sys.modules["code_memory.ner"])
    return importlib.import_module("code_memory.ner")


def test_ner_model_uses_explicit_env_path(tmp_path, monkeypatch):
    ner = _import_ner(tmp_path)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CODE_MEMORY_NER_MODEL", str(model_path))
    ner.NER_MODEL = os.getenv("CODE_MEMORY_NER_MODEL", "").strip()

    monkeypatch.setattr(ner, "download_gguf_from_repo", lambda repo_id: str(tmp_path / "downloaded.gguf"))
    resolved = ner.ensure_ner_model_path()

    assert resolved == str(model_path)


def test_ner_model_auto_download_is_lazy(tmp_path, monkeypatch):
    ner = _import_ner(tmp_path)

    monkeypatch.delenv("CODE_MEMORY_NER_MODEL", raising=False)
    monkeypatch.setenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "1")
    ner.NER_MODEL = ""

    called = {"download": 0}

    def _fake_download(repo_id: str):
        called["download"] += 1
        return str(tmp_path / "downloaded.gguf")

    monkeypatch.setattr(ner, "download_gguf_from_repo", _fake_download)

    # Importing the module should not trigger downloads (lazy behavior).
    assert called["download"] == 0

    resolved = ner.ensure_ner_model_path()
    # Not configured -> disabled (no auto-discovery).
    assert resolved == ""
    assert ner.NER_MODEL == ""
    assert called["download"] == 0


def test_ner_model_auto_download_can_be_disabled(tmp_path, monkeypatch):
    ner = _import_ner(tmp_path)

    monkeypatch.delenv("CODE_MEMORY_NER_MODEL", raising=False)
    monkeypatch.setenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "0")
    ner.NER_MODEL = ""

    monkeypatch.setattr(ner, "download_gguf_from_repo", lambda repo_id: (_ for _ in ()).throw(RuntimeError("should not run")))

    resolved = ner.ensure_ner_model_path()
    assert resolved == ""


def test_ner_model_repo_id_downloads(tmp_path, monkeypatch):
    ner = _import_ner(tmp_path)

    monkeypatch.setenv("CODE_MEMORY_NER_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF")
    monkeypatch.setenv("CODE_MEMORY_NER_AUTO_DOWNLOAD", "1")
    ner.NER_MODEL = os.getenv("CODE_MEMORY_NER_MODEL", "").strip()

    monkeypatch.setattr(ner, "download_gguf_from_repo", lambda repo_id: str(tmp_path / "downloaded.gguf"))
    downloaded = tmp_path / "downloaded.gguf"
    downloaded.write_text("x", encoding="utf-8")

    resolved = ner.ensure_ner_model_path()
    assert resolved == str(downloaded)

