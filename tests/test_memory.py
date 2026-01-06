"""
Script simples de smoke test para o servidor code-memory.
Executa chamadas diretas as tools (como funcoes) para validar fluxo basico.

Uso:
    python tests/test_memory.py
"""

from pathlib import Path
import json
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_main():
    from code_memory import server as main

    return main


def reset_db(main):
    # libsql "file:" points at a local sqlite-compatible file managed by libsql-client.
    # For this smoke test, just use a unique file in the repo and delete it if it exists.
    url = os.getenv("CODE_MEMORY_DB_URL", "")
    if url.startswith("file:"):
        p = Path(url[len("file:") :])
        if p.exists():
            p.unlink()


def run():
    print(":: Resetando banco...")
    os.environ.setdefault("CODE_MEMORY_DB_URL", f"file:{(ROOT / 'tests' / '.tmp-smoke.db')}")
    os.environ.setdefault("CODE_MEMORY_DB_AUTH_TOKEN", "")
    main = _load_main()
    reset_db(main)

    # Recarrega o store para recriar schema
    main.store = main.MemoryStore()

    print(":: Chamando add direto (espera sucesso)...")
    r1_id = main.store.add(
        content="Primeiro teste de memoria",
        session_id="S1",
        kind="test",
        tags="test",
        priority=2,
        metadata={"origin": "test_script"},
        ctx=None,
    )
    print("add ->", r1_id)
    assert r1_id != -1, "add nao retornou id"

    print(":: Chamando add duplicado (espera skip via -1)...")
    r2_id = main.store.add(
        content="Primeiro teste de memoria",
        session_id="S1",
        kind="test",
        tags="test",
        priority=2,
        metadata={"origin": "test_script"},
        ctx=None,
    )
    print("add dup ->", r2_id)
    assert r2_id == -1, "duplicado deveria ser -1 (skip)"

    print(":: search_memory por 'Primeiro' (espera 1 resultado)...")
    s1 = main.store.search(query="Primeiro", session_id="S1", limit=5)
    print("search_memory ->", s1)
    assert len(s1) >= 1, "search_memory deveria retornar ao menos 1"

    print(":: list_recent (espera 1 resultado)...")
    lr = main.store.recent(limit=5)
    print("list_recent ->", lr)
    assert len(lr) >= 1, "list_recent deveria retornar ao menos 1"

    print(":: list_entities do primeiro id (pode ser vazio)...")
    entities = main.store.list_entities(observation_id=r1_id)
    print("list_entities ->", entities)

    print(":: upsert_graph_entity + add_graph_relation...")
    e1 = main.store.upsert_graph_entity(
        name="UserService",
        entity_type="class",
        observations=["UserService handles auth logic"],
        memory_id=r1_id,
    )
    e2 = main.store.upsert_graph_entity(
        name="AuthController",
        entity_type="class",
        observations=["AuthController calls UserService"],
        memory_id=r1_id,
    )
    rel_id = main.store.add_graph_relation("AuthController", "UserService", "calls", memory_id=r1_id)
    print("graph entity ids ->", e1, e2, "relation ->", rel_id)

    print(":: get_context_graph (semantic)...")
    graph = main.store.search_graph(query="auth", limit=5)
    print("graph ->", graph)

    print(":: health (dados diretos)...")
    h = {
        "status": "ok",
        "db_url": os.getenv("CODE_MEMORY_DB_URL"),
        "embedding_dim": main.store.embedder.dim,
        "model": main.EMBED_MODEL_NAME,
        "model_cache_dir": str(main.MODEL_CACHE_DIR),
        "tree_sitter": main.TREE_SITTER_AVAILABLE,
    }
    print("health ->", h)

    print("\n[OK] Smoke test finalizado com sucesso.")


if __name__ == "__main__":
    run()
