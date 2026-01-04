# Arquitetura

## Componentes

- FastMCP: define tools e resources MCP
- MemoryStore: camada de persistencia em SQLite
- fastembed: gera embeddings em CPU
- sqlite-vec: tabela vetorial para busca semantica
- FTS5: indice textual opcional para re-rank
- Graph (opcional): entidades, observacoes e relacoes
- Summary (opcional): GGUF via llama-cpp-python

## Fluxo do remember

1. resolve session_id (input, contexto MCP ou env)
2. filtra conteudo sensivel e deduplica por hash recente
3. opcionalmente gera resumo local e tags simples
4. grava em `memories` e gera embedding para `vec_memories`
5. atualiza `memories_fts` via triggers (se habilitado)
6. extrai entidades e atualiza grafo (se habilitado)

## Fluxo do search_memory

1. gera embedding do query
2. consulta `vec_memories` e recupera candidatos (oversample)
3. aplica bonus de FTS (se habilitado)
4. re-rank com recencia e prioridade
5. aplica `top_p` para reduzir a lista pelo fator de recencia

## Fluxo do get_context_graph

- sem query: retorna grafo completo
- com query: busca semantica no grafo (se vetores habilitados)

## Schema (resumo)

- `memories`: conteudo, summary, tags, priority, session_id, metadata, created_at
- `vec_memories`: embedding vetorial
- `memories_fts`: indice FTS5 de `memories`
- `entities`: entidades extraidas por regex/tree-sitter
- `graph_entities`, `graph_observations`, `graph_relations`: grafo (opcional)

## Resources MCP

- `resource://workspace`: leitura do workspace
- `resource://readme`: README do projeto
