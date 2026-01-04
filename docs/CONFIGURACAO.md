# Configuracao

Esta pagina detalha todas as variaveis de ambiente aceitas pelo servidor.
Para um resumo rapido, veja `README.md`.

## Armazenamento e workspace

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_DB_PATH | caminho completo do SQLite | `CODE_MEMORY_DB_DIR/code_memory.db` |
| CODE_MEMORY_DB_DIR | diretorio para o banco | cwd |
| CODE_MEMORY_WORKSPACE | raiz para resources MCP | cwd |

Nota: `CODE_MEMORY_DB_PATH` tem prioridade sobre `CODE_MEMORY_DB_DIR`.

## Embeddings

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_EMBED_MODEL | nome do modelo | `BAAI/bge-small-en-v1.5` |
| CODE_MEMORY_EMBED_DIM | dimensao do embedding (obrigatoria em modelo custom) | 384 |
| CODE_MEMORY_MODEL_DIR | cache local de modelos | `~/.cache/code-memory` |

## Busca e ranking

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_TOP_K | limite base por busca | 12 |
| CODE_MEMORY_TOP_P | filtro por recencia (0-1) | 0.6 |
| CODE_MEMORY_OVERSAMPLE_K | multiplicador de candidatos | 4 |
| CODE_MEMORY_PRIORITY_WEIGHT | peso da prioridade no re-rank | 0.15 |
| CODE_MEMORY_RECENCY_WEIGHT | peso de recencia no re-rank | 0.2 |
| CODE_MEMORY_FTS_BONUS | bonus para hits no FTS | 0.1 |

## Flags de recursos

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_ENABLE_VEC | liga/desliga vetores | 1 |
| CODE_MEMORY_ENABLE_FTS | liga/desliga FTS | 1 |
| CODE_MEMORY_ENABLE_GRAPH | liga/desliga grafo | 0 |

## Resumo local (GGUF)

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_SUMMARY_MODEL | caminho para GGUF (obrigatorio para habilitar) | vazio |
| CODE_MEMORY_SUMMARY_CTX | contexto do modelo | 2048 |
| CODE_MEMORY_SUMMARY_THREADS | threads | 4 |
| CODE_MEMORY_SUMMARY_MAX_TOKENS | max tokens por resumo | 200 |
| CODE_MEMORY_SUMMARY_TEMPERATURE | temperatura | 0.2 |
| CODE_MEMORY_SUMMARY_TOP_P | top_p | 0.9 |
| CODE_MEMORY_SUMMARY_REPEAT_PENALTY | repeat penalty | 1.05 |
| CODE_MEMORY_SUMMARY_GPU_LAYERS | camadas GPU | 0 |
| CODE_MEMORY_SUMMARY_MAX_CHARS | max chars retornados | 300 |
| CODE_MEMORY_SUMMARY_PROMPT | prompt custom | vazio |
| CODE_MEMORY_AUTO_INSTALL | auto install llama-cpp-python | 1 |
| CODE_MEMORY_PIP_ARGS | flags extras para pip | vazio |

## Logs e diagnostico

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_LOG_LEVEL | nivel de log (INFO, DEBUG, etc) | INFO |
| CODE_MEMORY_LOG_DIR | diretorio de logs | vazio |
| CODE_MEMORY_LOG_FILE | arquivo de log | vazio |

## Sessao

| Variavel | Descricao | Padrao |
| --- | --- | --- |
| CODE_MEMORY_SESSION_ID | fallback de session_id | vazio |

## Exemplos

Config basica para reduzir ruido e manter o DB no workspace:

```bash
export CODE_MEMORY_DB_DIR="C:/repo"
export CODE_MEMORY_LOG_DIR="C:/repo/.logs"
export CODE_MEMORY_TOP_K="8"
```

Habilitar grafo e resumo local:

```bash
export CODE_MEMORY_ENABLE_GRAPH="1"
export CODE_MEMORY_SUMMARY_MODEL="C:/models/llama-3.gguf"
```

Observacao: ao trocar o modelo de embeddings, defina `CODE_MEMORY_EMBED_DIM`
e recrie o banco caso haja incompatibilidade.
