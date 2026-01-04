# Operacao

## Logs e diagnostico

- `CODE_MEMORY_LOG_LEVEL` controla o nivel de log
- `CODE_MEMORY_LOG_DIR` ou `CODE_MEMORY_LOG_FILE` direcionam logs
- `diagnostics()` retorna flags, defaults e contagem de tabelas
- `health()` retorna estado geral e configuracoes efetivas

## Manutencao

Tool `maintenance(action, confirm, session_id, older_than_days)`:

- `vacuum`: executa VACUUM (nao requer confirm)
- `purge_all`: remove todas as memorias (confirm=true)
- `purge_session`: remove por session_id (confirm=true)
- `prune_older_than`: remove por idade em dias (confirm=true)

## Privacidade e dedupe

- conteudo sensivel e ignorado por padrao (tokens, senhas, secrets)
- duplicados recentes sao filtrados por hash

## Troubleshooting

- erro de dimensao do embedding: defina `CODE_MEMORY_EMBED_DIM` correto e recrie o banco
- summary nao ativa: verifique `CODE_MEMORY_SUMMARY_MODEL` e instalacao do llama-cpp-python
- erro ao carregar sqlite-vec: confirme a dependencia `sqlite-vec` instalada
