# libSQL local (sqld) backend

This project can run against a libSQL server using the Python `libsql-client`.
This is useful when you want:

- a single-process “server DB” (still on your machine) that multiple MCP clients can share
- libSQL vector functions (`vector32`, `vector_distance_cos`, `libsql_vector_idx`) for semantic search

## Configure

Set `CODE_MEMORY_DB_URL` to your libSQL server URL:

```json
{
  "CODE_MEMORY_DB_URL": "libsql://127.0.0.1:8080",
  "CODE_MEMORY_DB_AUTH_TOKEN": ""
}
```

Notes:

- `CODE_MEMORY_DB_URL` is required by this project.
- Vector search uses libSQL vector functions.

## Running `sqld` locally on Windows

`sqld` is typically distributed for Linux. On Windows the most reliable options are:

- **WSL2**: run `sqld` in WSL and expose it on `127.0.0.1:<port>`
- **Docker Desktop**: run a Linux container exposing `8080` to localhost

Once `sqld` is listening on localhost, the MCP server can connect using `CODE_MEMORY_DB_URL=libsql://127.0.0.1:8080`.
