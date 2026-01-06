from __future__ import annotations

from typing import Optional

try:
    import libsql_client
except Exception:  # pragma: no cover
    libsql_client = None


def connect_db(
    *,
    enable_vec: bool,
    load_vec: bool = False,
    db_url: str,
    db_auth_token: Optional[str] = None,
):
    if libsql_client is None:
        raise RuntimeError("libsql-client is not installed; install libsql-client to use CODE_MEMORY_DB_URL")
    # libsql-client returns a synchronous client whose `.execute(sql, args)` matches our usage well.
    return libsql_client.create_client_sync(db_url, auth_token=db_auth_token or None)
