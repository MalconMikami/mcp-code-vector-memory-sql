from __future__ import annotations

from typing import Optional

import sqlite_vec

sqlite3 = None


def _lazy_sqlite():
    global sqlite3
    if sqlite3 is None:
        import sqlite3 as _sqlite3

        sqlite3 = _sqlite3
    return sqlite3


def apply_pragmas(conn) -> None:
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA page_size=8192")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.execute("PRAGMA optimize")


def connect_db(
    *,
    db_path,
    enable_vec: bool,
    load_vec: bool = False,
):
    conn = _lazy_sqlite().connect(db_path)
    apply_pragmas(conn)
    if load_vec and enable_vec:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    return conn

