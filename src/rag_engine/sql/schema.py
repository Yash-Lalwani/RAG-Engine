"""Describe only the allowlisted tables to the LLM: columns, types and foreign keys."""

from functools import lru_cache

import psycopg

from rag_engine.config import settings
from rag_engine.models import EngineError

_COLUMNS = """
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = ANY(%s)
ORDER BY table_name, ordinal_position
"""

_FOREIGN_KEYS = """
SELECT conrelid::regclass::text AS table_name, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE contype = 'f'
  AND conrelid::regclass::text = ANY(%s)
  AND confrelid::regclass::text = ANY(%s)
ORDER BY 1, 2
"""


def database_url(database: str) -> str:
    url = settings.sql_database_urls.get(database)
    if url is None:
        raise EngineError(f"SQL database {database!r} is not configured in SQL_DATABASES")
    return url


@lru_cache
def describe_schema(database: str, allowed_tables: tuple[str, ...]) -> str:
    """Pass allowed_tables as a sorted tuple (it is part of the cache key)."""
    tables = list(allowed_tables)
    with psycopg.connect(database_url(database), connect_timeout=5) as conn:
        columns = conn.execute(_COLUMNS, (tables,)).fetchall()
        foreign_keys = conn.execute(_FOREIGN_KEYS, (tables, tables)).fetchall()

    by_table: dict[str, list[str]] = {}
    for table, column, data_type in columns:
        by_table.setdefault(table, []).append(f"{column} {data_type}")
    missing = sorted(set(tables) - set(by_table))
    if missing:
        raise EngineError(f"Allowed table(s) not found in {database!r}: {', '.join(missing)}")

    lines = [f"{table}({', '.join(cols)})" for table, cols in by_table.items()]
    lines += [f"-- {table}: {definition}" for table, definition in foreign_keys]
    return "\n".join(lines)
