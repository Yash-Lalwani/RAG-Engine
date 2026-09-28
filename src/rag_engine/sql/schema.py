from functools import lru_cache

import psycopg


@lru_cache
def describe_schema(database_url: str) -> str:
    with psycopg.connect(database_url) as conn:
        rows = conn.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """).fetchall()

    tables: dict[str, list[str]] = {}
    for table, col, dtype in rows:
        tables.setdefault(table, []).append(f"{col} ({dtype})")

    lines = ["Database schema:"]
    for table, cols in tables.items():
        lines.append(f"  {table}: {', '.join(cols)}")
    return "\n".join(lines)
