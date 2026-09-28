import datetime
import decimal
import uuid
from typing import Any

import psycopg

from rag_engine.cache.store import query_cache
from rag_engine.config import settings
from rag_engine.sql.safety import is_select_only


def serialize_value(value: Any) -> Any:
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return str(value)
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: serialize_value(v) for k, v in row.items()}


def execute_sql(sql: str) -> list[dict]:
    if not is_select_only(sql):
        raise ValueError("Only SELECT statements are allowed")

    cached = query_cache.get_sql_result(sql)
    if cached is not None:
        return cached

    with psycopg.connect(settings.database_url) as conn:
        cur = conn.execute(sql)
        columns = [desc.name for desc in cur.description] if cur.description else []
        rows = cur.fetchall()

    result = [_serialize_row(dict(zip(columns, row, strict=True))) for row in rows]
    query_cache.set_sql_result(sql, result)
    return result
