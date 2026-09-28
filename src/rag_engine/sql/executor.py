"""run_sql(): execute validated SQL as the read-only role, with a timeout and a row limit."""

import datetime
import decimal
import uuid
from typing import Any

import psycopg

from rag_engine.cache.keys import SQL_RESULT_TIER, SQL_RESULT_TTL, sql_result_key
from rag_engine.cache.store import cache
from rag_engine.models import EngineError, SqlResult
from rag_engine.sql.safety import MAX_ROWS, validate_sql
from rag_engine.sql.schema import database_url

STATEMENT_TIMEOUT_MS = 5000


def run_sql(sql: str, database: str, allowed_tables: list[str]) -> SqlResult:
    safe_sql = validate_sql(sql, allowed_tables)
    url = database_url(database)
    key = sql_result_key(database, safe_sql)
    cached = cache.get(SQL_RESULT_TIER, key)
    if cached is not None:
        return SqlResult.model_validate_json(cached)

    try:
        with psycopg.connect(
            url, connect_timeout=5, options=f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"
        ) as conn:
            conn.read_only = True
            cursor = conn.execute(safe_sql)
            columns = [column.name for column in cursor.description or []]
            rows = cursor.fetchmany(MAX_ROWS + 1)
    except psycopg.errors.QueryCanceled:
        raise EngineError(f"The SQL query timed out after {STATEMENT_TIMEOUT_MS // 1000} s") from None
    except psycopg.Error as error:
        message = error.diag.message_primary if error.diag else None
        raise EngineError(f"The SQL query failed: {message or error}") from None

    result = SqlResult(
        sql=safe_sql,
        columns=columns,
        rows=[{c: serialize_value(v) for c, v in zip(columns, row, strict=True)} for row in rows[:MAX_ROWS]],
        row_count=min(len(rows), MAX_ROWS),
        truncated=len(rows) > MAX_ROWS,
    )
    cache.set(SQL_RESULT_TIER, key, result.model_dump_json(), SQL_RESULT_TTL)
    return result


def serialize_value(value: Any) -> Any:
    """Turn database values into JSON-friendly ones."""
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
