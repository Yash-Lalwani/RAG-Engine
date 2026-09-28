"""Check generated SQL before it runs: one read-only SELECT over allowlisted tables."""

import sqlglot
from sqlglot import exp

from rag_engine.models import EngineError

MAX_ROWS = 200

_CHANGES_DATA = (
    exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Grant, exp.Copy, exp.Command, exp.Set, exp.Transaction, exp.Commit,
)


def validate_sql(sql: str, allowed_tables: list[str]) -> str:
    """Return the query as safe SQL (with LIMIT added if missing), or raise EngineError."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except sqlglot.errors.ParseError as error:
        raise EngineError(f"The SQL could not be parsed: {str(error).splitlines()[0]}") from None
    if len(statements) != 1:
        raise EngineError("Exactly one SQL statement is allowed")

    query = statements[0]
    if not isinstance(query, exp.Select | exp.SetOperation):
        raise EngineError("Only SELECT queries are allowed")
    if next(query.find_all(*_CHANGES_DATA), None) is not None:
        raise EngineError("The query must not change data")
    for select in query.find_all(exp.Select):
        if select.args.get("into"):
            raise EngineError("SELECT ... INTO is not allowed")
        if select.args.get("locks"):
            raise EngineError("Locking clauses (FOR UPDATE / FOR SHARE) are not allowed")

    _check_tables(query, allowed_tables)
    if not query.args.get("limit"):
        query = query.limit(MAX_ROWS)
    return query.sql(dialect="postgres")


def _check_tables(query: exp.Expression, allowed_tables: list[str]) -> None:
    allowed = {table.lower() for table in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in query.find_all(exp.CTE)}
    for table in query.find_all(exp.Table):
        if _is_cte_reference(table, cte_names):
            continue
        name = table.name.lower()
        if table.catalog or table.db.lower() not in ("", "public") or name not in allowed:
            shown = table.sql(dialect="postgres") or "a table function"
            raise EngineError(
                f"The query reads {shown}, which is not an allowed table "
                f"({', '.join(sorted(allowed)) or 'none'})"
            )


def _is_cte_reference(table: exp.Table, cte_names: set[str]) -> bool:
    """True if the name means a CTE. Inside a non-recursive CTE of the same name, it means the
    real table (e.g. WITH pg_roles AS (SELECT * FROM pg_roles) reads the system catalog)."""
    name = table.name.lower()
    if table.db or table.catalog or name not in cte_names:
        return False
    cte = table.find_ancestor(exp.CTE)
    while cte is not None:
        if cte.alias_or_name.lower() == name:
            return bool(cte.parent and cte.parent.args.get("recursive"))
        cte = cte.find_ancestor(exp.CTE)
    return True
