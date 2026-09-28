import re


def is_select_only(sql: str) -> bool:
    """Return True if the SQL is a SELECT statement only."""
    cleaned = sql.strip().lower()
    if not cleaned.startswith("select"):
        return False
    forbidden = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "grant", "revoke"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", cleaned):
            return False
    return True
