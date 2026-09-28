"""State of one ask() run. Every key a node returns must be declared here: LangGraph silently
drops undeclared keys. Values are plain JSON data so checkpoints stay simple."""

from operator import add
from typing import Annotated, Any, TypedDict


def add_numbers(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer that sums nested dicts of numbers, e.g. token counts from several nodes."""
    merged = dict(left or {})
    for key, value in (right or {}).items():
        if isinstance(value, dict):
            merged[key] = add_numbers(merged.get(key, {}), value)
        else:
            merged[key] = merged.get(key, 0) + value
    return merged


class AskState(TypedDict, total=False):
    # Input, set by ask()
    query_id: str
    caller: str
    created_at: str
    collection_id: str
    question: str
    options: dict[str, Any]
    filters: dict[str, Any]
    settings: dict[str, Any]

    # Routing and retrieval
    intent: str
    search_query: str
    chunks: list[dict[str, Any]]
    search_info: dict[str, Any]

    # Text2SQL
    sql: str | None
    sql_explanation: str | None
    sql_error: str | None
    sql_approved: bool | None
    sql_rows: list[dict[str, Any]]
    sql_row_count: int
    sql_truncated: bool

    # Answer, Self-RAG and verification
    answer: dict[str, Any]
    answer_check: dict[str, Any] | None
    retries: int
    first_attempt: dict[str, Any] | None
    self_rag: dict[str, Any] | None
    verification: dict[str, Any] | None

    # Output, set by finalize
    status: str
    message: str | None
    final_answer: str
    final_statements: list[dict[str, Any]]
    sources: list[dict[str, Any]]

    # Bookkeeping, added up across nodes (and across the pause for approval)
    warnings: Annotated[list[str], add]
    usage: Annotated[dict[str, int], add_numbers]
    cache: Annotated[dict[str, dict[str, int]], add_numbers]
    timings_ms: Annotated[dict[str, float], add_numbers]
