"""Graph nodes. Each node calls one core function and returns only the state fields it changes."""

import time
from collections.abc import Callable
from typing import Any

from langgraph.types import interrupt

from rag_engine.cache.store import track_cache
from rag_engine.generation.generator import citation_passages, generate_answer, render_answer
from rag_engine.grading.citations import verify_citations
from rag_engine.grading.self_rag import keep_retry, rewrite_query, self_check
from rag_engine.graph.state import AskState
from rag_engine.llm import track_usage
from rag_engine.models import CitedAnswer, CollectionSettings, EngineError, SearchChunk, SqlResult
from rag_engine.retrieval.search import search
from rag_engine.routing.intent_router import classify_intent
from rag_engine.sql.executor import run_sql
from rag_engine.sql.safety import validate_sql
from rag_engine.sql.text_to_sql import generate_sql

NOT_APPROVED = "Query not approved."
STATE_KEYS = set(AskState.__annotations__)

Node = Callable[[AskState], dict[str, Any]]


def tracked(name: str, node: Node) -> Node:
    """Record the node's token usage, cache hits and time in the state, and refuse undeclared keys
    (LangGraph would silently drop them)."""

    def run(state: AskState) -> dict[str, Any]:
        started = time.perf_counter()
        with track_usage() as usage, track_cache() as cache_counts:
            update = dict(node(state))
        undeclared = set(update) - STATE_KEYS
        if undeclared:
            raise RuntimeError(f"Node {name!r} returned undeclared state keys: {sorted(undeclared)}")
        update["usage"] = usage.model_dump()
        update["cache"] = cache_counts
        update["timings_ms"] = {name: round((time.perf_counter() - started) * 1000, 1)}
        return update

    return run


def route_intent_node(state: AskState) -> dict[str, Any]:
    try:
        intent = classify_intent(state["question"], state["collection_id"], _settings(state))
    except Exception as exc:
        return {"intent": "rag", "warnings": [f"Intent routing failed, answering from documents: {exc}"]}
    return {"intent": intent}


def retrieve_node(state: AskState) -> dict[str, Any]:
    result = search(
        state["collection_id"],
        state["search_query"],
        filters=state.get("filters"),
        options=state.get("options"),
    )
    return {
        "chunks": [chunk.model_dump() for chunk in result.chunks],
        "search_info": result.info.model_dump(),
        "warnings": result.info.warnings,
    }


def generate_node(state: AskState) -> dict[str, Any]:
    answer = generate_answer(
        state["question"],  # always the original question, also on the Self-RAG retry
        _chunks(state),
        _sql_result(state),
        _settings(state).domain_description,
    )
    return {"answer": answer.model_dump()}


def self_check_node(state: AskState) -> dict[str, Any]:
    chunks = _chunks(state)
    answer_text, _ = render_answer(_answer(state).statements, chunks)
    try:
        check = self_check(state["question"], answer_text, chunks).model_dump()
    except Exception as exc:
        return {"answer_check": None, "warnings": [f"Self-check failed, no retry: {exc}"]}

    first = state.get("first_attempt")
    if first is None:
        return {"answer_check": check, "self_rag": {"first_score": check["score"]}}

    info = {**state["self_rag"], "retried": True, "retry_score": check["score"]}
    if keep_retry(first["answer_check"]["score"], check["score"]):
        return {"answer_check": check, "self_rag": {**info, "kept": "retry"}}
    return {**first, "self_rag": {**info, "kept": "first"}}


def rewrite_node(state: AskState) -> dict[str, Any]:
    try:
        query = rewrite_query(
            state["question"], state["answer_check"]["reason"], _settings(state).domain_description
        )
    except Exception as exc:
        return {"retries": 1, "warnings": [f"Query rewrite failed, no retry: {exc}"]}
    first = {key: state.get(key) for key in ("answer", "chunks", "search_info", "answer_check")}
    return {
        "retries": 1,
        "search_query": query,
        "first_attempt": first,
        "self_rag": {**state["self_rag"], "retry_query": query},
    }


def generate_sql_node(state: AskState) -> dict[str, Any]:
    settings = _settings(state)
    try:
        draft = generate_sql(
            state["question"],
            settings.sql_database,
            settings.sql_allowed_tables,
            settings.domain_description,
        )
    except Exception as exc:
        return _sql_failed(state, f"SQL generation failed: {exc}")
    return {"sql": draft.sql, "sql_explanation": draft.explanation}


def validate_sql_node(state: AskState) -> dict[str, Any]:
    if state.get("sql_error"):
        return {}
    try:
        safe_sql = validate_sql(state["sql"], _settings(state).sql_allowed_tables)
    except EngineError as exc:
        return _sql_failed(state, f"The generated SQL was rejected: {exc}")
    return {"sql": safe_sql}


def approve_node(state: AskState) -> dict[str, Any]:
    """Pause here until approve_sql() resumes the run with {"approved": True/False}."""
    decision = interrupt({"sql": state["sql"], "explanation": state.get("sql_explanation")})
    approved = isinstance(decision, dict) and decision.get("approved") is True
    update: dict[str, Any] = {"sql_approved": approved}
    if not approved and state["intent"] == "hybrid":
        update["warnings"] = ["The SQL was not approved. Answered from documents only."]
    return update


def execute_sql_node(state: AskState) -> dict[str, Any]:
    settings = _settings(state)
    try:
        result = run_sql(state["sql"], settings.sql_database, settings.sql_allowed_tables)
    except EngineError as exc:
        return _sql_failed(state, str(exc))
    return {
        "sql": result.sql,
        "sql_rows": result.rows,
        "sql_row_count": result.row_count,
        "sql_truncated": result.truncated,
    }


def verify_node(state: AskState) -> dict[str, Any]:
    result = verify_citations(
        _answer(state).statements,
        citation_passages(_chunks(state), _sql_result(state)),
        strict=_settings(state).citation_mode == "strict",
    )
    return {"verification": result.model_dump(), "warnings": result.warnings}


def finalize_node(state: AskState) -> dict[str, Any]:
    if state["intent"] == "sql" and state.get("sql_error"):
        return _final("error", state["sql_error"])
    if state["intent"] == "sql" and state.get("sql_approved") is False:
        return _final("completed", NOT_APPROVED, NOT_APPROVED)

    verification = state.get("verification")
    statements = CitedAnswer.model_validate(
        {"statements": verification["statements"], "insufficient_context": False}
        if verification
        else state["answer"]
    ).statements
    text, sources = render_answer(statements, _chunks(state))
    return {
        **_final("completed", None, text),
        "final_statements": [s.model_dump() for s in statements],
        "sources": [s.model_dump() for s in sources],
    }


def _final(status: str, message: str | None, text: str = "") -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "final_answer": text,
        "final_statements": [],
        "sources": [],
    }


def _sql_failed(state: AskState, message: str) -> dict[str, Any]:
    update: dict[str, Any] = {"sql_error": message}
    if state["intent"] == "hybrid":
        update["warnings"] = [f"{message} Answered from documents only."]
    return update


def _settings(state: AskState) -> CollectionSettings:
    return CollectionSettings.model_validate(state["settings"])


def _chunks(state: AskState) -> list[SearchChunk]:
    return [SearchChunk.model_validate(chunk) for chunk in state.get("chunks") or []]


def _sql_result(state: AskState) -> SqlResult | None:
    """The executed SQL result, or None if SQL did not run (rejected, failed or rag intent)."""
    if "sql_rows" not in state:
        return None
    return SqlResult(
        sql=state["sql"],
        columns=list(state["sql_rows"][0]) if state["sql_rows"] else [],
        rows=state["sql_rows"],
        row_count=state["sql_row_count"],
        truncated=state["sql_truncated"],
    )


def _answer(state: AskState) -> CitedAnswer:
    return CitedAnswer.model_validate(state["answer"])
