"""Public API of the Engine. The MCP server and the tester call only these functions."""

import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.types import Command
from pydantic import ValidationError

from rag_engine import collections, config, db
from rag_engine.cache.keys import ANSWER_TIER, ANSWER_TTL, answer_key
from rag_engine.cache.store import cache, track_cache
from rag_engine.grading import citations
from rag_engine.graph.builder import get_graph
from rag_engine.guardrails.input_checks import guard_input
from rag_engine.guardrails.output_checks import guard_answer
from rag_engine.ingestion import pipeline
from rag_engine.models import (
    AskMetadata,
    AskResult,
    Blocked,
    ChunkPreview,
    Collection,
    CollectionSettings,
    DeleteResult,
    DocumentRecord,
    EngineError,
    HealthStatus,
    IngestResult,
    Passage,
    RerankedPassage,
    SearchInfo,
    SearchResult,
    SelfRagInfo,
    Statement,
    TokenUsage,
    VerificationResult,
    validation_message,
)
from rag_engine.retrieval import embeddings, vector_store
from rag_engine.retrieval import rerank as rerank_module
from rag_engine.retrieval import search as search_module

logger = logging.getLogger(__name__)

PAUSE_LIMIT = timedelta(hours=24)
PREVIEW_ROWS = 20
PREVIEW_CHARS = 300


def setup() -> None:
    """Create the Postgres tables, the Qdrant collection and the ask() graph, and delete paused
    runs older than 24 hours."""
    collections.ensure_schema()
    vector_store.ensure_collection()
    get_graph()
    delete_expired_runs()


def health() -> HealthStatus:
    """Check Postgres, Qdrant and Redis. Makes no paid API calls."""
    postgres, qdrant, redis = db.ping(), vector_store.ping(), cache.ping()
    healthy = postgres and qdrant and redis != "error"
    return HealthStatus(status="ok" if healthy else "degraded", postgres=postgres, qdrant=qdrant, redis=redis)


def warm_up() -> None:
    """Load the local models (reranker, BM25, guardrail classifiers) so the first request is not slow.
    Makes no paid API calls; a model that cannot load is only logged."""
    steps = {
        "reranker": lambda: rerank_module.rerank("warm up", [Passage(id="w", text="warm up")]),
        "BM25": lambda: embeddings.embed_sparse_query("warm up"),
        "guardrail classifiers": lambda: guard_input("warm up"),
    }
    for name, step in steps.items():
        try:
            step()
        except Exception as exc:
            logger.warning("Could not warm up the %s: %s", name, exc)


def create_collection(
    collection_id: str,
    name: str,
    description: str = "",
    settings: dict[str, Any] | None = None,
) -> Collection:
    parsed = CollectionSettings.from_dict(settings)
    _check_sql_database(parsed)
    collection = collections.create(collection_id, name, description, parsed)
    _ensure_filter_indexes(parsed)
    return collection


def update_collection(collection_id: str, settings: dict[str, Any]) -> Collection:
    merged = collections.get(collection_id).settings.merged(settings)
    _check_sql_database(merged)
    collection = collections.update_settings(collection_id, merged)
    _ensure_filter_indexes(merged)
    return collection


def list_collections() -> list[Collection]:
    return collections.list_all()


def delete_collection(collection_id: str) -> DeleteResult:
    collections.get(collection_id)
    vector_store.delete_points(collection_id)
    collections.delete(collection_id)
    return DeleteResult(collection_id=collection_id)


def ingest_document(
    collection_id: str,
    file_path: str | None = None,
    content_base64: str | None = None,
    filename: str | None = None,
    doc_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    return pipeline.ingest_document(
        collection_id, file_path, content_base64, filename, doc_id, metadata
    )


def list_documents(collection_id: str) -> list[DocumentRecord]:
    collections.get(collection_id)
    return collections.list_documents(collection_id)


def delete_document(collection_id: str, doc_id: str) -> DeleteResult:
    collections.get(collection_id)
    if collections.get_document(collection_id, doc_id) is None:
        raise EngineError(f"Document {doc_id!r} does not exist in collection {collection_id!r}")
    vector_store.delete_points(collection_id, doc_id)
    collections.delete_document(collection_id, doc_id)
    return DeleteResult(collection_id=collection_id, doc_id=doc_id)


def search(
    collection_id: str,
    query: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> SearchResult:
    """Raises Blocked if the query fails G1 or G2."""
    if not query.strip():
        raise EngineError("query must not be empty")
    input_warnings = guard_input(query)
    result = search_module.search(collection_id, query, top_k, filters, options)
    result.info.warnings = input_warnings + result.info.warnings
    return result


def rerank(
    query: str,
    passages: list[Passage | dict[str, Any]],
    top_k: int | None = None,
    backend: rerank_module.RerankBackend = "local",
) -> list[RerankedPassage]:
    """Rerank passages supplied by the caller (for example live results from another source)."""
    try:
        parsed = [Passage.model_validate(p) for p in passages]
    except ValidationError as error:
        raise EngineError(f"Invalid passages: {validation_message(error)}") from None
    return rerank_module.rerank(query, parsed, top_k, backend)


def ask(
    collection_id: str, question: str, options: dict[str, Any] | None = None, *, caller: str
) -> AskResult:
    """Answer a question. SQL and hybrid questions pause with status "pending_sql" until
    approve_sql() is called with the returned query_id."""
    question = question.strip()
    if not question:
        raise EngineError("question must not be empty")
    query_id = str(uuid.uuid4())
    try:
        input_warnings = guard_input(question)
    except Blocked as blocked:
        return AskResult(status="blocked", query_id=query_id, message=str(blocked))
    collection = collections.get(collection_id)
    options = dict(options or {})
    filters = options.pop("filters", None) or {}
    settings = collection.settings.merged(options)
    search_module.check_filters(filters, settings.filterable_fields)

    effective = {"settings": settings.model_dump(), "filters": filters}
    key = answer_key(collection_id, collection.version, question, effective)
    started = time.perf_counter()
    with track_cache() as cache_counts:
        cached = cache.get(ANSWER_TIER, key)
    if cached is not None:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        return _from_answer_cache(cached, query_id, cache_counts, elapsed_ms, input_warnings)

    state = {
        "query_id": query_id,
        "caller": caller,
        "created_at": datetime.now(UTC).isoformat(),
        "collection_id": collection_id,
        "question": question,
        "search_query": question,
        "options": options,
        "filters": filters,
        "settings": settings.model_dump(),
        "retries": 0,
        "warnings": input_warnings,
        "cache": cache_counts,
    }
    return _run(state, query_id, answer_cache_key=key)


def approve_sql(query_id: str, approve: bool, *, caller: str) -> AskResult:
    """Resume a paused ask() run. Problems are returned as status "error", not raised."""
    graph = get_graph()
    snapshot = graph.get_state(_thread(query_id))
    values = snapshot.values
    if not values:
        return _error(query_id, f"Unknown query_id {query_id!r}")
    if values.get("caller") != caller:
        return _error(query_id, "This query belongs to another caller")
    if "approve" not in snapshot.next:
        return _error(query_id, "This query is not waiting for SQL approval")
    if _is_expired(values["created_at"]):
        graph.checkpointer.delete_thread(query_id)
        return _error(query_id, "This query waited more than 24 hours for approval and has expired")
    return _run(Command(resume={"approved": bool(approve)}), query_id)


def delete_expired_runs() -> int:
    """Delete paused ask() runs older than PAUSE_LIMIT. Returns how many were deleted."""
    graph = get_graph()
    thread_ids = {item.config["configurable"]["thread_id"] for item in graph.checkpointer.list(None)}
    deleted = 0
    for thread_id in thread_ids:
        created_at = graph.get_state(_thread(thread_id)).values.get("created_at")
        if created_at is None or _is_expired(created_at):
            graph.checkpointer.delete_thread(thread_id)
            deleted += 1
    return deleted


def _run(graph_input: Any, query_id: str, answer_cache_key: str | None = None) -> AskResult:
    graph = get_graph()
    try:
        state = graph.invoke(graph_input, _thread(query_id))
    except Exception as exc:
        logger.exception("ask() run %s failed", query_id)
        graph.checkpointer.delete_thread(query_id)
        return _error(query_id, f"The question could not be answered: {exc}")

    if "__interrupt__" in state:
        return _to_result(state, pending=True)

    graph.checkpointer.delete_thread(query_id)
    result = guard_answer(_to_result(state, pending=False))
    cacheable = result.status == "completed" and result.intent == "rag" and not result.metadata.warnings
    if answer_cache_key and cacheable:
        cache.set(ANSWER_TIER, answer_cache_key, result.model_dump_json(), ANSWER_TTL)
    return result


def _to_result(state: dict[str, Any], pending: bool) -> AskResult:
    timings = dict(state.get("timings_ms") or {})
    timings["total_ms"] = round(sum(timings.values()), 1)
    search_info = state.get("search_info")
    metadata = AskMetadata(
        cache=state.get("cache") or {},
        timings_ms=timings,
        search=SearchInfo.model_validate(search_info) if search_info else None,
        self_rag=SelfRagInfo.model_validate(state["self_rag"]) if state.get("self_rag") else None,
        sql_row_count=state.get("sql_row_count"),
        sql_truncated=state.get("sql_truncated", False),
        token_usage=TokenUsage.model_validate(state.get("usage") or {}),
        warnings=state.get("warnings") or [],
    )
    common = {
        "query_id": state["query_id"],
        "intent": state.get("intent"),
        "chunks": [_preview(chunk) for chunk in state.get("chunks") or []],
        "sql": state.get("sql"),
        "sql_explanation": state.get("sql_explanation"),
        "metadata": metadata,
    }
    if pending:
        return AskResult(
            status="pending_sql",
            message="Review the SQL, then call approve_sql with this query_id to run or reject it.",
            **common,
        )
    return AskResult(
        status=state["status"],
        answer=state.get("final_answer", ""),
        statements=state.get("final_statements") or [],
        sources=state.get("sources") or [],
        rows_preview=(state.get("sql_rows") or [])[:PREVIEW_ROWS],
        verification=state.get("verification"),
        insufficient_context=(state.get("answer") or {}).get("insufficient_context", False),
        message=state.get("message"),
        **common,
    )


def _from_answer_cache(
    cached: str, query_id: str, cache_counts: dict, elapsed_ms: float, warnings: list[str]
) -> AskResult:
    result = AskResult.model_validate_json(cached)
    metadata = result.metadata.model_copy(
        update={
            "cache_hit": True,
            "cache": cache_counts,
            "token_usage": TokenUsage(),
            "timings_ms": {"total_ms": elapsed_ms},
            "warnings": warnings,
        }
    )
    return result.model_copy(update={"query_id": query_id, "metadata": metadata})


def _preview(chunk: dict[str, Any]) -> ChunkPreview:
    return ChunkPreview.model_validate({**chunk, "text": chunk["text"][:PREVIEW_CHARS]})


def _error(query_id: str, message: str) -> AskResult:
    return AskResult(status="error", query_id=query_id, message=message)


def _is_expired(created_at: str) -> bool:
    return datetime.now(UTC) - datetime.fromisoformat(created_at) > PAUSE_LIMIT


def _thread(query_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": query_id}}


def verify_citations(
    statements: list[Statement | dict[str, Any]],
    passages: list[Passage | dict[str, Any]],
    strict: bool = False,
) -> VerificationResult:
    """Check caller-supplied statements against caller-supplied passages (see grading/citations.py)."""
    try:
        parsed_statements = [Statement.model_validate(s) for s in statements]
        parsed_passages = [Passage.model_validate(p) for p in passages]
    except ValidationError as error:
        raise EngineError(f"Invalid input: {validation_message(error)}") from None
    return citations.verify_citations(parsed_statements, parsed_passages, strict)


def _check_sql_database(settings: CollectionSettings) -> None:
    if settings.sql_database is None:
        return
    known = config.settings.sql_database_urls
    if settings.sql_database not in known:
        raise EngineError(
            f"sql_database {settings.sql_database!r} is not configured in SQL_DATABASES "
            f"(known: {', '.join(known) or 'none'})"
        )
    if not settings.sql_allowed_tables:
        raise EngineError("A collection with sql_database must also list sql_allowed_tables")


def _ensure_filter_indexes(settings: CollectionSettings) -> None:
    for field in settings.filterable_fields:
        vector_store.ensure_field_index(f"metadata.{field}")
