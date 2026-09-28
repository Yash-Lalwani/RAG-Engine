"""Public API of the Engine. The MCP server and the tester call only these functions."""

from typing import Any

from pydantic import ValidationError

from rag_engine import collections, config
from rag_engine.ingestion import pipeline
from rag_engine.models import (
    Collection,
    CollectionSettings,
    DeleteResult,
    DocumentRecord,
    EngineError,
    IngestResult,
    Passage,
    RerankedPassage,
    SearchResult,
    validation_message,
)
from rag_engine.retrieval import rerank as rerank_module
from rag_engine.retrieval import search as search_module
from rag_engine.retrieval import vector_store


def setup() -> None:
    """Create the Postgres tables and the Qdrant collection if they do not exist."""
    collections.ensure_schema()
    vector_store.ensure_collection()


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
    return search_module.search(collection_id, query, top_k, filters, options)


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


def _check_sql_database(settings: CollectionSettings) -> None:
    known = config.settings.sql_database_urls
    if settings.sql_database is not None and settings.sql_database not in known:
        raise EngineError(
            f"sql_database {settings.sql_database!r} is not configured in SQL_DATABASES "
            f"(known: {', '.join(known) or 'none'})"
        )


def _ensure_filter_indexes(settings: CollectionSettings) -> None:
    for field in settings.filterable_fields:
        vector_store.ensure_field_index(f"metadata.{field}")
