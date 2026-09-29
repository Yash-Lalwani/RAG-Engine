"""ingest_document(): read -> hash check -> parse -> chunk -> embed -> replace points -> record."""

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any

from langsmith import traceable
from qdrant_client.models import PointStruct

from rag_engine import collections
from rag_engine.config import settings
from rag_engine.ingestion.chunker import chunk_document
from rag_engine.ingestion.parser import parse_document
from rag_engine.models import EngineError, IngestResult
from rag_engine.retrieval import embeddings, vector_store

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024


@traceable(name="ingest_document")
def ingest_document(
    collection_id: str,
    file_path: str | None = None,
    content_base64: str | None = None,
    filename: str | None = None,
    doc_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    collection = collections.get(collection_id)
    data, source_name = read_input(file_path, content_base64, filename)
    doc_id = doc_id or source_name
    metadata = dict(metadata or {})
    check_metadata(metadata, collection.settings.filterable_fields)

    content_hash = document_hash(data, metadata)
    existing = collections.get_document(collection_id, doc_id)
    if existing and existing.content_hash == content_hash:
        return IngestResult(
            collection_id=collection_id,
            doc_id=doc_id,
            status="unchanged",
            chunk_count=existing.chunk_count,
        )

    chunks = chunk_document(parse_document(data, source_name))
    if not chunks:
        raise EngineError(f"No text could be extracted from {source_name!r}")
    points = build_points(collection_id, doc_id, source_name, chunks, metadata)

    # Only now, with the new points ready, remove the old ones (a failed parse keeps the old doc).
    vector_store.delete_points(collection_id, doc_id)
    vector_store.upsert_points(points)
    collections.save_document(
        collection_id, doc_id, source_name, content_hash, len(points), metadata
    )
    return IngestResult(
        collection_id=collection_id,
        doc_id=doc_id,
        status="replaced" if existing else "ingested",
        chunk_count=len(points),
    )


def read_input(
    file_path: str | None, content_base64: str | None, filename: str | None
) -> tuple[bytes, str]:
    """Return (bytes, source name). Exactly one of file_path or content_base64 must be given."""
    if (file_path is None) == (content_base64 is None):
        raise EngineError("Pass exactly one of file_path or content_base64")

    if file_path is not None:
        path = Path(file_path).resolve()
        allowed_dir = Path(settings.ingest_dir).resolve()
        if not path.is_relative_to(allowed_dir):
            raise EngineError(f"file_path must be inside the ingest directory ({allowed_dir})")
        if not path.is_file():
            raise EngineError(f"File not found: {file_path}")
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise EngineError(_too_large_message())
        data, source_name = path.read_bytes(), filename or path.name
    else:
        if not filename:
            raise EngineError("filename is required with content_base64")
        try:
            data = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError):
            raise EngineError("content_base64 is not valid base64") from None
        if len(data) > MAX_DOCUMENT_BYTES:
            raise EngineError(_too_large_message())
        source_name = filename

    if not data:
        raise EngineError("The document is empty")
    return data, Path(source_name).name


def document_hash(data: bytes, metadata: dict[str, Any]) -> str:
    """SHA-256 of the content and the metadata, so a metadata-only change is also a change."""
    try:
        metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        raise EngineError("metadata must be JSON-serializable") from None
    return hashlib.sha256(data + b"\0" + metadata_json.encode("utf-8")).hexdigest()


def check_metadata(metadata: dict[str, Any], filterable_fields: list[str]) -> None:
    """Filterable fields must hold one simple value so exact-match filters work on them."""
    for field in filterable_fields:
        value = metadata.get(field)
        if value is not None and not isinstance(value, str | int):
            raise EngineError(
                f"metadata {field!r} is a filterable field, so its value must be text, "
                "an integer or true/false"
            )
    url = metadata.get("url")
    if url is not None and not isinstance(url, str):
        raise EngineError("metadata 'url' must be text")


def build_points(
    collection_id: str,
    doc_id: str,
    source_name: str,
    chunks: list[dict],
    metadata: dict[str, Any],
) -> list[PointStruct]:
    texts = [chunk["text"] for chunk in chunks]
    dense_vectors = embeddings.embed_dense(texts)
    sparse_vectors = embeddings.embed_sparse_documents(texts)

    points = []
    for index, (chunk, dense, sparse) in enumerate(
        zip(chunks, dense_vectors, sparse_vectors, strict=True)
    ):
        chunk_metadata = dict(metadata)
        if chunk["page_number"] is not None:
            chunk_metadata["page_number"] = chunk["page_number"]
        points.append(
            PointStruct(
                id=vector_store.point_id(collection_id, doc_id, index),
                vector={vector_store.DENSE_VECTOR: dense, vector_store.SPARSE_VECTOR: sparse},
                payload={
                    "collection_id": collection_id,
                    "doc_id": doc_id,
                    "source": source_name,
                    "chunk_index": index,
                    "text": chunk["text"],
                    "url": metadata.get("url"),
                    "metadata": chunk_metadata,
                },
            )
        )
    return points


def _too_large_message() -> str:
    return f"The document is larger than {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB"
