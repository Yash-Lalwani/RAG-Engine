"""Qdrant access: one collection holding every Engine chunk, separated by `collection_id`."""

import uuid
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from rag_engine.config import settings
from rag_engine.models import EngineError

QDRANT_COLLECTION = "rag_engine_chunks"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
EMBEDDING_SIZES = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_POINT_ID_NAMESPACE = uuid.UUID("5b8c6f1e-2f0a-4d8e-9c1b-7a3e4d2f6b10")


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=30)


def ping() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def point_id(collection_id: str, doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, f"{collection_id}/{doc_id}/{chunk_index}"))


def dense_vector_size() -> int:
    size = EMBEDDING_SIZES.get(settings.embedding_model)
    if size is None:
        raise EngineError(f"Unknown vector size for embedding model {settings.embedding_model!r}")
    return size


def ensure_collection() -> None:
    client = get_client()
    size = dense_vector_size()
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            QDRANT_COLLECTION,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
    else:
        existing = client.get_collection(QDRANT_COLLECTION).config.params.vectors[DENSE_VECTOR].size
        if existing != size:
            raise EngineError(
                f"Qdrant collection {QDRANT_COLLECTION!r} stores {existing}-dim vectors but "
                f"{settings.embedding_model!r} produces {size}. Re-create the Qdrant collection."
            )
    ensure_field_index("collection_id")
    ensure_field_index("doc_id")


def ensure_field_index(field_key: str) -> None:
    """Create a keyword payload index (e.g. on "metadata.section") if it does not exist yet."""
    client = get_client()
    if field_key not in client.get_collection(QDRANT_COLLECTION).payload_schema:
        client.create_payload_index(
            QDRANT_COLLECTION, field_key, field_schema=models.PayloadSchemaType.KEYWORD, wait=True
        )


def build_filter(
    collection_id: str, doc_id: str | None = None, metadata_filters: dict[str, Any] | None = None
) -> models.Filter:
    """All conditions must match. A list value means "any of these values"."""
    conditions = [_match("collection_id", collection_id)]
    if doc_id is not None:
        conditions.append(_match("doc_id", doc_id))
    for field, value in (metadata_filters or {}).items():
        conditions.append(_match(f"metadata.{field}", value))
    return models.Filter(must=conditions)


def _match(key: str, value: Any) -> models.FieldCondition:
    if isinstance(value, list):
        return models.FieldCondition(key=key, match=models.MatchAny(any=value))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def upsert_points(points: list[models.PointStruct]) -> None:
    get_client().upsert(QDRANT_COLLECTION, points=points, wait=True)


def delete_points(collection_id: str, doc_id: str | None = None) -> None:
    get_client().delete(
        QDRANT_COLLECTION,
        points_selector=models.FilterSelector(filter=build_filter(collection_id, doc_id)),
        wait=True,
    )


def count_points(collection_id: str, doc_id: str | None = None) -> int:
    return get_client().count(
        QDRANT_COLLECTION, count_filter=build_filter(collection_id, doc_id), exact=True
    ).count


def get_texts(point_ids: list[str]) -> dict[str, str]:
    """Full chunk text for the given point ids (ids that are not stored, like web results, are skipped)."""
    points = get_client().retrieve(QDRANT_COLLECTION, ids=point_ids, with_payload=["text"])
    return {str(point.id): point.payload["text"] for point in points}


def query(
    vector: list[float] | models.SparseVector,
    using: str,
    query_filter: models.Filter,
    limit: int,
) -> list[models.ScoredPoint]:
    return get_client().query_points(
        QDRANT_COLLECTION,
        query=vector,
        using=using,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points
