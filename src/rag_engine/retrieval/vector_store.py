from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_engine.config import settings
from rag_engine.models import RetrievedChunk

QDRANT_COLLECTION = "rag_engine_chunks"
VECTOR_SIZE = 1536


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in client.get_collections().collections}

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_chunks(chunks: list[RetrievedChunk], embeddings: list[list[float]]) -> None:
    ensure_collection()
    client = get_client()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk.text, "source": chunk.source},
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def search(query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
    client = get_client()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        RetrievedChunk(
            text=p.payload.get("text", ""),
            source=p.payload.get("source", ""),
            score=float(p.score),
        )
        for p in results
    ]
