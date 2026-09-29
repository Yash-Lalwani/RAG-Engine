"""Dense embeddings (OpenAI, cached) and BM25 sparse vectors (fastembed)."""

import json
from functools import lru_cache

from fastembed import SparseTextEmbedding
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from qdrant_client.models import SparseVector

from rag_engine.cache.keys import EMBEDDING_TIER, EMBEDDING_TTL, embedding_key
from rag_engine.cache.store import cache
from rag_engine.config import settings

EMBEDDING_BATCH_SIZE = 100
BM25_MODEL = "Qdrant/bm25"


@lru_cache
def _openai_client() -> OpenAI:
    return wrap_openai(OpenAI(api_key=settings.openai_api_key))


@lru_cache
def _bm25_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(BM25_MODEL)


def embed_dense(texts: list[str]) -> list[list[float]]:
    model = settings.embedding_model
    keys = [embedding_key(model, text) for text in texts]
    cached = cache.get_many(EMBEDDING_TIER, keys)
    vectors: list[list[float] | None] = [json.loads(v) if v else None for v in cached]

    missing = [i for i, vector in enumerate(vectors) if vector is None]
    for start in range(0, len(missing), EMBEDDING_BATCH_SIZE):
        batch = missing[start : start + EMBEDDING_BATCH_SIZE]
        response = _openai_client().embeddings.create(model=model, input=[texts[i] for i in batch])
        for item in response.data:
            vectors[batch[item.index]] = item.embedding
        cache.set_many(
            EMBEDDING_TIER, {keys[i]: json.dumps(vectors[i]) for i in batch}, EMBEDDING_TTL
        )

    return vectors  # type: ignore[return-value]


def embed_sparse_documents(texts: list[str]) -> list[SparseVector]:
    return [_to_sparse_vector(e) for e in _bm25_model().embed(texts)]


def embed_sparse_query(text: str) -> SparseVector:
    return _to_sparse_vector(next(iter(_bm25_model().query_embed(text))))


def _to_sparse_vector(embedding) -> SparseVector:
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())
