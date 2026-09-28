"""search(): (HyDE) -> dense and/or BM25 legs -> RRF fusion -> rerank -> CRAG grading."""

import logging
import time
from typing import Any

from qdrant_client.models import ScoredPoint

from rag_engine import collections
from rag_engine.grading.crag import apply_crag
from rag_engine.models import (
    CollectionSettings,
    EngineError,
    Passage,
    SearchChunk,
    SearchInfo,
    SearchResult,
)
from rag_engine.retrieval import embeddings, hyde, vector_store
from rag_engine.retrieval.fusion import rrf_fuse
from rag_engine.retrieval.rerank import rerank

logger = logging.getLogger(__name__)


def search(
    collection_id: str,
    query: str,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> SearchResult:
    started = time.perf_counter()
    query = query.strip()
    if not query:
        raise EngineError("query must not be empty")

    collection = collections.get(collection_id)
    overrides = dict(options or {})
    if top_k is not None:
        overrides["top_k"] = top_k
    config = collection.settings.merged(overrides)
    query_filter = vector_store.build_filter(
        collection_id, metadata_filters=check_filters(filters, config.filterable_fields)
    )

    info = SearchInfo(mode=config.search_mode)
    timings: dict[str, float] = {}
    ranked_lists: list[list[str]] = []
    payloads: dict[str, dict[str, Any]] = {}

    if config.search_mode in ("dense", "hybrid"):
        leg_started = time.perf_counter()
        vector = _dense_query_vector(query, config, info)
        hits = vector_store.query(vector, vector_store.DENSE_VECTOR, query_filter, config.fetch_k)
        ranked_lists.append(_collect(hits, payloads))
        timings["dense_ms"] = _ms_since(leg_started)

    if config.search_mode in ("sparse", "hybrid"):
        leg_started = time.perf_counter()
        sparse_vector = embeddings.embed_sparse_query(query)
        hits = []
        if sparse_vector.indices:
            hits = vector_store.query(
                sparse_vector, vector_store.SPARSE_VECTOR, query_filter, config.fetch_k
            )
        ranked_lists.append(_collect(hits, payloads))
        timings["sparse_ms"] = _ms_since(leg_started)

    fused = rrf_fuse(ranked_lists, config.rrf_k)[: config.fetch_k]
    chunks = [_to_chunk(pid, score, payloads[pid]) for pid, score in fused]
    info.candidates = len(chunks)

    if config.rerank and chunks:
        step_started = time.perf_counter()
        try:
            chunks = _rerank_chunks(query, chunks, config)
            info.reranked = True
        except Exception as exc:
            logger.warning("Reranking failed, keeping fused order: %s", exc)
            info.warnings.append(f"Reranking failed: {exc}")
        timings["rerank_ms"] = _ms_since(step_started)
    chunks = chunks[: config.top_k]

    if config.crag:
        step_started = time.perf_counter()
        outcome = apply_crag(query, chunks, config.crag_threshold, config.crag_web_fallback)
        chunks = outcome.chunks
        info.crag_action = outcome.action
        info.web_used = outcome.web_used
        info.insufficient_context = outcome.insufficient_context
        info.warnings.extend(outcome.warnings)
        timings["crag_ms"] = _ms_since(step_started)

    timings["total_ms"] = _ms_since(started)
    info.timings_ms = timings
    return SearchResult(collection_id=collection_id, query=query, chunks=chunks, info=info)


def _dense_query_vector(query: str, config: CollectionSettings, info: SearchInfo) -> list[float]:
    """HyDE vector if enabled (falls back to the plain query on failure), else the query embedding."""
    if config.hyde:
        try:
            vector = hyde.hyde_vector(query, config.domain_description)
            info.hyde_used = True
            return vector
        except Exception as exc:
            logger.warning("HyDE failed, using the plain query: %s", exc)
            info.warnings.append(f"HyDE failed: {exc}")
    return embeddings.embed_dense([query])[0]


def check_filters(
    filters: dict[str, Any] | None, filterable_fields: list[str]
) -> dict[str, Any]:
    """Exact-match filters only: one value (text, integer, true/false) or a list of them."""
    for field, value in (filters or {}).items():
        if field not in filterable_fields:
            allowed = ", ".join(filterable_fields) or "none"
            raise EngineError(
                f"Cannot filter on {field!r}: it is not in this collection's filterable_fields "
                f"({allowed})"
            )
        if isinstance(value, list):
            if not value or not (
                all(isinstance(v, str) for v in value)
                or all(isinstance(v, int) and not isinstance(v, bool) for v in value)
            ):
                raise EngineError(
                    f"Filter {field!r}: a list must be non-empty and all text or all integers"
                )
        elif not isinstance(value, str | int):
            raise EngineError(f"Filter {field!r}: value must be text, an integer or true/false")
    return dict(filters or {})


def _collect(hits: list[ScoredPoint], payloads: dict[str, dict[str, Any]]) -> list[str]:
    ids = []
    for hit in hits:
        point_id = str(hit.id)
        payloads[point_id] = hit.payload or {}
        ids.append(point_id)
    return ids


def _to_chunk(point_id: str, fused_score: float, payload: dict[str, Any]) -> SearchChunk:
    return SearchChunk(
        id=point_id,
        doc_id=payload["doc_id"],
        source=payload["source"],
        chunk_index=payload["chunk_index"],
        text=payload["text"],
        fused_score=fused_score,
        url=payload.get("url"),
        metadata=payload.get("metadata", {}),
    )


def _rerank_chunks(
    query: str, chunks: list[SearchChunk], config: CollectionSettings
) -> list[SearchChunk]:
    by_id = {chunk.id: chunk for chunk in chunks}
    ranked = rerank(
        query,
        [Passage(id=chunk.id, text=chunk.text) for chunk in chunks],
        top_k=config.top_k,
        backend=config.reranker,
    )
    return [by_id[p.id].model_copy(update={"rerank_score": p.score}) for p in ranked]


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
