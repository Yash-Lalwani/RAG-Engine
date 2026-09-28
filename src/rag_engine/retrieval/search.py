from __future__ import annotations

import logging

from rag_engine.config import settings
from rag_engine.grading.crag import crag_pipeline
from rag_engine.models import RetrievedChunk
from rag_engine.retrieval.embeddings import embed_texts
from rag_engine.retrieval.hyde import HyDERetriever
from rag_engine.retrieval.rerank import Reranker
from rag_engine.retrieval.vector_store import hybrid_search, search, sparse_search

logger = logging.getLogger(__name__)


def _flag(flags: dict | None, key: str, default):
    if not isinstance(flags, dict):
        return default
    return flags.get(key, default)


def _retrieve(question: str, flags: dict | None = None) -> list[RetrievedChunk]:
    final_top_k = int(_flag(flags, "top_k", 5))
    mode = _flag(flags, "search_mode", "dense")
    rerank = bool(_flag(flags, "enable_rerank", False))
    hyde = bool(_flag(flags, "enable_hyde", False))
    enable_crag = bool(_flag(flags, "enable_crag", settings.crag_enabled_by_default))

    retrieve_k = settings.reranker_initial_top_k if rerank else final_top_k

    if hyde:
        chunks = HyDERetriever().retrieve(question, top_k=retrieve_k)
    elif mode == "sparse":
        chunks = sparse_search(question, top_k=retrieve_k)
    elif mode == "hybrid":
        query_embedding = embed_texts([question])[0]
        chunks = hybrid_search(query_embedding, question, top_k=retrieve_k)
    else:
        query_embedding = embed_texts([question])[0]
        chunks = search(query_embedding, top_k=retrieve_k)

    if rerank and chunks:
        chunks = Reranker().rerank(question, chunks, top_k=final_top_k)
    else:
        chunks = chunks[:final_top_k]

    chunks, evaluation, used_web = crag_pipeline(
        question=question,
        chunks=chunks,
        enable_crag=enable_crag,
    )
    logger.info(
        "CRAG | enabled=%s score=%s label=%s used_web=%s",
        enable_crag,
        evaluation.relevance_score,
        evaluation.relevance_label,
        used_web,
    )

    return chunks
