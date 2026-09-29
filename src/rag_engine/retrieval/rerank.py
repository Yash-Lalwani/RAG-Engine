"""Rerank passages with a local cross-encoder (default) or the Voyage API."""

from functools import lru_cache
from typing import Literal

from langsmith import traceable

from rag_engine.config import settings
from rag_engine.models import EngineError, Passage, RerankedPassage

RerankBackend = Literal["local", "voyage"]


@lru_cache
def _cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


@lru_cache
def _voyage_client():
    if not settings.voyage_api_key:
        raise EngineError("The voyage reranker needs VOYAGE_API_KEY")
    import voyageai

    return voyageai.Client(api_key=settings.voyage_api_key)


@traceable(name="rerank")
def rerank(
    query: str,
    passages: list[Passage],
    top_k: int | None = None,
    backend: RerankBackend = "local",
) -> list[RerankedPassage]:
    """Score every passage against the query and return the best `top_k`, highest first."""
    if not passages:
        return []
    texts = [p.text for p in passages]
    scores = _voyage_scores(query, texts) if backend == "voyage" else _local_scores(query, texts)
    ranked = sorted(zip(passages, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [
        RerankedPassage(**passage.model_dump(), score=score)
        for passage, score in ranked[: top_k or len(passages)]
    ]


def _local_scores(query: str, texts: list[str]) -> list[float]:
    return [float(s) for s in _cross_encoder().predict([(query, text) for text in texts])]


def _voyage_scores(query: str, texts: list[str]) -> list[float]:
    result = _voyage_client().rerank(query=query, documents=texts, model=settings.voyage_model)
    scores = [0.0] * len(texts)
    for item in result.results:
        scores[item.index] = float(item.relevance_score)
    return scores
