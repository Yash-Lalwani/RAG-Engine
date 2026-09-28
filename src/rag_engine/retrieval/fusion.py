from rag_engine.models import RetrievedChunk


def fuse_rrf(
    result_lists: list[list[RetrievedChunk]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            key = chunk.text
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in meta:
                meta[key] = {"text": chunk.text, "source": chunk.source}

    return [
        RetrievedChunk(text=text, source=meta[text]["source"], score=score)
        for text, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
