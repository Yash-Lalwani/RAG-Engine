"""Self-RAG: score the answer, and allow one retry with a rewritten search query.

The retry loop itself is wired in the ask() graph; these are the plain steps it calls.
"""

from langsmith import traceable
from pydantic import BaseModel

from rag_engine import llm
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.guardrails.spotlight import spotlight_documents
from rag_engine.models import SearchChunk, SelfCheck


class _Scores(BaseModel):
    groundedness: float
    completeness: float
    reason: str


@traceable(name="self_check")
def self_check(question: str, answer: str, chunks: list[SearchChunk]) -> SelfCheck:
    documents = spotlight_documents([(f"c{n}", c.source, c.text) for n, c in enumerate(chunks, 1)])
    scores = llm.generate_structured(
        prompts.SELF_CHECK_SYSTEM,
        f"Question: {question}\n\nAnswer:\n{answer}\n\n{documents}",
        _Scores,
        model=settings.llm_model_small,
    )
    groundedness, completeness = _clamp(scores.groundedness), _clamp(scores.completeness)
    return SelfCheck(
        groundedness=groundedness,
        completeness=completeness,
        score=min(groundedness, completeness),
        reason=scores.reason,
    )


def should_retry(score: float, threshold: float, retries_done: int) -> bool:
    return retries_done == 0 and score < threshold


def rewrite_query(question: str, reason: str, domain_description: str = "") -> str:
    """A better search query. It is used only for retrieval; the answer still targets `question`."""
    rewritten = llm.generate_text(
        prompts.rewrite_system(domain_description),
        f"Question: {question}\nWhy the first answer fell short: {reason}",
        model=settings.llm_model_small,
    )[0].strip().strip('"')
    return rewritten or question


def keep_retry(first_score: float, retry_score: float) -> bool:
    """Keep the retried answer only if it scored strictly higher than the first one."""
    return retry_score > first_score


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
