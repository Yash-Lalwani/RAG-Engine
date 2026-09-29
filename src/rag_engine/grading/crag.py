"""CRAG: grade every retrieved chunk, drop weak ones, fall back to web search if nothing is relevant."""

import hashlib
import logging
from typing import Literal, NamedTuple

from langsmith import traceable
from pydantic import BaseModel

from rag_engine import llm, web_search
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.guardrails.spotlight import spotlight_documents
from rag_engine.models import SearchChunk

logger = logging.getLogger(__name__)

Label = Literal["relevant", "partial", "irrelevant"]


class _Grade(BaseModel):
    document: int
    score: float
    label: Label


class _Grades(BaseModel):
    grades: list[_Grade]


class CragOutcome(NamedTuple):
    chunks: list[SearchChunk]
    action: str  # correct | web_fallback | insufficient | grading_failed
    web_used: bool
    insufficient_context: bool
    warnings: list[str]


@traceable(name="grade_chunks")
def grade_chunks(query: str, chunks: list[SearchChunk]) -> list[SearchChunk]:
    """Return the chunks with `grade` and `grade_score` set, from one small-model call."""
    if not chunks:
        return []
    documents = [(str(n), c.source, c.text) for n, c in enumerate(chunks, start=1)]
    result = llm.generate_structured(
        prompts.GRADE_SYSTEM,
        f"Question: {query}\n\n{spotlight_documents(documents)}",
        _Grades,
        model=settings.llm_model_small,
    )
    grades = {g.document: g for g in result.grades}
    graded = []
    for n, chunk in enumerate(chunks, start=1):
        grade = grades.get(n)
        if grade is None:
            logger.warning("CRAG returned no grade for document %d; treating it as irrelevant", n)
        graded.append(
            chunk.model_copy(
                update={
                    "grade": grade.label if grade else "irrelevant",
                    "grade_score": min(max(grade.score, 0.0), 1.0) if grade else 0.0,
                }
            )
        )
    return graded


def apply_crag(
    query: str, chunks: list[SearchChunk], threshold: float, web_fallback: bool
) -> CragOutcome:
    try:
        graded = grade_chunks(query, chunks)
    except Exception as exc:
        logger.warning("CRAG grading failed: %s", exc)
        return CragOutcome(chunks, "grading_failed", False, False, [f"CRAG grading failed: {exc}"])

    kept = [c for c in graded if c.grade_score >= threshold and c.grade != "irrelevant"]
    if any(c.grade == "relevant" for c in kept):
        return CragOutcome(kept, "correct", False, False, [])

    warnings = []
    if web_fallback and settings.tavily_api_key:
        try:
            web_chunks = [_web_chunk(r) for r in web_search.search_web(query)]
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            web_chunks, warnings = [], [f"Web search failed: {exc}"]
        if web_chunks:
            return CragOutcome(kept + web_chunks, "web_fallback", True, False, warnings)
    return CragOutcome(kept, "insufficient", False, True, warnings)


def _web_chunk(result: web_search.WebResult) -> SearchChunk:
    return SearchChunk(
        id="web-" + hashlib.sha256(result.url.encode()).hexdigest()[:16],
        doc_id=result.url,
        source=result.url,
        chunk_index=0,
        text=result.content,
        fused_score=0.0,
        url=result.url,
        metadata={"source_type": "web", "title": result.title},
    )
