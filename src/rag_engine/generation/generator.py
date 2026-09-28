from __future__ import annotations

from rag_engine.generation.prompts import build_system_prompt
from rag_engine.grading.self_rag import reflect_on_answer, should_regenerate
from rag_engine.guardrails.spotlight import build_spotlighted_context
from rag_engine.llm import generate
from rag_engine.models import (
    ChatResponse,
    ResponseMetadata,
    RetrievedChunk,
    RetrievedChunkPreview,
)
from rag_engine.retrieval.search import _flag


def _generate(
    question: str,
    chunks: list[RetrievedChunk],
    flags: dict | None = None,
) -> ChatResponse:
    enable_self_reflective = bool(_flag(flags, "enable_self_reflective", False))

    spotlighted = build_spotlighted_context(chunks)
    system = build_system_prompt()

    def _raw(q: str) -> str:
        return generate(system, f"{spotlighted}\n\nQuestion: {q}")["text"]

    working_q = question
    raw = _raw(working_q)

    iterations = 0
    last_score: float | None = None
    final_refined: str | None = None
    if enable_self_reflective:
        while True:
            reflection = reflect_on_answer(
                question=working_q,
                answer=raw,
                context=spotlighted,
            )
            last_score = float(reflection.reflection_score)
            if not should_regenerate(reflection, iterations):
                break
            final_refined = reflection.refined_question or working_q
            working_q = final_refined
            raw = _raw(working_q)
            iterations += 1

    chunk_previews = [
        RetrievedChunkPreview(text=c.text, source=c.source, score=c.score) for c in chunks
    ]
    return ChatResponse(
        answer=raw,
        sources=list({c.source for c in chunks}),
        confidence=0.7,
        metadata=ResponseMetadata(
            route="rag",
            retrieved_chunks=chunk_previews,
            reflection_iterations=iterations,
            reflection_score=last_score,
            refined_question=final_refined,
        ),
    )
