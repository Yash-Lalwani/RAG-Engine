"""HyDE: embed a few hypothetical answers together with the query for the dense leg."""

from rag_engine import llm
from rag_engine.config import settings
from rag_engine.generation import prompts
from rag_engine.retrieval import embeddings

HYPOTHESES = 3


def hypothetical_answers(query: str, domain_description: str = "") -> list[str]:
    """One small-model call that returns HYPOTHESES short passages (the `n` parameter)."""
    answers = llm.generate_text(
        prompts.hyde_system(domain_description),
        query,
        model=settings.llm_model_small,
        temperature=0.7,
        n=HYPOTHESES,
    )
    return [answer.strip() for answer in answers if answer.strip()]


def hyde_vector(query: str, domain_description: str = "") -> list[float]:
    texts = [query, *hypothetical_answers(query, domain_description)]
    return average_vectors(embeddings.embed_dense(texts))


def average_vectors(vectors: list[list[float]]) -> list[float]:
    return [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]
