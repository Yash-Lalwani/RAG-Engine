import pytest

from rag_engine import llm
from rag_engine.retrieval import embeddings, hyde


def test_average_vectors():
    assert hyde.average_vectors([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]) == pytest.approx([1.0, 1.0])


def test_hyde_vector_averages_the_query_and_three_hypotheses(monkeypatch):
    seen = {}

    def fake_text(system, user, model=None, temperature=0.0, n=1):
        seen["n"], seen["temperature"] = n, temperature
        return ["h1", "h2", " "]

    def fake_embed(texts):
        seen["texts"] = texts
        return [[float(i), 1.0] for i in range(len(texts))]

    monkeypatch.setattr(llm, "generate_text", fake_text)
    monkeypatch.setattr(embeddings, "embed_dense", fake_embed)
    vector = hyde.hyde_vector("what is a pod")
    assert seen["n"] == 3 and seen["temperature"] == 0.7
    assert seen["texts"] == ["what is a pod", "h1", "h2"]
    assert vector == pytest.approx([1.0, 1.0])
