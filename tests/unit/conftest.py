import pytest

from rag_engine.models import SearchChunk


@pytest.fixture
def make_chunk():
    def _make(chunk_id: str, text: str = "some text", source: str = "doc.md") -> SearchChunk:
        return SearchChunk(
            id=chunk_id, doc_id=source, source=source, chunk_index=0, text=text, fused_score=0.1
        )

    return _make


@pytest.fixture
def fake_structured(monkeypatch):
    """Replace llm.generate_structured with a function returning the given object (or raising)."""
    from rag_engine import llm

    calls = []

    def _install(result):
        def fake(system, user, schema, model=None, temperature=0.0):
            calls.append({"system": system, "user": user, "schema": schema, "model": model})
            if isinstance(result, Exception):
                raise result
            return result(schema) if callable(result) else result

        monkeypatch.setattr(llm, "generate_structured", fake)
        return calls

    return _install
