import pytest

from rag_engine import engine, llm

pytestmark = pytest.mark.integration

POD_TEXT = "# Pods\n\nA pod is the smallest deployable unit. Pods run containers on nodes."
WEATHER_TEXT = "# Weather\n\nThe weather today is sunny with light wind."


@pytest.fixture
def collection(make_collection, ingest_text):
    cid = make_collection()
    ingest_text(cid, "pods", POD_TEXT)
    ingest_text(cid, "weather", WEATHER_TEXT)
    return cid


def fake_grader(system, user, schema, model=None, temperature=0.0):
    """Grades a document 'relevant' if it mentions pods, else 'irrelevant'."""
    documents = user.split("<document ")[1:]
    return schema(
        grades=[
            {"document": n, "score": 0.9 if "pod" in doc.lower() else 0.1,
             "label": "relevant" if "pod" in doc.lower() else "irrelevant"}
            for n, doc in enumerate(documents, start=1)
        ]
    )


def test_crag_grades_and_drops_irrelevant_chunks(collection, monkeypatch):
    monkeypatch.setattr(llm, "generate_structured", fake_grader)
    result = engine.search(collection, "pod containers weather", options={"crag": True})
    assert [c.doc_id for c in result.chunks] == ["pods"]
    assert result.chunks[0].grade == "relevant" and result.chunks[0].grade_score == 0.9
    assert result.info.crag_action == "correct" and not result.info.insufficient_context


def test_crag_reports_insufficient_context(collection, monkeypatch):
    monkeypatch.setattr(
        llm, "generate_structured",
        lambda s, u, schema, **k: schema(grades=[{"document": 1, "score": 0.1, "label": "irrelevant"}]),
    )
    result = engine.search(collection, "weather", top_k=1, options={"crag": True})
    assert result.chunks == [] and result.info.crag_action == "insufficient"
    assert result.info.insufficient_context


def test_hyde_is_skipped_in_sparse_mode(collection, monkeypatch):
    monkeypatch.setattr(llm, "generate_text", lambda *a, **k: pytest.fail("HyDE must not run"))
    result = engine.search(collection, "pod", options={"hyde": True, "search_mode": "sparse"})
    assert not result.info.hyde_used and result.chunks


def test_hyde_is_used_for_the_dense_leg(collection, monkeypatch):
    monkeypatch.setattr(llm, "generate_text", lambda *a, **k: ["Pods run containers."] * 3)
    result = engine.search(collection, "smallest unit", options={"hyde": True, "search_mode": "dense"})
    assert result.info.hyde_used and result.chunks[0].doc_id == "pods"


def test_hyde_failure_falls_back_to_the_plain_query(collection, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("OpenAI timeout")

    monkeypatch.setattr(llm, "generate_text", broken)
    result = engine.search(collection, "pod", options={"hyde": True, "search_mode": "hybrid"})
    assert not result.info.hyde_used and result.chunks
    assert "OpenAI timeout" in result.info.warnings[0]
