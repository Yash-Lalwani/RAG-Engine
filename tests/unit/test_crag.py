import pytest

from rag_engine import web_search
from rag_engine.config import settings
from rag_engine.grading import crag


def grades(*items):
    return lambda schema: schema(
        grades=[{"document": n, "score": score, "label": label} for n, score, label in items]
    )


@pytest.fixture
def chunks(make_chunk):
    return [make_chunk("a", "Pods run containers."), make_chunk("b", "Ingress."), make_chunk("c", "Weather.")]


def test_grades_map_back_to_chunks(fake_structured, chunks):
    fake_structured(grades((1, 0.9, "relevant"), (2, 0.4, "partial"), (3, 1.7, "irrelevant")))
    graded = crag.grade_chunks("what is a pod", chunks)
    assert [(c.id, c.grade, c.grade_score) for c in graded] == [
        ("a", "relevant", 0.9), ("b", "partial", 0.4), ("c", "irrelevant", 1.0)
    ]


def test_missing_grade_counts_as_irrelevant(fake_structured, chunks):
    fake_structured(grades((1, 0.9, "relevant")))
    assert [c.grade for c in crag.grade_chunks("q", chunks)] == ["relevant", "irrelevant", "irrelevant"]


def test_weak_and_irrelevant_chunks_are_dropped(fake_structured, chunks):
    fake_structured(grades((1, 0.9, "relevant"), (2, 0.3, "partial"), (3, 0.8, "irrelevant")))
    outcome = crag.apply_crag("q", chunks, threshold=0.5, web_fallback=False)
    assert [c.id for c in outcome.chunks] == ["a"]
    assert outcome.action == "correct" and not outcome.insufficient_context


def test_web_results_are_added_alongside_partial_local_chunks(fake_structured, chunks, monkeypatch):
    fake_structured(grades((1, 0.6, "partial"), (2, 0.1, "irrelevant"), (3, 0.1, "irrelevant")))
    monkeypatch.setattr(settings, "tavily_api_key", "test-key")
    monkeypatch.setattr(
        web_search, "search_web",
        lambda q: [web_search.WebResult(url="https://x.io/pods", title="Pods", content="Pod facts.")],
    )
    outcome = crag.apply_crag("q", chunks, threshold=0.5, web_fallback=True)
    assert [c.id for c in outcome.chunks][0] == "a"
    web = outcome.chunks[1]
    assert web.url == web.source == "https://x.io/pods" and web.metadata["source_type"] == "web"
    assert outcome.action == "web_fallback" and outcome.web_used and not outcome.insufficient_context


def test_failed_web_search_keeps_local_chunks(fake_structured, chunks, monkeypatch):
    fake_structured(grades((1, 0.6, "partial"), (2, 0.55, "partial"), (3, 0.1, "irrelevant")))
    monkeypatch.setattr(settings, "tavily_api_key", "test-key")

    def broken(query):
        raise TimeoutError("Tavily is down")

    monkeypatch.setattr(web_search, "search_web", broken)
    outcome = crag.apply_crag("q", chunks, threshold=0.5, web_fallback=True)
    assert [c.id for c in outcome.chunks] == ["a", "b"]
    assert outcome.action == "insufficient" and outcome.insufficient_context
    assert not outcome.web_used and "Tavily is down" in outcome.warnings[0]


def test_no_web_key_means_insufficient_without_calling_the_web(fake_structured, chunks, monkeypatch):
    fake_structured(grades((1, 0.2, "irrelevant"), (2, 0.2, "irrelevant"), (3, 0.2, "irrelevant")))
    monkeypatch.setattr(settings, "tavily_api_key", "")
    monkeypatch.setattr(web_search, "search_web", lambda q: pytest.fail("web search must not run"))
    outcome = crag.apply_crag("q", chunks, threshold=0.5, web_fallback=True)
    assert outcome.chunks == [] and outcome.action == "insufficient" and outcome.insufficient_context


def test_grading_failure_returns_the_chunks_ungraded(fake_structured, chunks):
    fake_structured(RuntimeError("OpenAI timeout"))
    outcome = crag.apply_crag("q", chunks, threshold=0.5, web_fallback=False)
    assert outcome.chunks == chunks and outcome.action == "grading_failed"
    assert "OpenAI timeout" in outcome.warnings[0]


def test_no_chunks_skips_the_llm(fake_structured):
    calls = fake_structured(grades())
    outcome = crag.apply_crag("q", [], threshold=0.5, web_fallback=False)
    assert calls == [] and outcome.action == "insufficient"
