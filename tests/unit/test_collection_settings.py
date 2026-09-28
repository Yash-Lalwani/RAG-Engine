import pytest

from rag_engine.models import CollectionSettings, EngineError


def test_defaults_match_the_plan():
    s = CollectionSettings()
    assert (s.search_mode, s.top_k, s.fetch_k, s.rrf_k) == ("hybrid", 5, 20, 60)
    assert s.rerank and s.reranker == "local"
    assert not s.hyde and s.crag and not s.crag_web_fallback and not s.self_rag
    assert (s.crag_threshold, s.self_rag_threshold) == (0.5, 0.7)
    assert s.citation_mode == "verify" and s.sql_database is None and s.filterable_fields == []


def test_merged_applies_overrides_and_keeps_the_rest():
    base = CollectionSettings(filterable_fields=["team"])
    merged = base.merged({"search_mode": "dense", "top_k": 3})
    assert merged.search_mode == "dense" and merged.top_k == 3
    assert merged.filterable_fields == ["team"]
    assert base.search_mode == "hybrid"


@pytest.mark.parametrize(
    "overrides",
    [
        {"serch_mode": "dense"},
        {"search_mode": "fuzzy"},
        {"top_k": 0},
        {"top_k": 30, "fetch_k": 20},
        {"filterable_fields": ["bad.name"]},
    ],
)
def test_invalid_settings_raise_engine_error(overrides):
    with pytest.raises(EngineError):
        CollectionSettings().merged(overrides)
