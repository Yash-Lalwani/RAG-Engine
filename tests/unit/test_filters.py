import pytest
from qdrant_client import models

from rag_engine.models import EngineError
from rag_engine.retrieval.search import check_filters
from rag_engine.retrieval.vector_store import build_filter, point_id


def test_allowed_filters_pass_through():
    filters = {"team": "alpha", "year": 2024, "active": True, "ids": ["a", "b"], "n": [1, 2]}
    assert check_filters(filters, ["team", "year", "active", "ids", "n"]) == filters


def test_no_filters():
    assert check_filters(None, []) == {}


def test_field_not_filterable_is_a_clear_error():
    with pytest.raises(EngineError, match="not in this collection's filterable_fields"):
        check_filters({"team": "alpha"}, ["source_type"])


@pytest.mark.parametrize("value", [1.5, {"a": 1}, [], ["a", 1], [True, False], None])
def test_unsupported_filter_values(value):
    with pytest.raises(EngineError):
        check_filters({"team": value}, ["team"])


def test_build_filter_always_restricts_to_the_collection():
    query_filter = build_filter("coll-a", metadata_filters={"team": "alpha", "ids": ["x", "y"]})
    keys = [condition.key for condition in query_filter.must]
    assert keys == ["collection_id", "metadata.team", "metadata.ids"]
    assert query_filter.must[0].match == models.MatchValue(value="coll-a")
    assert query_filter.must[2].match == models.MatchAny(any=["x", "y"])


def test_point_ids_are_deterministic_and_distinct():
    assert point_id("c", "doc", 0) == point_id("c", "doc", 0)
    assert len({point_id("c", "doc", 0), point_id("c", "doc", 1), point_id("d", "doc", 0)}) == 3
