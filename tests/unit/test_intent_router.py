import uuid

from rag_engine.models import CollectionSettings
from rag_engine.routing.intent_router import classify_intent

SQL_SETTINGS = CollectionSettings(
    sql_database="k8s_ops", sql_allowed_tables=["pods", "incidents"], domain_description="K8s docs"
)


def route(intent):
    return lambda schema: schema(intent=intent)


def test_collection_without_sql_database_is_always_rag_without_an_llm_call(fake_structured):
    calls = fake_structured(route("sql"))
    assert classify_intent("how many pods?", "docs", CollectionSettings()) == "rag"
    assert calls == []


def test_llm_decides_and_the_prompt_lists_domain_and_tables(fake_structured):
    calls = fake_structured(route("hybrid"))
    question = f"how many pods crash and why? {uuid.uuid4()}"
    assert classify_intent(question, "k8s", SQL_SETTINGS) == "hybrid"
    assert "pods, incidents" in calls[0]["system"] and "K8s docs" in calls[0]["system"]


def test_intent_is_cached_per_collection_and_tables(fake_structured):
    calls = fake_structured(route("sql"))
    question = f"count incidents {uuid.uuid4()}"
    classify_intent(question, "k8s", SQL_SETTINGS)
    classify_intent(question, "k8s", SQL_SETTINGS)
    assert len(calls) == 1
    classify_intent(question, "k8s", SQL_SETTINGS.merged({"sql_allowed_tables": ["pods"]}))
    classify_intent(question, "other", SQL_SETTINGS)
    assert len(calls) == 3
