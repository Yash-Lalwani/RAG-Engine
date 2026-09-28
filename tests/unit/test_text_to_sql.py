import pytest

from rag_engine.models import EngineError, SqlDraft
from rag_engine.sql import text_to_sql


@pytest.fixture
def fake_schema(monkeypatch):
    monkeypatch.setattr(text_to_sql, "describe_schema", lambda db, tables: "clusters(name text)")


def test_generated_sql_is_cleaned_and_cached(fake_schema, fake_structured):
    calls = fake_structured(SqlDraft(sql=" SELECT name FROM clusters; ", explanation=" Lists names. "))
    question = f"list clusters {id(calls)}"
    first = text_to_sql.generate_sql(question, "k8s_ops", ["clusters"], "Kubernetes ops")
    second = text_to_sql.generate_sql(question, "k8s_ops", ["clusters"])
    assert first == second == SqlDraft(sql="SELECT name FROM clusters", explanation="Lists names.")
    assert len(calls) == 1 and "Kubernetes ops" in calls[0]["system"]
    assert "clusters(name text)" in calls[0]["user"]


def test_cache_key_depends_on_allowed_tables(fake_schema, fake_structured):
    calls = fake_structured(SqlDraft(sql="SELECT name FROM clusters", explanation="x"))
    question = f"list clusters {id(calls)}"
    text_to_sql.generate_sql(question, "k8s_ops", ["clusters"])
    text_to_sql.generate_sql(question, "k8s_ops", ["clusters", "nodes"])
    assert len(calls) == 2


def test_unsafe_sql_is_returned_but_never_cached(fake_schema, fake_structured):
    calls = fake_structured(SqlDraft(sql="DELETE FROM clusters", explanation="x"))
    question = f"delete everything {id(calls)}"
    assert text_to_sql.generate_sql(question, "k8s_ops", ["clusters"]).sql == "DELETE FROM clusters"
    text_to_sql.generate_sql(question, "k8s_ops", ["clusters"])
    assert len(calls) == 2


def test_no_allowed_tables_is_an_error():
    with pytest.raises(EngineError, match="No SQL tables"):
        text_to_sql.generate_sql("q", "k8s_ops", [])
