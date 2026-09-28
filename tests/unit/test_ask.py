from datetime import UTC, datetime, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from rag_engine import engine
from rag_engine.graph import builder
from rag_engine.models import Collection, CollectionSettings, EngineError


@pytest.fixture
def graph(monkeypatch, core):
    """engine.ask() on the in-memory graph with fake core functions and a fake collection."""
    compiled = builder.build_graph(InMemorySaver())
    monkeypatch.setattr(engine, "get_graph", lambda: compiled)
    version = {"n": 1}

    def fake_collection(collection_id):
        return Collection(
            id=collection_id, name="Test", version=version["n"], created_at=datetime.now(UTC),
            settings=CollectionSettings(
                sql_database="k8s_ops", sql_allowed_tables=["pods"], filterable_fields=["team"]
            ),
        )

    monkeypatch.setattr(engine.collections, "get", fake_collection)
    compiled.version = version
    return compiled


def ask(core, intent, question=None, **kwargs):
    core.intent = intent
    return engine.ask("c", question or f"question {intent} {id(core)}", caller="dev", **kwargs)


def test_rag_answer_has_every_result_field(graph, core):
    result = ask(core, "rag", options={"filters": {"team": "a"}, "top_k": 3})
    assert result.status == "completed" and result.intent == "rag"
    assert result.answer.startswith("answer from 1 chunks") and result.statements
    assert result.chunks[0].source == "doc.md" and result.verification is not None
    assert core.calls["search"][0]["filters"] == {"team": "a"}
    assert core.calls["search"][0]["options"] == {"top_k": 3}
    meta = result.metadata
    assert meta.search.crag_action == "correct" and meta.timings_ms["total_ms"] >= 0
    assert meta.cache["answer"] == {"hits": 0, "misses": 1} and not meta.cache_hit
    assert graph.get_state(engine._thread(result.query_id)).values == {}


def test_rag_answers_are_cached_until_the_collection_version_changes(graph, core):
    first = ask(core, "rag", "what is a pod")
    second = ask(core, "rag", "what is a pod")
    assert second.metadata.cache_hit and second.query_id != first.query_id
    assert second.answer == first.answer and len(core.calls["search"]) == 1
    graph.version["n"] += 1
    assert not ask(core, "rag", "what is a pod").metadata.cache_hit


def test_answers_with_warnings_are_not_cached(graph, core, monkeypatch):
    monkeypatch.setattr(engine.cache, "set", lambda *a: pytest.fail("must not cache"))
    core.scores = []
    monkeypatch.setattr(engine.collections, "get", lambda cid: Collection(
        id=cid, name="T", created_at=datetime.now(UTC), settings=CollectionSettings(self_rag=True)))
    result = ask(core, "rag")
    assert result.status == "completed" and "Self-check failed" in result.metadata.warnings[0]


@pytest.mark.parametrize("intent", ["sql", "hybrid"])
def test_sql_and_hybrid_answers_are_never_served_from_the_answer_cache(graph, core, intent):
    first = ask(core, intent, "same question")
    assert first.status == "pending_sql"
    assert engine.approve_sql(first.query_id, True, caller="dev").status == "completed"
    again = ask(core, intent, "same question")
    assert again.status == "pending_sql" and not again.metadata.cache_hit


def test_pending_then_approved(graph, core):
    pending = ask(core, "sql")
    assert pending.status == "pending_sql" and pending.sql.endswith("LIMIT 200")
    assert pending.sql_explanation == "Counts pods." and "run_sql" not in core.calls

    done = engine.approve_sql(pending.query_id, True, caller="dev")
    assert done.status == "completed" and done.query_id == pending.query_id
    assert len(done.rows_preview) == engine.PREVIEW_ROWS and done.metadata.sql_row_count == 60
    assert done.sources[0].source == "SQL query results"
    assert {"generate_sql", "approve", "execute_sql", "generate"} <= set(done.metadata.timings_ms)
    assert graph.get_state(engine._thread(pending.query_id)).values == {}


def test_rejected_sql(graph, core):
    pending = ask(core, "sql")
    rejected = engine.approve_sql(pending.query_id, False, caller="dev")
    assert rejected.status == "completed" and rejected.answer == "Query not approved."


def test_unknown_query_id_is_an_error_result(graph):
    result = engine.approve_sql("no-such-id", True, caller="dev")
    assert result.status == "error" and "Unknown query_id" in result.message


def test_another_caller_cannot_approve(graph, core):
    pending = ask(core, "sql")
    other = engine.approve_sql(pending.query_id, True, caller="astra")
    assert other.status == "error" and "another caller" in other.message
    assert "run_sql" not in core.calls
    assert engine.approve_sql(pending.query_id, True, caller="dev").status == "completed"


def test_a_run_cannot_be_approved_twice(graph, core):
    pending = ask(core, "sql")
    engine.approve_sql(pending.query_id, True, caller="dev")
    again = engine.approve_sql(pending.query_id, True, caller="dev")
    assert again.status == "error" and len(core.calls["run_sql"]) == 1


def test_expired_runs_are_rejected_and_cleaned_up(graph, core, monkeypatch):
    pending = ask(core, "sql")
    monkeypatch.setattr(engine, "PAUSE_LIMIT", timedelta(seconds=-1))
    result = engine.approve_sql(pending.query_id, True, caller="dev")
    assert result.status == "error" and "expired" in result.message
    assert graph.get_state(engine._thread(pending.query_id)).values == {}


def test_delete_expired_runs(graph, core, monkeypatch):
    fresh, old = ask(core, "sql"), ask(core, "hybrid")
    long_ago = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    graph.update_state(engine._thread(old.query_id), {"created_at": long_ago})
    assert engine.delete_expired_runs() == 1
    assert graph.get_state(engine._thread(old.query_id)).values == {}
    assert engine.approve_sql(fresh.query_id, True, caller="dev").status == "completed"


def test_a_failing_run_returns_an_error_result(graph, core, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("OpenAI is down")

    monkeypatch.setattr(builder.nodes, "generate_answer", broken)
    result = ask(core, "rag")
    assert result.status == "error" and "OpenAI is down" in result.message
    assert graph.get_state(engine._thread(result.query_id)).values == {}


def test_bad_input_raises_engine_errors(graph, core):
    with pytest.raises(EngineError, match="Invalid settings"):
        ask(core, "rag", options={"serch_mode": "dense"})
    with pytest.raises(EngineError, match="filterable_fields"):
        ask(core, "rag", options={"filters": {"owner": "sam"}})
    with pytest.raises(EngineError, match="empty"):
        engine.ask("c", "   ", caller="dev")
    with pytest.raises(TypeError):
        engine.ask("c", "question")
