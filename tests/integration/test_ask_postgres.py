import psycopg
import pytest

from rag_engine import engine
from rag_engine.config import settings
from rag_engine.graph import builder, nodes
from rag_engine.models import CitedAnswer, SqlDraft, Statement, VerificationResult
from rag_engine.sql.safety import validate_sql

pytestmark = pytest.mark.integration

PENDING_SQL = "SELECT count(*) AS pending_pods FROM pods WHERE status = 'Pending'"


@pytest.fixture
def hybrid_collection(make_collection, ingest_text, monkeypatch):
    cid = make_collection(
        {"sql_database": "k8s_ops", "sql_allowed_tables": ["pods"], "domain_description": "Kubernetes"}
    )
    ingest_text(cid, "pending", "# Pending pods\n\nA pod stays Pending when no node can run it.")

    def fake_answer(question, chunks, sql_result=None, domain_description=""):
        ids = [c.id for c in chunks] + (["sql_results"] if sql_result else [])
        text = f"{sql_result.rows[0]['pending_pods']} pods are pending." if sql_result else "Pods can stay pending."
        return CitedAnswer(statements=[Statement(text=text, chunk_ids=ids)], insufficient_context=False)

    monkeypatch.setattr(nodes, "classify_intent", lambda *args: "hybrid")
    monkeypatch.setattr(nodes, "generate_sql", lambda *args: SqlDraft(sql=PENDING_SQL, explanation="Counts pending pods."))
    monkeypatch.setattr(nodes, "generate_answer", fake_answer)
    monkeypatch.setattr(
        nodes, "verify_citations",
        lambda statements, passages, strict=False: VerificationResult(
            strict=strict, all_supported=True, checked=len(statements), removed_count=0,
            statements=statements, checks=[], failing=[],
        ),
    )
    return cid


def expected_pending_count() -> int:
    with psycopg.connect(settings.sql_database_urls["k8s_ops"]) as conn:
        return conn.execute("SELECT count(*) FROM pods WHERE status = 'Pending'").fetchone()[0]


def test_paused_run_survives_in_postgres_and_resumes(hybrid_collection):
    pending = engine.ask(hybrid_collection, "How many pods are pending and why?", caller="dev")
    assert pending.status == "pending_sql" and pending.sql == validate_sql(PENDING_SQL, ["pods"])

    builder.get_graph.cache_clear()  # a fresh graph and connection: state must come from Postgres
    done = engine.approve_sql(pending.query_id, True, caller="dev")
    assert done.status == "completed" and done.intent == "hybrid"
    assert done.rows_preview == [{"pending_pods": expected_pending_count()}]
    assert done.answer.startswith(f"{expected_pending_count()} pods are pending.")
    assert [s.source for s in done.sources] == ["pending.md", "SQL query results"]
    assert engine.get_graph().get_state(engine._thread(pending.query_id)).values == {}


def test_rejected_hybrid_answers_from_documents(hybrid_collection):
    pending = engine.ask(hybrid_collection, "How many pods are pending and why?", caller="dev")
    done = engine.approve_sql(pending.query_id, False, caller="dev")
    assert done.status == "completed" and done.answer.startswith("Pods can stay pending.")
    assert done.rows_preview == [] and "not approved" in done.metadata.warnings[-1]


def test_setup_removes_abandoned_runs(hybrid_collection):
    pending = engine.ask(hybrid_collection, "How many pods are pending and why?", caller="dev")
    graph = engine.get_graph()
    graph.update_state(engine._thread(pending.query_id), {"created_at": "2020-01-01T00:00:00+00:00"})
    engine.setup()
    assert graph.get_state(engine._thread(pending.query_id)).values == {}
