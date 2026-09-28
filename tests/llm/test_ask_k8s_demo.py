"""End to end on the seeded k8s-demo collection (run scripts/seed_demo.py first). Real OpenAI calls."""

import psycopg
import pytest

from rag_engine import engine
from rag_engine.config import settings

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module", autouse=True)
def k8s_demo():
    engine.setup()
    if "k8s-demo" not in {c.id for c in engine.list_collections()}:
        pytest.skip("k8s-demo is not seeded; run scripts/seed_demo.py")


def db_value(sql: str):
    with psycopg.connect(settings.sql_database_urls["k8s_ops"]) as conn:
        return conn.execute(sql).fetchone()[0]


def test_rag_question():
    result = engine.ask("k8s-demo", "How do I roll back a deployment?", caller="dev")
    assert result.status == "completed" and result.intent == "rag"
    assert "rollout undo" in result.answer and result.sources
    assert result.verification.all_supported


def test_sql_question_with_approval():
    pending = engine.ask("k8s-demo", "How many P1 incidents are there in total?", caller="dev")
    assert pending.status == "pending_sql" and pending.intent == "sql"
    assert "incidents" in pending.sql.lower()
    done = engine.approve_sql(pending.query_id, True, caller="dev")
    expected = db_value("SELECT count(*) FROM incidents WHERE severity = 'P1'")
    assert done.status == "completed" and str(expected) in done.answer


def test_hybrid_question_with_approval():
    question = "How many pods are currently in Pending status, and what can keep a pod in Pending?"
    pending = engine.ask("k8s-demo", question, caller="dev")
    assert pending.status == "pending_sql" and pending.intent == "hybrid"
    done = engine.approve_sql(pending.query_id, True, caller="dev")
    expected = db_value("SELECT count(*) FROM pods WHERE status = 'Pending'")
    assert done.status == "completed" and str(expected) in done.answer
    assert {"SQL query results"} < {s.source for s in done.sources}
    assert done.verification.removed_count == 0
