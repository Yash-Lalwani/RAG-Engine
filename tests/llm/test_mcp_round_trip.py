"""ask -> pending_sql -> approve_sql through the MCP protocol on the seeded k8s-demo collection."""

import psycopg
import pytest

from rag_engine import engine
from rag_engine.config import settings

pytestmark = pytest.mark.llm

DEV_KEY, OTHER_KEY = "llm-test-dev", "llm-test-other"


@pytest.fixture
def call(mcp_call, monkeypatch):
    monkeypatch.setattr(settings, "engine_api_keys", f"dev:{DEV_KEY},other:{OTHER_KEY}")
    engine.setup()
    if "k8s-demo" not in {c.id for c in engine.list_collections()}:
        pytest.skip("k8s-demo is not seeded; run scripts/seed_demo.py")
    return mcp_call


def test_sql_approval_round_trip_over_mcp(call):
    pending = call("ask", {"collection_id": "k8s-demo", "question": "How many P1 incidents are there in total?"},
                   key=DEV_KEY).structured_content
    assert pending["status"] == "pending_sql" and "incidents" in pending["sql"].lower()

    stolen = call("approve_sql", {"query_id": pending["query_id"], "approve": True}, key=OTHER_KEY)
    assert "another caller" in stolen.structured_content["message"]

    done = call("approve_sql", {"query_id": pending["query_id"], "approve": True}, key=DEV_KEY).structured_content
    with psycopg.connect(settings.sql_database_urls["k8s_ops"]) as conn:
        expected = conn.execute("SELECT count(*) FROM incidents WHERE severity = 'P1'").fetchone()[0]
    assert done["status"] == "completed" and str(expected) in done["answer"]
    assert done["metadata"]["token_usage"]["total_tokens"] > 0
