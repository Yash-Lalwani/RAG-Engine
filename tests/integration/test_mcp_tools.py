"""The real engine behind the MCP tools (Docker services; fake dense embeddings, no LLM calls)."""

import base64
import uuid

import pytest

from rag_engine import engine
from rag_engine.config import settings
from rag_engine.guardrails import rate_limit

pytestmark = pytest.mark.integration

KEY = "integration-key"


@pytest.fixture
def call(mcp_call, monkeypatch):
    monkeypatch.setattr(settings, "engine_api_keys", f"dev:{KEY}")
    monkeypatch.setattr(rate_limit, "_memory", rate_limit.defaultdict(rate_limit.deque))
    return lambda tool, arguments=None: mcp_call(tool, arguments, key=KEY)


def test_health_reports_real_services(mcp_call):
    result = mcp_call("health").structured_content
    assert result["postgres"] and result["qdrant"] and result["status"] == "ok"


def test_collection_ingest_search_rerank_and_delete_through_mcp(call):
    cid = f"test-mcp-{uuid.uuid4().hex[:6]}"
    try:
        created = call("create_collection", {"collection_id": cid, "name": "MCP test",
                                             "settings": {"rerank": False, "crag": False}})
        assert not created.is_error and created.structured_content["id"] == cid

        text = "# Pods\n\nA pod is the smallest deployable unit. Pods run containers on nodes."
        ingested = call("ingest_document", {
            "collection_id": cid, "filename": "pods.md",
            "content_base64": base64.b64encode(text.encode()).decode(),
        })
        assert ingested.structured_content["status"] == "ingested"
        docs = call("list_documents", {"collection_id": cid}).structured_content["result"]
        assert [d["doc_id"] for d in docs] == ["pods.md"]

        found = call("search", {"collection_id": cid, "query": "smallest deployable unit"})
        assert found.structured_content["chunks"][0]["source"] == "pods.md"

        reranked = call("rerank", {"query": "what is a pod", "passages": [
            {"id": "a", "text": "Ingress routes HTTP traffic."},
            {"id": "b", "text": "A pod is the smallest deployable unit."}]})
        assert reranked.structured_content["result"][0]["id"] == "b"

        missing = call("search", {"collection_id": "no-such-collection", "query": "pod"})
        assert missing.is_error and "does not exist" in missing.content[0].text

        assert not call("delete_document", {"collection_id": cid, "doc_id": "pods.md"}).is_error
        assert call("list_documents", {"collection_id": cid}).structured_content["result"] == []
    finally:
        if cid in {c.id for c in engine.list_collections()}:
            engine.delete_collection(cid)
