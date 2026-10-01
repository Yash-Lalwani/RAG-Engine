"""The MCP server over real streamable HTTP with a real MCP client; the engine is faked."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from rag_engine import engine, llm
from rag_engine.config import settings
from rag_engine.guardrails import rate_limit, token_budget
from rag_engine.models import AskResult, Collection, CollectionSettings, EngineError, HealthStatus

DEV_KEY, ASTRA_KEY = "test-dev-key", "test-astra-key"


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr(settings, "engine_api_keys", f"dev:{DEV_KEY},astra:{ASTRA_KEY}")
    monkeypatch.setattr(type(rate_limit.cache), "redis", property(lambda self: None))
    monkeypatch.setattr(rate_limit, "_memory", rate_limit.defaultdict(rate_limit.deque))
    monkeypatch.setattr(token_budget, "_memory", {})
    calls = []

    def fake_ask(collection_id, question, options=None, *, caller):
        calls.append(("ask", caller))
        llm._record_usage(SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20)))
        return AskResult(status="pending_sql", query_id="q1", intent="sql", sql="SELECT 1")

    def fake_approve(query_id, approve, *, caller):
        calls.append(("approve_sql", caller))
        return AskResult(status="completed", query_id=query_id, intent="sql", answer="1 row. [1]")

    def fake_create(collection_id, name, description="", settings=None):
        raise EngineError(f"Collection {collection_id!r} already exists")

    def fake_list_documents(collection_id):
        raise RuntimeError("internal detail: postgresql://secret@db")

    monkeypatch.setattr(engine, "health", lambda: HealthStatus(status="ok", postgres=True, qdrant=True, redis="not configured"))
    monkeypatch.setattr(engine, "list_collections", lambda: [
        Collection(id="k8s-demo", name="Demo", settings=CollectionSettings(), created_at=datetime.now(UTC))])
    monkeypatch.setattr(engine, "ask", fake_ask)
    monkeypatch.setattr(engine, "approve_sql", fake_approve)
    monkeypatch.setattr(engine, "create_collection", fake_create)
    monkeypatch.setattr(engine, "list_documents", fake_list_documents)
    return calls


@pytest.fixture
def call(mcp_call):
    def dev_call(tool, arguments=None, key=DEV_KEY, header=None):
        return mcp_call(tool, arguments, key=key, header=header)

    return dev_call


def text(result) -> str:
    return result.content[0].text


def test_all_tools_are_listed_and_rerank_has_no_backend(call):
    tools = {t.name: t for t in call("__list__", key=None).tools}
    assert set(tools) == {
        "health", "create_collection", "update_collection", "list_collections", "delete_collection",
        "ingest_document", "list_documents", "delete_document", "search", "rerank",
        "verify_citations", "ask", "approve_sql",
    }
    assert set(tools["rerank"].input_schema["properties"]) == {"query", "passages", "top_k"}


def test_health_needs_no_key(call):
    result = call("health", key=None)
    assert not result.is_error and result.structured_content["status"] == "ok"


@pytest.mark.parametrize("kwargs", [{"key": None}, {"key": "wrong-key"}, {"header": f"Token {DEV_KEY}"}])
def test_missing_or_bad_key_is_rejected(call, kwargs):
    result = call("list_collections", **kwargs)
    assert result.is_error and text(result).endswith("Unauthorized: missing or invalid API key")


def test_valid_key_calls_the_engine(call):
    result = call("list_collections")
    assert not result.is_error and result.structured_content["result"][0]["id"] == "k8s-demo"


def test_engine_errors_are_passed_through_and_crashes_stay_generic(call):
    result = call("create_collection", {"collection_id": "k8s-demo", "name": "x"})
    assert result.is_error and text(result).endswith("Collection 'k8s-demo' already exists")
    crash = call("list_documents", {"collection_id": "k8s-demo"})
    assert crash.is_error and "secret" not in text(crash)


def test_question_length_is_enforced_by_the_schema(call):
    result = call("ask", {"collection_id": "k8s-demo", "question": "x" * 2001})
    assert result.is_error and "2000" in text(result)


def test_rate_limit_blocks_tools_and_ask(call, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    call("list_collections")
    call("list_collections")
    blocked = call("list_collections")
    assert blocked.is_error and "Blocked: Rate limit reached" in text(blocked)
    answer = call("ask", {"collection_id": "k8s-demo", "question": "How many pods?"})
    assert answer.structured_content["status"] == "blocked"
    assert "Rate limit" in answer.structured_content["message"]
    assert call("list_collections", key=ASTRA_KEY).is_error is False  # per caller


def test_budget_records_actual_tokens_and_then_blocks(call, monkeypatch):
    monkeypatch.setattr(settings, "daily_token_budget", 150)
    args = {"collection_id": "k8s-demo", "question": "How many pods?"}
    assert call("ask", args).structured_content["status"] == "pending_sql"
    assert token_budget.tokens_used("dev") == 120
    call("ask", args)
    blocked = call("ask", args)
    assert blocked.structured_content["status"] == "blocked" and "budget" in blocked.structured_content["message"]
    assert token_budget.tokens_used("astra") == 0


def test_the_key_decides_the_caller_and_approve_sql_is_guarded(call, fake_engine):
    call("ask", {"collection_id": "k8s-demo", "question": "How many pods?"}, key=ASTRA_KEY)
    assert call("approve_sql", {"query_id": "q1", "approve": True}, key=None).is_error
    done = call("approve_sql", {"query_id": "q1", "approve": True})
    assert done.structured_content["answer"] == "1 row. [1]"
    assert fake_engine == [("ask", "astra"), ("approve_sql", "dev")]


INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.mark.parametrize(("host", "accepted"), [("rag.example.test", True), ("evil.example.com", False)])
def test_only_allowed_hosts_reach_the_mcp_endpoint(mcp_url, host, accepted):
    import httpx

    response = httpx.post(mcp_url, json=INITIALIZE, headers={**MCP_HEADERS, "Host": host})
    assert (response.status_code == 200) is accepted, response.status_code


def test_healthz_needs_no_key(mcp_url):
    import httpx

    response = httpx.get(mcp_url.replace("/mcp", "/healthz"))
    assert response.status_code == 200 and response.json() == {"status": "ok"}


def test_get_stream_is_refused_so_idle_clients_cannot_keep_the_server_awake(mcp_url):
    import httpx

    response = httpx.get(mcp_url, headers={"Accept": "text/event-stream"}, timeout=5)
    assert response.status_code == 405 and response.headers["allow"] == "POST, DELETE"


def test_uploads_larger_than_the_default_4mb_are_accepted(call, monkeypatch):
    import base64

    from rag_engine.models import IngestResult

    seen = {}

    def fake_ingest(collection_id, file_path=None, content_base64=None, filename=None, doc_id=None, metadata=None):
        seen["bytes"] = len(base64.b64decode(content_base64))
        return IngestResult(collection_id=collection_id, doc_id=filename, status="ingested", chunk_count=1)

    monkeypatch.setattr(engine, "ingest_document", fake_ingest)
    content = base64.b64encode(b"x" * 6_000_000).decode()
    result = call("ingest_document", {"collection_id": "k8s-demo", "content_base64": content, "filename": "big.txt"})
    assert not result.is_error and seen["bytes"] == 6_000_000
