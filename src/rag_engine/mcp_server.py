"""The Engine as an MCP server over streamable HTTP.

Tools only wire auth and the caller guardrails (G3 rate limit, G4 token budget) around engine
functions; the content guardrails (G1, G2, G6) run inside the engine itself.

Run: uv run python -m rag_engine.mcp_server   (serves http://MCP_HOST:MCP_PORT/mcp)
Clients send the header  Authorization: Bearer <api key>
"""

import logging
from collections.abc import Callable
from typing import Annotated, Any

import uvicorn
from langsmith import tracing_context
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from rag_engine import config, engine
from rag_engine.guardrails.api_keys import authenticate
from rag_engine.guardrails.rate_limit import check_rate_limit
from rag_engine.guardrails.token_budget import check_budget, record_usage
from rag_engine.ingestion.pipeline import MAX_DOCUMENT_BYTES
from rag_engine.llm import track_usage
from rag_engine.models import (
    AskResult,
    AuthError,
    Blocked,
    Collection,
    DeleteResult,
    DocumentRecord,
    EngineError,
    HealthStatus,
    IngestResult,
    Passage,
    RerankedPassage,
    SearchResult,
    Statement,
    VerificationResult,
)

logger = logging.getLogger(__name__)

LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
# A document of MAX_DOCUMENT_BYTES is about a third larger as base64, plus the JSON around it.
MAX_REQUEST_BYTES = int(MAX_DOCUMENT_BYTES * 1.4) + 1024 * 1024

Question = Annotated[str, Field(min_length=1, max_length=2000)]
Options = Annotated[
    dict[str, Any] | None,
    Field(description="Per-call overrides of the collection settings, e.g. {\"search_mode\": \"dense\", \"top_k\": 3}"),
]
Filters = Annotated[
    dict[str, Any] | None,
    Field(description="Exact-match metadata filters on the collection's filterable_fields, e.g. {\"section\": \"concepts\"} or {\"section\": [\"tasks\", \"tutorials\"]}"),
]

server = MCPServer(
    "rag-engine",
    instructions=(
        "A retrieval-augmented question answering engine over document collections. "
        "Use list_collections to see what exists, search for ranked passages, and ask for a "
        "cited answer. When ask returns status 'pending_sql', show the SQL to the user and "
        "call approve_sql only after they decide."
    ),
)


def run_tool[T](
    ctx: Context,
    call: Callable[[str], T],
    *,
    budget: bool = False,
    on_blocked: Callable[[str], T] | None = None,
) -> T:
    """Auth, G3 and (for tools that use the LLM) G4 around one engine call. `call` gets the caller."""
    try:
        caller = authenticate(_bearer_key(ctx))
    except AuthError:
        raise ToolError("Unauthorized: missing or invalid API key") from None
    try:
        check_rate_limit(caller)
        if budget:
            check_budget(caller)
        with track_usage() as usage, tracing_context(tags=[f"caller:{caller}"], metadata={"caller": caller}):
            try:
                return call(caller)
            finally:
                if budget:
                    record_usage(caller, usage.total_tokens)
    except Blocked as blocked:
        if on_blocked is not None:
            return on_blocked(str(blocked))
        raise ToolError(f"Blocked: {blocked}") from None
    except EngineError as error:
        raise ToolError(str(error)) from None


def _bearer_key(ctx: Context) -> str | None:
    scheme, _, key = (ctx.headers or {}).get("authorization", "").partition(" ")
    return key.strip() if scheme.lower() == "bearer" else None


def _blocked_answer(reason: str) -> AskResult:
    return AskResult(status="blocked", query_id="", message=reason)


@server.tool()
def health() -> HealthStatus:
    """Check that the Engine's services are up (Postgres, Qdrant, Redis). Needs no API key and
    makes no paid API calls."""
    return engine.health()


@server.tool()
def list_collections(ctx: Context) -> list[Collection]:
    """List all collections with their settings, document counts and versions."""
    return run_tool(ctx, lambda caller: engine.list_collections())


@server.tool()
def create_collection(
    ctx: Context,
    collection_id: str,
    name: str,
    description: str = "",
    settings: dict[str, Any] | None = None,
) -> Collection:
    """Create a collection: a separate set of documents with its own search settings.
    collection_id uses lowercase letters, digits, '-' or '_'. settings optionally overrides the
    defaults, e.g. {"search_mode": "hybrid", "top_k": 5, "filterable_fields": ["source_type"]}."""
    return run_tool(ctx, lambda caller: engine.create_collection(collection_id, name, description, settings))


@server.tool()
def update_collection(ctx: Context, collection_id: str, settings: dict[str, Any]) -> Collection:
    """Change some settings of a collection. The given keys are merged into the current settings."""
    return run_tool(ctx, lambda caller: engine.update_collection(collection_id, settings))


@server.tool()
def delete_collection(ctx: Context, collection_id: str) -> DeleteResult:
    """Delete a collection with all its documents. This cannot be undone."""
    return run_tool(ctx, lambda caller: engine.delete_collection(collection_id))


@server.tool()
def ingest_document(
    ctx: Context,
    collection_id: str,
    content_base64: str | None = None,
    filename: str | None = None,
    file_path: str | None = None,
    doc_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> IngestResult:
    """Add or update one document (PDF, DOCX, HTML, Markdown, TXT, ...; at most 20 MB).
    Send the file as content_base64 together with its filename, or give a file_path inside the
    server's ingest directory. doc_id defaults to the filename. Re-sending identical content
    returns status 'unchanged'; changed content or metadata returns 'replaced'."""
    return run_tool(
        ctx,
        lambda caller: engine.ingest_document(
            collection_id, file_path, content_base64, filename, doc_id, metadata
        ),
    )


@server.tool()
def list_documents(ctx: Context, collection_id: str) -> list[DocumentRecord]:
    """List the documents in a collection."""
    return run_tool(ctx, lambda caller: engine.list_documents(collection_id))


@server.tool()
def delete_document(ctx: Context, collection_id: str, doc_id: str) -> DeleteResult:
    """Delete one document and its chunks from a collection."""
    return run_tool(ctx, lambda caller: engine.delete_document(collection_id, doc_id))


@server.tool()
def search(
    ctx: Context,
    collection_id: str,
    query: Question,
    top_k: int | None = None,
    filters: Filters = None,
    options: Options = None,
) -> SearchResult:
    """Find the passages in a collection that best match the query (hybrid dense + keyword search,
    reranking and relevance grading, as set for the collection). Returns ranked chunks with their
    source, scores and grade. Use this when you want passages rather than a finished answer."""
    return run_tool(
        ctx, lambda caller: engine.search(collection_id, query, top_k, filters, options), budget=True
    )


@server.tool()
def rerank(
    ctx: Context, query: str, passages: list[Passage], top_k: int | None = None
) -> list[RerankedPassage]:
    """Reorder passages you already have (for example live results from another system) by how
    well they answer the query. Each passage needs an id and text."""
    return run_tool(ctx, lambda caller: engine.rerank(query, passages, top_k))


@server.tool()
def verify_citations(
    ctx: Context, statements: list[Statement], passages: list[Passage], strict: bool = False
) -> VerificationResult:
    """Check that each statement is supported by the passages it cites (chunk_ids are passage ids).
    Default mode removes unsupported statements; strict=true keeps them and flags the result."""
    return run_tool(
        ctx, lambda caller: engine.verify_citations(statements, passages, strict), budget=True
    )


@server.tool()
def ask(ctx: Context, collection_id: str, question: Question, options: Options = None) -> AskResult:
    """Answer a question from a collection, with numbered citations. The Engine decides whether it
    needs documents, its SQL database, or both. If the result has status 'pending_sql', show the
    SQL and its explanation to the user and wait for their decision before calling approve_sql
    with the returned query_id. Put metadata filters in options as {"filters": {...}}."""
    return run_tool(
        ctx,
        lambda caller: engine.ask(collection_id, question, options, caller=caller),
        budget=True,
        on_blocked=_blocked_answer,
    )


@server.tool()
def approve_sql(ctx: Context, query_id: str, approve: bool) -> AskResult:
    """Run (approve=true) or reject (approve=false) the SQL of a paused ask. Only the caller who
    asked can approve, within 24 hours. A rejected SQL question ends with 'Query not approved';
    a rejected hybrid question is answered from documents only."""
    return run_tool(
        ctx,
        lambda caller: engine.approve_sql(query_id, approve, caller=caller),
        budget=True,
        on_blocked=_blocked_answer,
    )


@server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def healthz(request: Request) -> JSONResponse:
    """Liveness check for the hosting platform: the process is up. Needs no key, calls nothing."""
    return JSONResponse({"status": "ok"})


def build_app() -> Starlette:
    """The HTTP app: stateless MCP (no per-client session to lose on a restart), DNS-rebinding
    protection that also accepts the public hostnames, and room for 20 MB base64 uploads."""
    allowed = LOCAL_HOSTS + [h for host in config.settings.allowed_hosts for h in (host, f"{host}:*")]
    return server.streamable_http_app(
        stateless_http=True,
        max_request_body_size=MAX_REQUEST_BYTES,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=allowed
        ),
    )


def warn_about_api_keys() -> None:
    callers = config.settings.api_key_callers
    if not callers:
        logger.warning("ENGINE_API_KEYS is empty: every tool except health will be rejected")
    for key, caller in callers.items():
        if key.startswith("change-me"):
            logger.warning("The API key for %r still starts with 'change-me'; replace it in .env", caller)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    warn_about_api_keys()
    engine.setup()
    engine.warm_up()
    uvicorn.run(build_app(), host=config.settings.mcp_host, port=config.settings.mcp_port)


if __name__ == "__main__":
    main()
