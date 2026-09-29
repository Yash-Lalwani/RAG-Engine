import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from rag_engine.mcp_server import server


@pytest.fixture(scope="session", autouse=True)
def no_tracing():
    """Tests never send traces to LangSmith, even when .env has a key."""
    from rag_engine.config import settings
    from rag_engine.tracing import configure_tracing

    settings.langsmith_tracing = False
    configure_tracing()


@pytest.fixture(scope="session")
def mcp_url():
    """The MCP server running in a background thread on a free local port."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    http_server = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), port=port, log_level="warning"))
    threading.Thread(target=http_server.run, daemon=True).start()
    while not http_server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    http_server.should_exit = True


@pytest.fixture
def mcp_call(mcp_url):
    """call(tool, arguments, key=..., header=...) with a real MCP client; tool "__list__" lists tools."""

    def call(tool, arguments=None, key=None, header=None):
        async def run():
            headers = {"Authorization": header or f"Bearer {key}"} if (key or header) else {}
            async with create_mcp_http_client(headers=headers) as http:
                async with streamable_http_client(mcp_url, http_client=http) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        if tool == "__list__":
                            return await session.list_tools()
                        return await session.call_tool(tool, arguments or {})

        return asyncio.run(run())

    return call
