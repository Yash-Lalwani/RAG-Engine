"""Tavily web search, used by the CRAG web fallback."""

from pydantic import BaseModel

from rag_engine.config import settings
from rag_engine.models import EngineError

WEB_TIMEOUT_SECONDS = 15


class WebResult(BaseModel):
    url: str
    title: str
    content: str


def search_web(query: str, max_results: int = 5) -> list[WebResult]:
    if not settings.tavily_api_key:
        raise EngineError("TAVILY_API_KEY is not set")
    from tavily import TavilyClient

    response = TavilyClient(api_key=settings.tavily_api_key).search(
        query=query, max_results=max_results, search_depth="basic", timeout=WEB_TIMEOUT_SECONDS
    )
    return [
        WebResult(url=r["url"], title=r.get("title", ""), content=r.get("content", ""))
        for r in response.get("results", [])
        if r.get("content")
    ]
