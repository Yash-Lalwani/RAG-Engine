from __future__ import annotations

from abc import ABC, abstractmethod

from rag_engine.config import settings
from rag_engine.generation.generator import _generate
from rag_engine.models import ChatResponse, RetrievedChunk
from rag_engine.retrieval.search import _retrieve


class SkippedIntent(Exception):
    pass

class Invoker(ABC):

    @abstractmethod
    def invoke(
        self, question: str, flags: dict, intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        ...


class ServiceInvoker(Invoker):
    SUPPORTED_INTENTS = {"rag", "web_fallback"}

    def invoke(
        self, question: str, flags: dict, intent: str
    ) -> tuple[ChatResponse, list[RetrievedChunk]]:
        if intent not in self.SUPPORTED_INTENTS:
            raise SkippedIntent(f"intent={intent} not supported in service mode")

        if intent == "web_fallback" and not settings.tavily_api_key:
            raise SkippedIntent("tavily_unset: TAVILY_API_KEY not configured")

        chunks = _retrieve(question, flags=flags)
        return _generate(question, chunks, flags=flags), chunks
