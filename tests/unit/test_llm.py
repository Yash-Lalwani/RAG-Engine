from types import SimpleNamespace

import pytest

from rag_engine import llm
from rag_engine.config import settings
from rag_engine.models import EngineError


def test_client_is_created_lazily_and_needs_a_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "")
    llm._client.cache_clear()
    with pytest.raises(EngineError, match="OPENAI_API_KEY"):
        llm._client()
    llm._client.cache_clear()


def test_track_usage_adds_up_calls_inside_the_block():
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4))
    llm._record_usage(response)  # outside a block: ignored
    with llm.track_usage() as usage:
        llm._record_usage(response)
        llm._record_usage(response)
    llm._record_usage(response)  # after the block: ignored
    assert (usage.prompt_tokens, usage.completion_tokens, usage.calls) == (20, 8, 2)
    assert usage.total_tokens == 28
