"""OpenAI chat calls (plain text and structured outputs) with per-request token counting."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel

from rag_engine.config import settings
from rag_engine.models import EngineError, TokenUsage

_current_usage: ContextVar[TokenUsage | None] = ContextVar("current_usage", default=None)


@contextmanager
def track_usage() -> Iterator[TokenUsage]:
    """Count the tokens of every LLM call made inside the `with` block."""
    outer = _current_usage.get()
    usage = TokenUsage()
    token = _current_usage.set(usage)
    try:
        yield usage
    finally:
        _current_usage.reset(token)
        if outer is not None:  # nested blocks also count toward the enclosing block
            for model, tokens in usage.by_model.items():
                outer.add(model, tokens["prompt"], tokens["completion"], calls=0)
            outer.calls += usage.calls


@lru_cache
def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise EngineError("OPENAI_API_KEY is not set")
    return wrap_openai(OpenAI(api_key=settings.openai_api_key))


def generate_text(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
    n: int = 1,
) -> list[str]:
    """Return `n` completions (one API call; `n > 1` bills the prompt only once)."""
    response = _client().chat.completions.create(
        model=model or settings.llm_model_strong,
        messages=_messages(system, user),
        temperature=temperature,
        n=n,
    )
    _record_usage(response)
    return [choice.message.content or "" for choice in response.choices]


def generate_structured[Schema: BaseModel](
    system: str,
    user: str,
    schema: type[Schema],
    model: str | None = None,
    temperature: float = 0.0,
) -> Schema:
    """Return the model's answer parsed into `schema` (OpenAI structured outputs)."""
    response = _client().chat.completions.parse(
        model=model or settings.llm_model_strong,
        messages=_messages(system, user),
        temperature=temperature,
        response_format=schema,
    )
    _record_usage(response)
    message = response.choices[0].message
    if message.refusal:
        raise EngineError(f"The model refused the request: {message.refusal}")
    if message.parsed is None:
        raise EngineError("The model returned no structured output")
    return message.parsed


def _messages(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _record_usage(response) -> None:
    usage = _current_usage.get()
    if usage is not None and response.usage is not None:
        model = getattr(response, "model", None) or "unknown"
        usage.add(model, response.usage.prompt_tokens, response.usage.completion_tokens)
