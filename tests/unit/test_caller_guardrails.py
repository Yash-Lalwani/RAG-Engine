import pytest

from rag_engine.config import settings
from rag_engine.guardrails import rate_limit, token_budget
from rag_engine.guardrails.api_keys import authenticate
from rag_engine.models import AuthError, Blocked


@pytest.fixture(autouse=True)
def memory_backends(monkeypatch):
    """Force the in-memory stores and start each test empty."""
    monkeypatch.setattr(type(rate_limit.cache), "redis", property(lambda self: None))
    monkeypatch.setattr(rate_limit, "_memory", rate_limit.defaultdict(rate_limit.deque))
    monkeypatch.setattr(token_budget, "_memory", {})


def test_valid_key_returns_the_caller(monkeypatch):
    monkeypatch.setattr(settings, "engine_api_keys", "dev:key-dev,astra:key-astra")
    assert authenticate("key-astra") == "astra"


@pytest.mark.parametrize("key", [None, "", "wrong", "key-dev "])
def test_bad_or_missing_key_is_rejected(monkeypatch, key):
    monkeypatch.setattr(settings, "engine_api_keys", "dev:key-dev")
    with pytest.raises(AuthError):
        authenticate(key)


def test_no_keys_configured_rejects_everything(monkeypatch):
    monkeypatch.setattr(settings, "engine_api_keys", "")
    with pytest.raises(AuthError):
        authenticate("anything")


def test_rate_limit_triggers_per_caller(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    for _ in range(3):
        rate_limit.check_rate_limit("layer")
    with pytest.raises(Blocked, match="at most 3 requests per minute"):
        rate_limit.check_rate_limit("layer")
    rate_limit.check_rate_limit("astra")


def test_rate_limit_window_slides_and_rejections_do_not_count(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    clock = {"now": 1000.0}
    monkeypatch.setattr(rate_limit.time, "time", lambda: clock["now"])
    rate_limit.check_rate_limit("dev")
    clock["now"] += 30
    rate_limit.check_rate_limit("dev")
    for _ in range(5):
        with pytest.raises(Blocked):
            rate_limit.check_rate_limit("dev")
    clock["now"] += 31  # the first request is now older than 60 s
    rate_limit.check_rate_limit("dev")


def test_token_budget_triggers_on_actual_usage(monkeypatch):
    monkeypatch.setattr(settings, "daily_token_budget", 1000)
    token_budget.check_budget("layer")
    token_budget.record_usage("layer", 600)
    token_budget.check_budget("layer")
    token_budget.record_usage("layer", 400)
    with pytest.raises(Blocked, match="budget of 1000 tokens"):
        token_budget.check_budget("layer")
    token_budget.check_budget("astra")
    assert token_budget.tokens_used("layer") == 1000
