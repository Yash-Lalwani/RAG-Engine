"""G4: daily token budget per caller, from the actual token usage of each request."""

import logging
import threading
from datetime import UTC, datetime

from rag_engine.cache.store import cache
from rag_engine.config import settings
from rag_engine.models import Blocked

logger = logging.getLogger(__name__)

KEY_TTL_SECONDS = 2 * 24 * 3600  # the key contains the date, so it only needs to outlive the day

_memory: dict[str, int] = {}
_lock = threading.Lock()


def check_budget(caller: str) -> None:
    """Before a call: raise Blocked once the caller has used today's budget."""
    budget = settings.daily_token_budget
    if tokens_used(caller) >= budget:
        raise Blocked(f"Daily token budget of {budget} tokens is used up; it resets at 00:00 UTC")


def record_usage(caller: str, tokens: int) -> None:
    """After a call: add the tokens it actually used (from llm.track_usage())."""
    if tokens <= 0:
        return
    key = _key(caller)
    if cache.redis:
        try:
            pipeline = cache.redis.pipeline()
            pipeline.incrby(key, tokens)
            pipeline.expire(key, KEY_TTL_SECONDS)
            pipeline.exec()
            return
        except Exception:
            logger.exception("Redis token budget failed; recording in memory")
    with _lock:
        _memory[key] = _memory.get(key, 0) + tokens


def tokens_used(caller: str) -> int:
    key = _key(caller)
    if cache.redis:
        try:
            return int(cache.redis.get(key) or 0)
        except Exception:
            logger.exception("Redis token budget failed; reading from memory")
    with _lock:
        return _memory.get(key, 0)


def _key(caller: str) -> str:
    return f"budget:{caller}:{datetime.now(UTC):%Y-%m-%d}"
