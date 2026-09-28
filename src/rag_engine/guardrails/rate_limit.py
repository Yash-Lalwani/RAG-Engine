"""G3: sliding-window rate limit per caller (Upstash sorted set, or memory when not configured)."""

import logging
import threading
import time
import uuid
from collections import defaultdict, deque

from rag_engine.cache.store import cache
from rag_engine.config import settings
from rag_engine.models import Blocked

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 60

_memory: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def check_rate_limit(caller: str) -> None:
    """Count this request; raise Blocked if the caller already made the maximum in the last minute.
    Rejected requests are not counted, so a blocked caller is free again once old requests expire."""
    limit = settings.rate_limit_per_minute
    allowed = _allow_redis(caller, limit) if cache.redis else _allow_memory(caller, limit)
    if not allowed:
        raise Blocked(f"Rate limit reached: at most {limit} requests per minute")


def _allow_memory(caller: str, limit: int) -> bool:
    now = time.time()
    with _lock:
        requests = _memory[caller]
        while requests and requests[0] <= now - WINDOW_SECONDS:
            requests.popleft()
        if len(requests) >= limit:
            return False
        requests.append(now)
        return True


def _allow_redis(caller: str, limit: int) -> bool:
    key, now = f"ratelimit:{caller}", time.time()
    member = f"{now}:{uuid.uuid4().hex}"
    try:
        pipeline = cache.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
        pipeline.zadd(key, {member: now})
        pipeline.zcard(key)
        pipeline.expire(key, WINDOW_SECONDS)
        count = pipeline.exec()[2]
        if count > limit:
            cache.redis.zrem(key, member)
            return False
        return True
    except Exception:
        logger.exception("Redis rate limit failed; using the in-memory window")
        return _allow_memory(caller, limit)
