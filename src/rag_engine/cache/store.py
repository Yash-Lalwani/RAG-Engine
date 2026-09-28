"""Key-value cache: Upstash Redis when configured, otherwise an in-process dict."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from rag_engine.config import settings

logger = logging.getLogger(__name__)

_request_counts: ContextVar[dict[str, dict[str, int]] | None] = ContextVar(
    "request_cache_counts", default=None
)


@contextmanager
def track_cache() -> Iterator[dict[str, dict[str, int]]]:
    """Count cache hits and misses per tier for everything inside the `with` block."""
    counts: dict[str, dict[str, int]] = {}
    token = _request_counts.set(counts)
    try:
        yield counts
    finally:
        _request_counts.reset(token)


class CacheStore:
    def __init__(self) -> None:
        self._redis = self._build_redis_client()
        self._memory: dict[str, tuple[float, str]] = {}
        self._stats: dict[str, dict[str, int]] = defaultdict(lambda: {"hits": 0, "misses": 0})
        self._lock = threading.Lock()

    @staticmethod
    def _build_redis_client() -> Any | None:
        if not settings.upstash_redis_rest_url or not settings.upstash_redis_rest_token:
            logger.info("Upstash Redis not configured; using the in-memory cache")
            return None
        from upstash_redis import Redis

        return Redis(url=settings.upstash_redis_rest_url, token=settings.upstash_redis_rest_token)

    def get(self, tier: str, key: str) -> str | None:
        return self.get_many(tier, [key])[0]

    def set(self, tier: str, key: str, value: str, ttl_seconds: int) -> None:
        self.set_many(tier, {key: value}, ttl_seconds)

    def get_many(self, tier: str, keys: list[str]) -> list[str | None]:
        if not keys:
            return []
        values = self._redis_get_many(keys) if self._redis else self._memory_get_many(keys)
        hits = sum(value is not None for value in values)
        with self._lock:
            self._stats[tier]["hits"] += hits
            self._stats[tier]["misses"] += len(keys) - hits
        request_counts = _request_counts.get()
        if request_counts is not None:
            tier_counts = request_counts.setdefault(tier, {"hits": 0, "misses": 0})
            tier_counts["hits"] += hits
            tier_counts["misses"] += len(keys) - hits
        return values

    def set_many(self, tier: str, items: dict[str, str], ttl_seconds: int) -> None:
        if not items:
            return
        if self._redis:
            try:
                pipeline = self._redis.pipeline()
                for key, value in items.items():
                    pipeline.set(key, value, ex=ttl_seconds)
                pipeline.exec()
                return
            except Exception:
                logger.exception("Redis write failed for tier %s; keeping values in memory", tier)
        expires_at = time.time() + ttl_seconds
        with self._lock:
            for key, value in items.items():
                self._memory[key] = (expires_at, value)

    def stats(self) -> dict[str, dict[str, int]]:
        with self._lock:
            return {tier: dict(counts) for tier, counts in self._stats.items()}

    def _redis_get_many(self, keys: list[str]) -> list[str | None]:
        try:
            return [None if value is None else str(value) for value in self._redis.mget(*keys)]
        except Exception:
            logger.exception("Redis read failed; falling back to the in-memory cache")
            return self._memory_get_many(keys)

    def _memory_get_many(self, keys: list[str]) -> list[str | None]:
        now = time.time()
        values: list[str | None] = []
        with self._lock:
            for key in keys:
                entry = self._memory.get(key)
                if entry is None or entry[0] < now:
                    self._memory.pop(key, None)
                    values.append(None)
                else:
                    values.append(entry[1])
        return values


cache = CacheStore()
