"""
Libra Core Security - In-Memory Token Bucket Rate Limiter

Implements high-performance, zero-cost token bucket rate limiting per client IP/session
with automatic bucket replenishment and burst smoothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class RateLimitInfo:
    """Status metadata returned on every rate limit evaluation."""

    allowed: bool
    remaining_tokens: int
    limit: int
    retry_after_seconds: float
    reset_epoch_seconds: float


class TokenBucket:
    """Represents a single client's token bucket."""

    def __init__(self, capacity: int, replenish_rate: float, now: float | None = None) -> None:
        self.capacity = capacity
        self.replenish_rate = replenish_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_update = now if now is not None else time.time()

    def replenish(self, now: float) -> None:
        elapsed = max(0.0, now - self.last_update)
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.replenish_rate)
        self.last_update = now

    def try_consume(self, cost: int, now: float) -> tuple[bool, float]:
        self.replenish(now)
        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0
        # Calculate time needed to replenish required tokens
        deficit = cost - self.tokens
        retry_after = deficit / self.replenish_rate if self.replenish_rate > 0 else 60.0
        return False, round(retry_after, 2)


class TokenBucketRateLimiter:
    """Thread-safe rate limiter managing token buckets per client key."""

    def __init__(
        self,
        requests_per_minute: int = 120,
        burst_capacity: int = 30,
        cleanup_interval_seconds: float = 300.0,
    ) -> None:
        self.limit = requests_per_minute
        self.capacity = burst_capacity
        self.replenish_rate = requests_per_minute / 60.0
        self.cleanup_interval = cleanup_interval_seconds
        self._buckets: dict[str, TokenBucket] = {}
        self._last_cleanup = time.time()
        self._lock = Lock()

    def check(self, client_id: str, cost: int = 1) -> RateLimitInfo:
        """Evaluates whether a request from client_id is permitted."""
        now = time.time()
        with self._lock:
            # Periodically evict idle buckets
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_idle_buckets(now)

            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(self.capacity, self.replenish_rate, now=now)

            bucket = self._buckets[client_id]
            allowed, retry_after = bucket.try_consume(cost, now)
            remaining = int(max(0.0, bucket.tokens))
            reset_epoch = now + (self.capacity - bucket.tokens) / self.replenish_rate

            return RateLimitInfo(
                allowed=allowed,
                remaining_tokens=remaining,
                limit=self.limit,
                retry_after_seconds=retry_after,
                reset_epoch_seconds=round(reset_epoch, 2),
            )

    def _cleanup_idle_buckets(self, now: float) -> None:
        """Evicts client buckets that have remained full and idle for > 10 minutes."""
        idle_threshold = 600.0
        to_delete = [
            cid
            for cid, b in self._buckets.items()
            if (now - b.last_update) > idle_threshold and b.tokens >= b.capacity
        ]
        for cid in to_delete:
            del self._buckets[cid]
        self._last_cleanup = now

    def get_active_client_count(self) -> int:
        with self._lock:
            return len(self._buckets)

    def reset(self) -> None:
        """Clears all tracking state."""
        with self._lock:
            self._buckets.clear()
