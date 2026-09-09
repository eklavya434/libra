"""
Tests for TokenBucketRateLimiter (packages/core/security/rate_limiter.py)
"""

import time

from packages.core.security.rate_limiter import (
    TokenBucketRateLimiter,
)


def test_rate_limiter_permits_within_capacity():
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_capacity=5)
    client_id = "client-test-1"

    # Consume up to burst capacity
    for _ in range(5):
        info = limiter.check(client_id, cost=1)
        assert info.allowed is True
        assert info.retry_after_seconds == 0.0

    # 6th immediate request should be throttled
    throttled_info = limiter.check(client_id, cost=1)
    assert throttled_info.allowed is False
    assert throttled_info.retry_after_seconds > 0.0
    assert throttled_info.remaining_tokens == 0


def test_rate_limiter_replenishment():
    # 60 req/min = 1 req/sec; capacity = 2
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_capacity=2)
    client_id = "client-test-2"

    # Drain bucket
    limiter.check(client_id, cost=2)
    throttled = limiter.check(client_id, cost=1)
    assert throttled.allowed is False

    # Sleep slightly over 1 second to replenish ~1 token
    time.sleep(1.05)

    replenished = limiter.check(client_id, cost=1)
    assert replenished.allowed is True


def test_rate_limiter_client_isolation():
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst_capacity=2)
    # Drain client A
    limiter.check("client-A", cost=2)
    assert limiter.check("client-A", cost=1).allowed is False

    # Client B should still have full capacity
    assert limiter.check("client-B", cost=1).allowed is True
    assert limiter.check("client-B", cost=1).allowed is True
    assert limiter.check("client-B", cost=1).allowed is False
