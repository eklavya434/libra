"""
Shared fixtures for API endpoint tests.

Gives every test a fresh, full token bucket for the process-wide rate limiter.
The middleware enforces rate limiting per client IP over 15 compute prefixes;
when the whole suite shares one process and one TestClient identity, a drained
bucket would make later tests in any module flake with 429s. Resetting per test
keeps the suite order-independent; production behavior is untouched.
"""

import pytest

from apps.backend.middleware.security import reset_global_rate_limiter


@pytest.fixture(autouse=True)
def fresh_rate_limit_bucket():
    reset_global_rate_limiter()
    yield
    reset_global_rate_limiter()
