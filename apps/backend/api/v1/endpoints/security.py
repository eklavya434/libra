"""
Libra API v1 - Security Auditing & Threat Defense Endpoints

Provides REST APIs for:
1. POST /api/v1/security/scan: Prompt injection detection & credential scanning.
2. POST /api/v1/security/redact: Automated redaction of sensitive credentials.
3. GET /api/v1/security/stats: Real-time telemetry on rate limiting and security posture.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from apps.backend.middleware.security import get_global_rate_limiter
from packages.core.security.prompt_guard import PromptGuard
from packages.core.security.secret_scanner import SecretScanner

router = APIRouter(prefix="/security", tags=["Security & Defense"])

# Subsystem singletons
_prompt_guard = PromptGuard()
_secret_scanner = SecretScanner()

# Runtime security telemetry counters
_security_stats = {
    "scans_performed": 0,
    "injections_blocked": 0,
    "secrets_intercepted": 0,
}


class SecurityScanRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze for threats and credentials")


class SecurityRedactRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to redact")


class DetectedSecretItem(BaseModel):
    secret_type: str
    matched_value_masked: str
    start_index: int
    end_index: int
    entropy: float
    redacted_replacement: str


class PromptAuditItem(BaseModel):
    is_safe: bool
    risk_score: float
    matched_types: list[str]
    reasons: list[str]
    remediation: str
    sanitized_prompt: Optional[str]


class SecurityScanResponse(BaseModel):
    is_safe: bool
    overall_risk_score: float
    prompt_audit: PromptAuditItem
    secrets_detected: list[DetectedSecretItem]
    sanitized_text: str


@router.post(
    "/scan",
    response_model=SecurityScanResponse,
    summary="Scan text for prompt injections and secrets",
)
async def scan_text(request: SecurityScanRequest) -> Any:
    global _security_stats
    _security_stats["scans_performed"] += 1

    # 1. Evaluate prompt injection threats
    verdict = _prompt_guard.evaluate_prompt(request.text)
    if not verdict.is_safe:
        _security_stats["injections_blocked"] += 1

    # 2. Scan for credentials & secrets
    secrets = _secret_scanner.scan(request.text)
    if secrets:
        _security_stats["secrets_intercepted"] += len(secrets)

    # 3. Compute overall risk score and sanitized representation
    redacted_text = _secret_scanner.redact(verdict.sanitized_prompt or request.text)

    secret_risk = min(1.0, len(secrets) * 0.4)
    overall_risk = max(verdict.risk_score, secret_risk)
    overall_safe = verdict.is_safe and len(secrets) == 0

    detected_items = [
        DetectedSecretItem(
            secret_type=s.secret_type.value,
            matched_value_masked=s.matched_value[:4] + "..." + s.matched_value[-4:]
            if len(s.matched_value) > 8
            else "[HIDDEN]",
            start_index=s.start_index,
            end_index=s.end_index,
            entropy=s.entropy,
            redacted_replacement=s.redacted_replacement,
        )
        for s in secrets
    ]

    return SecurityScanResponse(
        is_safe=overall_safe,
        overall_risk_score=overall_risk,
        prompt_audit=PromptAuditItem(
            is_safe=verdict.is_safe,
            risk_score=verdict.risk_score,
            matched_types=[t.value for t in verdict.matched_types],
            reasons=verdict.reasons,
            remediation=verdict.remediation,
            sanitized_prompt=verdict.sanitized_prompt,
        ),
        secrets_detected=detected_items,
        sanitized_text=redacted_text,
    )


@router.post("/redact", summary="Redact sensitive credentials from text")
async def redact_text(request: SecurityRedactRequest) -> dict[str, Any]:
    secrets = _secret_scanner.scan(request.text)
    redacted = _secret_scanner.redact(request.text)
    return {
        "original_length": len(request.text),
        "redacted_length": len(redacted),
        "secrets_found": len(secrets),
        "redacted_text": redacted,
    }


@router.get("/stats", summary="Get runtime security telemetry")
async def get_security_stats() -> dict[str, Any]:
    limiter = get_global_rate_limiter()
    return {
        "status": "active",
        "counters": _security_stats,
        "rate_limiter": {
            "requests_per_minute_limit": limiter.limit,
            "burst_capacity": limiter.capacity,
            "active_clients": limiter.get_active_client_count(),
        },
        "capabilities": [
            "direct_prompt_injection_defense",
            "jailbreak_persona_defense",
            "canary_token_verification",
            "delimiter_spoofing_defense",
            "shannon_entropy_secret_scanner",
            "owasp_security_headers",
            "token_bucket_rate_limiter",
        ],
    }
