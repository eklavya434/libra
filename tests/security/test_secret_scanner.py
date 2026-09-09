"""
Tests for SecretScanner (packages/core/security/secret_scanner.py)
"""

from packages.core.security.secret_scanner import (
    SecretScanner,
    SecretType,
)

# Synthetic tokens assembled dynamically to prevent false-positive alerts in static git scanners
DUMMY_GEMINI_KEY = "AIzaSy" + "FakeTestDummyKey12345678901234567"
DUMMY_OPENAI_KEY = "sk-" + "projFakeTestDummyKey123456789012345"
DUMMY_AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"


def test_gemini_api_key_detection():
    scanner = SecretScanner()
    text = f"Loaded key: {DUMMY_GEMINI_KEY} for model routing."
    findings = scanner.scan(text)
    assert len(findings) == 1
    assert findings[0].secret_type == SecretType.GEMINI_API_KEY
    assert findings[0].matched_value == DUMMY_GEMINI_KEY

    redacted = scanner.redact(text)
    assert "[REDACTED_GEMINI_API_KEY]" in redacted
    assert "AIzaSy" not in redacted


def test_openai_api_key_detection():
    scanner = SecretScanner()
    text = f"Client initialized with {DUMMY_OPENAI_KEY} key."
    findings = scanner.scan(text)
    assert len(findings) == 1
    assert findings[0].secret_type == SecretType.OPENAI_API_KEY

    redacted = scanner.redact(text)
    assert "[REDACTED_OPENAI_API_KEY]" in redacted
    assert "sk-" not in redacted


def test_aws_key_detection():
    scanner = SecretScanner()
    text = f"Credentials: {DUMMY_AWS_KEY} stored locally."
    findings = scanner.scan(text)
    assert len(findings) == 1
    assert findings[0].secret_type == SecretType.AWS_ACCESS_KEY
    redacted = scanner.redact(text)
    assert "[REDACTED_AWS_KEY]" in redacted


def test_clean_natural_language_no_false_positives():
    scanner = SecretScanner()
    text = "The quick brown fox jumps over the lazy dog and explores transformer architectures."
    findings = scanner.scan(text)
    assert len(findings) == 0
    assert scanner.redact(text) == text


def test_shannon_entropy_calculation():
    # Repetitive low entropy
    low = SecretScanner.calculate_shannon_entropy("aaaaaaaaaaaaaaaaaaaaaaaa")
    assert low == 0.0

    # High entropy random hex
    high = SecretScanner.calculate_shannon_entropy("8f4a9b2c3d1e7f0a9b8c7d6e")
    assert high >= 3.5
