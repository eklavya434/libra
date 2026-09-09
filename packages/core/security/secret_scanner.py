"""
Libra Core Security - Automated Secret & Sensitive Credential Scanner

Features:
1. Regex pattern matching for known cloud & AI API keys (Gemini, OpenAI, Anthropic, AWS, GitHub).
2. Shannon entropy calculation for detecting raw high-entropy tokens and hex strings.
3. In-place redaction pipeline to prevent secret leakage in logs, traces, and chat responses.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum


class SecretType(str, Enum):
    GEMINI_API_KEY = "gemini_api_key"
    OPENAI_API_KEY = "openai_api_key"
    ANTHROPIC_API_KEY = "anthropic_api_key"
    AWS_ACCESS_KEY = "aws_access_key"
    GITHUB_TOKEN = "github_token"
    JWT_TOKEN = "jwt_token"
    PRIVATE_KEY = "private_key"
    GENERIC_HIGH_ENTROPY = "generic_high_entropy"


@dataclass
class DetectedSecret:
    """Represents a discovered credential match."""

    secret_type: SecretType
    matched_value: str
    start_index: int
    end_index: int
    entropy: float
    redacted_replacement: str


class SecretScanner:
    """Regex and entropy-based credential detector and redactor."""

    PATTERNS = [
        (SecretType.GEMINI_API_KEY, r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED_GEMINI_API_KEY]"),
        (SecretType.GEMINI_API_KEY, r"AQ\.[A-Za-z0-9_-]{40,}", "[REDACTED_GEMINI_API_KEY]"),
        (SecretType.OPENAI_API_KEY, r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED_OPENAI_API_KEY]"),
        (
            SecretType.ANTHROPIC_API_KEY,
            r"sk-ant-[A-Za-z0-9_-]{20,}",
            "[REDACTED_ANTHROPIC_API_KEY]",
        ),
        (SecretType.AWS_ACCESS_KEY, r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_KEY]"),
        (SecretType.GITHUB_TOKEN, r"ghp_[A-Za-z0-9]{36}", "[REDACTED_GITHUB_TOKEN]"),
        (SecretType.GITHUB_TOKEN, r"github_pat_[A-Za-z0-9_]{22,}", "[REDACTED_GITHUB_PAT]"),
        (SecretType.PRIVATE_KEY, r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "[REDACTED_PRIVATE_KEY]"),
        (
            SecretType.JWT_TOKEN,
            r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_.-]{10,}",
            "[REDACTED_JWT_TOKEN]",
        ),
    ]

    def __init__(self, entropy_threshold: float = 3.8, min_entropy_len: int = 24) -> None:
        self.entropy_threshold = entropy_threshold
        self.min_entropy_len = min_entropy_len
        self._compiled = [(st, re.compile(pat), red) for st, pat, red in self.PATTERNS]

    @staticmethod
    def calculate_shannon_entropy(data: str) -> float:
        """Calculates Shannon entropy of a string in bits per character."""
        if not data:
            return 0.0
        entropy = 0.0
        length = len(data)
        freq: dict[str, int] = {}
        for c in data:
            freq[c] = freq.get(c, 0) + 1
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return round(entropy, 4)

    def scan(self, text: str) -> list[DetectedSecret]:
        """Scans arbitrary text for known secret patterns and high-entropy tokens."""
        if not text:
            return []

        findings: list[DetectedSecret] = []
        covered_spans: set[tuple[int, int]] = set()

        # 1. Pattern-based detection
        for secret_type, regex, red_text in self._compiled:
            for match in regex.finditer(text):
                span = match.span()
                covered_spans.add(span)
                val = match.group(0)
                entropy = self.calculate_shannon_entropy(val)
                findings.append(
                    DetectedSecret(
                        secret_type=secret_type,
                        matched_value=val,
                        start_index=span[0],
                        end_index=span[1],
                        entropy=entropy,
                        redacted_replacement=red_text,
                    )
                )

        # 2. Entropy-based detection for unstructured tokens (e.g. hex or base64 keys)
        words = re.findall(r"\b[A-Za-z0-9_-]{20,}\b", text)
        for word in words:
            # Check if this word is already covered by a regex pattern
            for match in re.finditer(re.escape(word), text):
                span = match.span()
                if any(s[0] <= span[0] and s[1] >= span[1] for s in covered_spans):
                    continue
                # Exclude purely repetitive words or natural language words
                entropy = self.calculate_shannon_entropy(word)
                if len(word) >= self.min_entropy_len and entropy >= self.entropy_threshold:
                    covered_spans.add(span)
                    findings.append(
                        DetectedSecret(
                            secret_type=SecretType.GENERIC_HIGH_ENTROPY,
                            matched_value=word,
                            start_index=span[0],
                            end_index=span[1],
                            entropy=entropy,
                            redacted_replacement="[REDACTED_HIGH_ENTROPY_SECRET]",
                        )
                    )

        # Sort findings by start_index
        findings.sort(key=lambda s: s.start_index)
        return findings

    def redact(self, text: str) -> str:
        """Sanitizes text by replacing all identified credentials with redacted tokens."""
        findings = self.scan(text)
        if not findings:
            return text

        # Sort reverse by start_index to preserve indices while slicing
        findings.sort(key=lambda s: s.start_index, reverse=True)
        sanitized = text
        for secret in findings:
            sanitized = (
                sanitized[: secret.start_index]
                + secret.redacted_replacement
                + sanitized[secret.end_index :]
            )
        return sanitized
