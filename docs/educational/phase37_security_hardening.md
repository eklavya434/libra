# Phase 37: Security Hardening & Adversarial Robustness

## 1. WHAT Was Built
Phase 37 introduces a comprehensive, multi-layer defense-in-depth security perimeter for Project Libra.

Key modules implemented:
- **`PromptGuard` (`packages/core/security/prompt_guard.py`)**:
  - Direct prompt injection & system prompt override detection.
  - Adversarial jailbreak persona detection (DAN, Developer Mode, hypothetical evil twin bypasses).
  - Delimiter spoofing prevention (ChatML `<|im_start|>`, Llama `[INST]`, `<<SYS>>`).
  - GCG adversarial noise anomaly detection via non-alphanumeric token density and repetitive punctuation analysis.
  - Cryptographic canary token generator and leak detection.
  - Untrusted context framing for RAG pipelines to neutralize indirect prompt injection.
- **`SecretScanner` (`packages/core/security/secret_scanner.py`)**:
  - Pattern-based detection for Google Gemini, OpenAI, Anthropic, AWS, GitHub, JWT, and SSH private keys.
  - Shannon entropy calculator $H(s) = -\sum p_i \log_2 p_i$ for detecting unstructured high-entropy credentials.
  - In-place redaction engine preventing sensitive leaks in chat responses, logs, and tracebacks.
- **`TokenBucketRateLimiter` & Middleware (`packages/core/security/rate_limiter.py` & `apps/backend/middleware/security.py`)**:
  - Thread-safe token bucket rate limiting per client IP.
  - OWASP security response headers (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`).
  - Payload size limiter (rejecting payloads $> 10$ MB to prevent memory exhaustion).
- **Security REST APIs (`apps/backend/api/v1/endpoints/security.py`)**:
  - `POST /api/v1/security/scan`, `POST /api/v1/security/redact`, `GET /api/v1/security/stats`.
- **Frontend Security Lab (`apps/frontend/src/components/SecurityInspector.tsx`)**:
  - Interactive attack bench, secret redaction tester, and real-time telemetry monitor.

---

## 2. WHY It Exists
In traditional computing architectures (Von Neumann), instructions and data are separated at the hardware level: code executes in read-only memory pages while data is stored in non-executable memory (W^X / Data Execution Prevention).

In contrast, Large Language Models (LLMs) operate over a **unified token sequence**. All text—system instructions, conversational context, third-party retrieved documents, and untrusted user input—is encoded into identical dense token representations within the transformer's self-attention mechanism:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Because the model attends to all tokens simultaneously, an adversary can craft user inputs that override the system prompt ("Ignore previous instructions"), assume unconstrained personas ("DAN Mode"), or inject instructions inside retrieved RAG documents.

Phase 37 enforces defense-in-depth at multiple stages:
1. **Network Layer**: Token bucket rate limiting and payload limits prevent resource exhaustion.
2. **Ingress Layer**: Heuristic and entropy filters intercept injection payloads and delimiter spoofing before reaching the inference engine.
3. **Context Layer**: Retrieved RAG documents are encapsulated in `<untrusted_context>` tags and system prompts are equipped with canary verification tokens.
4. **Egress Layer**: Output text is scanned and redacted using Shannon entropy and regex signatures to guarantee zero credential leakage.

---

## 3. HOW It Works (The Math & Mechanics)

### A. Shannon Information Entropy for Secret Detection
Random cryptographic keys and hashes possess significantly higher information entropy than natural language words. The Shannon entropy $H(s)$ of a string of length $N$ with character distribution frequencies $p_i$ is computed as:
$$H(s) = -\sum_{i=1}^{K} p_i \log_2 p_i$$

For example:
- Natural English text ("welcome_to_libra"): $H \approx 2.7 - 3.2\text{ bits/char}$
- Random Base64/Hex API Key ("AIzaSy[REDACTED_33_CHAR_STRING]"): $H \ge 3.8 - 4.5\text{ bits/char}$

The `SecretScanner` flags tokens exceeding $H \ge 3.8$ bits/char with length $\ge 24$ characters as probable credentials.

### B. Token Bucket Rate Limiting
A client starts with a burst capacity of $C$ tokens. At any timestamp $t$, elapsed time $\Delta t = t - t_{\text{last}}$ adds tokens at replenishment rate $r$:
$$T_t = \min(C, T_{t_{\text{last}}} + \Delta t \cdot r)$$

If $T_t \ge 1$, the request is accepted and $T_t \leftarrow T_t - 1$. Otherwise, the request is rejected with HTTP 429 and `Retry-After`:
$$\text{Retry-After} = \frac{1 - T_t}{r}$$

### C. Canary Token Leak Verification
A cryptographically random UUID token $C_{\text{canary}}$ is injected into the system prompt:
$$\text{System} \leftarrow \text{System} \parallel \text{"[SECURITY PROTOCOL]: Canary is } C_{\text{canary}}\text{"}$$
If $C_{\text{canary}} \in \text{Completion}$, an alert is logged and the completion is quarantined.

---

## 4. TEST Verification
The security subsystem is validated via unit and API test suites:
- `tests/security/test_prompt_guard.py`: Direct injection, DAN jailbreak, canary leak detection, GCG adversarial noise, clean prompt pass-through.
- `tests/security/test_secret_scanner.py`: Google Gemini, OpenAI, AWS, and GitHub key detection and in-place redaction.
- `tests/security/test_rate_limiter.py`: Token bucket replenishment, burst tolerance, and client eviction.
- `tests/api/test_security_endpoint.py`: `/api/v1/security/scan`, `/redact`, `/stats` HTTP validation.

---

## 5. NEXT Steps
With Phase 37 complete, the repository possesses an industrial security perimeter. Phase 38 will focus on **Observability, Distributed Tracing & OpenTelemetry Logging** across the full inference lifecycle.
