# Architecture of Project Libra

## 1. System Overview

Libra is engineered with strict decoupling between the client UI, the serving API, the model providers, and the training laboratory:

```
+-------------------------------------------------------------+
|                      LIBRA APPLICATION                      |
|                  (Next.js App Router / TS)                  |
+-------------------------------------------------------------+
                               | (HTTP / SSE)
                               v
+-------------------------------------------------------------+
|                      LIBRA FASTAPI API                      |
|            (apps/backend: Auth, Router, Schemas)            |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                  MODEL ROUTER & REGISTRY                    |
|                    (packages/providers)                     |
+-------------------------------------------------------------+
         |                     |                     |
         v                     v                     v
+-----------------+   +-----------------+   +-----------------+
|  LIBRA LLM LAB  |   |  LOCAL ENGINES  |   |  EXTERNAL APIS  |
| (Custom Models) |   | (Ollama / HF)   |   | (Optional Keys) |
+-----------------+   +-----------------+   +-----------------+
```

## 2. Core Packages & Boundaries

1. **`packages/core`**:
   - Hardware detection and resource safeguards.
   - Operating system metrics and diagnostic reporting.
   - Core configuration abstractions.

2. **`packages/providers`**:
   - The unified `Provider` abstract base class.
   - Uniform `chat()`, `stream()`, `list_models()`, `health()` interfaces.
   - Decoupled adapters for Ollama, vLLM, OpenAI-compatible servers, and mock testing.

3. **`packages/models`**:
   - The educational LLM implementation (PyTorch).
   - Self-contained layers: embeddings, multi-head attention, MLP, RMSNorm, RoPE.

4. **`packages/training` & `packages/evaluation`**:
   - Training engine, optimization schedules, checkpoint serialization.
   - Perplexity calculations, loss recording, and validation harnesses.

## 3. Communication Protocols

- **Synchronous Commands**: Standard REST JSON via HTTP (`/api/v1/health`, `/api/v1/models`).
- **Text Generation**: Server-Sent Events (SSE) streaming for real-time token rendering.
- **Diagnostics**: Health and hardware reporting via structured Pydantic schemas.
