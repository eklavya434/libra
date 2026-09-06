# AGENTS.md — Developer & AI Agent Guidelines for Project Libra

Welcome to **Libra**. This repository is an educational LLM laboratory and ChatGPT-like AI assistant built from first principles.

This file establishes strict operational protocols for autonomous AI agents (primarily **Antigravity**, accompanied by **Codex**) and human collaborators.

---

## 1. Prime Directives

1. **Learning > Speed**: The user is a beginner. Never hide architectural complexity behind black-box magic. Every component must be explainable.
2. **Zero-Cost ($0 / ₹0)**: Never automatically invoke paid APIs, launch cloud GPUs, or initiate billing. Always use local, mock, or free tier resources unless explicitly approved by the user.
3. **Hardware Awareness**: Target hardware is **CPU-only** (Intel Core i5-12450H, 16GB RAM, ~72GB free disk). Do not download multi-gigabyte models or datasets without explicit confirmation.
4. **No Fake Implementations**: Never fabricate training metrics, loss curves, benchmark results, or mock outputs masked as real. Mocks must be explicitly marked as `MockProvider` or `DummyData`.
5. **No Monoliths**: Maintain a strict separation between:
   - The user-facing AI chat application
   - The educational LLM lab (tokenizers, transformer models, training loops)
   - External provider adapters

---

## 2. Directory Ownership & Structure

| Directory | Purpose | Agent Guidance |
| :--- | :--- | :--- |
| `apps/backend/` | FastAPI service, routers, endpoints, config | API layer; keep lean and delegate business logic to packages |
| `apps/frontend/` | Next.js (TypeScript + Tailwind CSS) | UI layer; communicates strictly via backend REST/SSE APIs |
| `packages/core/` | Shared utilities (hardware detection, logging) | Hardware introspection, config helpers |
| `packages/models/` | Educational transformer architectures (PyTorch) | Built step-by-step from first principles |
| `packages/providers/` | Unified model provider adapters | Abstract interface for Local, Ollama, OpenAI, Gemini, etc. |
| `packages/training/` | Training loops, optimizers, checkpointing | CPU-friendly training engine |
| `packages/evaluation/`| Perplexity, loss evaluation, benchmarks | Quantitative evaluation pipelines |
| `packages/rag/` | Vector retrieval, document chunking | RAG pipeline (Phase 13+) |
| `packages/tools/` | Tool definitions and execution sandbox | Safe tool execution (Phase 16+) |
| `packages/agents/` | Autonomous agent loops | Multi-step agent orchestrator (Phase 18+) |
| `docs/` | Architecture decision records & educational guides | Must be kept up to date after every phase |
| `tests/` | Pytest & frontend test suites | Every feature requires tests before merging |

---

## 3. Development Workflow & Commands

Always use the isolated virtual environment:

- **Python Virtualenv**: `.\.venv\Scripts\python.exe`
- **Run Backend**: `.\.venv\Scripts\uvicorn apps.backend.main:app --reload --port 8000`
- **Run Tests**: `.\.venv\Scripts\pytest -v tests/`
- **Run Linter**: `.\.venv\Scripts\ruff check .`
- **Format Code**: `.\.venv\Scripts\ruff format .`

---

## 4. Coding Standards

- **Python**: Python 3.11+ syntax, type annotations on all function signatures, Pydantic for validation.
- **Frontend**: Next.js App Router, TypeScript strict mode, functional components, Tailwind CSS.
- **Error Handling**: Graceful degradation (e.g. if CUDA is not found, CPU fallback with informative warning).
- **Security**: Never commit `.env` or secrets. Never execute untrusted shell commands from public API endpoints.

---

## 5. Educational Protocol (LIBRA_LEARNING_MODE=true)

Whenever completing a phase or major module, document:
- **WHAT**: What was built.
- **WHY**: Why it exists and what problem it solves.
- **HOW**: How the underlying math/code operates.
- **TEST**: How it was verified.
- **NEXT**: What the next phase requires.
