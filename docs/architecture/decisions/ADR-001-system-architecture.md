# ADR-001: Modular System Architecture & Stack Selection

## Context
Project Libra requires both an interactive, state-of-the-art AI assistant product and an educational laboratory for building language models from first principles. The system must run entirely on consumer hardware (Intel Core i5, 16GB RAM, integrated graphics) at zero financial cost, with beginner-friendly accessibility.

## Decision
1. **Frontend**: Next.js (React 18+, TypeScript, Tailwind CSS).
   - *Why*: Industry standard for modern conversational UIs with built-in server-side rendering, streaming support, and robust developer tooling.
2. **Backend**: FastAPI (Python 3.11+).
   - *Why*: High-performance asynchronous API framework natively integrated with Pydantic for data validation and OpenAPI auto-documentation. Essential for direct interoperability with PyTorch and Python data science ecosystems.
3. **Provider Abstraction**: A decoupled Provider interface.
   - *Why*: Prevents vendor lock-in. The frontend never talks directly to external APIs or proprietary SDKs; all calls flow through a unified backend provider layer.
4. **Educational Laboratory**: Decoupled in `packages/models` and `packages/training`.
   - *Why*: Educational transformer architectures can be developed, tested, and modified without risking the stability of the serving application.

## Consequences
- Clean separation of concerns allows parallel development of UI features and deep learning models.
- Minimal footprint allows both backend and frontend dev servers to run concurrently on 16GB RAM with zero performance bottlenecks.
