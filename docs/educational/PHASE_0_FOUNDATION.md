# Educational Guide: Phase 0 Foundation

Welcome to the educational lab of **Libra**!

As part of our commitment to **Learning > Speed**, every major phase of Libra includes this detailed breakdown explaining the computer science and software engineering principles behind what we built.

---

## 1. WHAT Was Built?

In Phase 0, we established the core engineering foundation:
1. **Isolated Git Repository**: An independent version control space that tracks our code without touching any unrelated files on your computer.
2. **Python Virtual Environment (`.venv`)**: An isolated sandbox containing only the Python libraries needed for Libra.
3. **Hardware Introspection Engine (`packages/core/hardware.py`)**: A smart utility that detects your CPU, RAM, disk space, and graphics card.
4. **Model Provider Abstraction (`packages/providers/base.py`)**: A universal blueprint so our system can communicate with any AI model (our own custom model, Ollama, OpenAI, Gemini, etc.) using the exact same code.
5. **Backend API (`apps/backend`)**: A FastAPI web server that provides health and hardware diagnostics.
6. **Frontend UI (`apps/frontend`)**: A modern Next.js interface with real-time system status.

---

## 2. WHY Do We Need It?

### Why not just write one big Python script?
When software grows, putting everything into one file creates a "monolith" that becomes impossible to debug or modify. By separating the **frontend** (what you see), the **backend** (the server coordinating tasks), and the **packages** (the brain and models), we can improve or replace one piece without breaking the others.

### Why do we need a Virtual Environment (`.venv`)?
If you install Python libraries globally on your machine, different projects might require different versions of the same library (e.g. Project A needs Pydantic v1, while Project B needs Pydantic v2). A virtual environment acts as an isolated sandbox for Libra, ensuring your computer stays clean and dependencies never conflict.

### Why do we need Hardware Detection?
Modern AI models can be tens or hundreds of gigabytes in size. If a program attempts to load a 70-billion-parameter model into your 16GB of RAM, your computer will freeze or crash. By detecting your CPU and RAM at startup, Libra can automatically recommend safe, CPU-friendly models and prevent out-of-memory errors.

---

## 3. HOW Does It Work?

### The Client-Server Architecture
```
[User Browser]
      │
      │ 1. HTTP GET http://localhost:8000/api/v1/health
      ▼
[FastAPI Server]
      │
      │ 2. Queries hardware.py (psutil & OS checks)
      ▼
[Operating System]
      │
      │ 3. Returns CPU: i5-12450H, RAM: 16GB, GPU: Intel UHD
      ▼
[FastAPI Server]
      │
      │ 4. Formats response as JSON: {"status": "ok", "hardware": {...}}
      ▼
[Next.js Frontend]
      │
      │ 5. Renders green "System Ready: 8 CPU Cores, 16 GB RAM" status badge
```

1. **Hardware Detection**: We use Python's `psutil` (process and system utilities) library to query the operating system for physical/logical core counts and memory metrics. We also safely check for NVIDIA GPU drivers without crashing if they do not exist.
2. **Abstract Base Classes**: In `packages/providers/base.py`, we define a Python `ABC` (Abstract Base Class). It declares methods like `chat()` and `list_models()`. Any AI provider (whether a mock for tests or a real model) must follow this exact contract.

---

## 4. How To TEST It

We verify the system through automated tests:
```powershell
# Run the test suite
.\.venv\Scripts\pytest -v tests/
```
These tests verify that:
1. `test_hardware.py`: Correctly identifies your CPU, RAM, and storage space.
2. `test_providers.py`: Verifies that our model provider interface works as expected.
3. `test_health.py`: Sends a request to the FastAPI health endpoint and verifies an HTTP 200 "ok" response.

---

## 5. NEXT: What Comes Next?

With the foundation solid, we move directly to **Phase 1: Educational Tiny LLM**.

In Phase 1, we will build an actual autoregressive Transformer model in PyTorch from scratch:
- Token Embeddings
- Causal Self-Attention
- Multi-Head Attention
- Residual Connections & Layer Normalization
- Next-token prediction loss and text generation on your CPU!
