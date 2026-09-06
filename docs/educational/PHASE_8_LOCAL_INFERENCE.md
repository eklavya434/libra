# Educational Guide: Phase 8 — Local Inference & The Ollama Engine

Welcome to **Phase 8** of **Libra**!

In this phase, we constructed the **Local Inference Engine** and **Provider Router**, enabling real-time token streaming from local models on your CPU.

---

## 1. WHAT Was Built?

1. **Ollama Provider** (`packages/providers/ollama.py`):
   - Asynchronous HTTP/SSE client connecting to the local Ollama daemon (`http://127.0.0.1:11434`).
   - Supports non-streaming completions, real-time token streaming (`stream: true`), embeddings, and full sampling controls (`temperature`, `top_k`, `top_p`, `stop`, `num_predict`).
   - Graceful offline detection with clear guidance if the daemon is not running.
2. **Local Transformer Provider** (`packages/providers/local_transformer.py`):
   - Serves our own PyTorch models (`ModernTransformerLM` / `TinyTransformerLM`) directly through the provider interface.
   - Bridges Goal 1 (AI Assistant) and Goal 2 (Educational LLM Lab) by letting you chat directly with your own checkpoints!
3. **vLLM Specification & Architectural Stub** (`packages/providers/vllm_stub.py`):
   - Implements Directive 6: documents vLLM as future GPU-only, preventing broken compilation attempts on a Windows CPU laptop.
4. **Prompt Template Formatter** (`packages/providers/prompt_template.py`):
   - Compiles conversation arrays into delimited strings formatted for **ChatML** (`<|im_start|>...`), **Llama 3** (`<|start_header_id|>...`), or **Plain Text**.
5. **FastAPI Chat Endpoint** (`apps/backend/api/v1/endpoints/chat.py`):
   - Exposes `POST /api/v1/chat/completions` supporting both standard JSON responses and Server-Sent Events (SSE) token streaming.

---

## 2. WHY Is Ollama Preferred for CPU (and Why Not vLLM)?

### A. The Secret Behind Ollama: `llama.cpp` and GGUF
Ollama is not written in Python. Under the hood, Ollama is a Go wrapper around **`llama.cpp`**, a hyper-optimized C/C++ inference library created by Georgi Gerganov:
- **CPU SIMD Acceleration**: It uses your Intel Core i5 processor's native vector instructions (**AVX-512** and **AVX2**). Instead of computing one float at a time, your CPU computes 8 or 16 numbers simultaneously in a single clock cycle!
- **GGUF Format**: Models are stored in GGUF binary format, allowing the operating system to memory-map (`mmap`) weights straight into RAM with zero conversion overhead.
- **Quantization**: Runs models in 4-bit (`Q4_K_M`), shrinking a 3-billion-parameter model from 6 GB down to 2 GB.

### B. Why vLLM Is GPU-Only
vLLM is famous in AI infrastructure for **PagedAttention**—a technique inspired by virtual memory in operating systems that eliminates memory fragmentation when serving hundreds of simultaneous users.
- **The Catch**: vLLM relies heavily on custom CUDA C++ kernels compiled specifically for NVIDIA GPUs (Ampere, Hopper, Blackwell).
- Its CPU backend is experimental, difficult to compile on Windows, and has massive memory overhead.
- In strict adherence to **Prime Directive 6**, we designate Ollama as our primary local CPU engine and document vLLM for future GPU infrastructure.

---

## 3. HOW Does Sampling and Prompt Templating Work?

### A. Why Do We Need Prompt Templates?
A neural network only understands a continuous sequence of tokens. When you have a chat dialogue:
```json
[
  {"role": "user", "content": "What is water?"}
]
```
The model has no inherent concept of "roles". The prompt template injects special control tokens:
```text
<|im_start|>user
What is water?<|im_end|>
<|im_start|>assistant
```
The model has been pre-trained to know that when it sees `<|im_start|>assistant`, it must generate the response and terminate by emitting `<|im_end|>`.

### B. The 4 Sampling Knobs
When generating tokens autoregressively from output logits:
1. **Temperature ($T$)**: Divides logits before softmax: $P(w_i) \propto \exp(z_i / T)$.
   - $T \to 0$: Greedy argmax (predictable, deterministic, factual).
   - $T = 0.7$: Balanced, creative, natural.
   - $T > 1.2$: High entropy, chaotic, prone to gibberish.
2. **Top-K**: Truncates the candidate pool to only the top $K$ most probable tokens (e.g., $K=40$). Prevents wild improbable words from being chosen.
3. **Top-P (Nucleus Sampling)**: Selects the smallest set of tokens whose cumulative probability exceeds $P$ (e.g., $P=0.90$). Dynamically adjusts the pool based on confidence!
4. **Stop Sequences**: String markers (e.g. `<|im_end|>`, `User:`) that immediately halt generation.

---

## 4. How We Verified It

1. **Automated Pytest Suite** (`tests/providers/` and `tests/api/test_chat_endpoint.py`):
   - Verified ChatML, Llama 3, and Plain prompt templates.
   - Tested offline daemon detection and error handling for Ollama.
   - Tested non-streaming and SSE streaming chat completions via FastAPI TestClient.
   - All 58 tests passed!
2. **CLI Demonstration** (`scripts/run_phase8_inference_demo.py`):
   - Executed live local inference using our PyTorch checkpoint at **349.5 tokens/sec** on CPU!

---

## 5. NEXT: Phase 9 & 10 — Multi-Provider Adapters & API Compatibility

In **Phase 9 & 10**, we will expand the provider abstraction to support:
- Cloud providers (OpenAI, Gemini, Claude, Groq, DeepSeek)
- Graceful mock and offline fallback when API keys are absent (respecting our $0 policy)
- Full OpenAI-compatible API routing.
