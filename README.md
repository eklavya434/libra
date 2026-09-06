# ♎ Libra: Personal LLM Laboratory + ChatGPT-Like AI Assistant

> **Libra** is an open-source, dual-track artificial intelligence project designed to build a fully capable, production-grade conversational AI assistant while simultaneously teaching how Large Language Models (LLMs) function from mathematical and code first principles.

---

## 🎯 The Two Interconnected Goals

```
                              PROJECT LIBRA
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
   TRACK 1: THE AI PRODUCT                             TRACK 2: THE LLM LAB
   • ChatGPT-like Assistant                            • Tokenizer from Scratch
   • Web UI & REST/Streaming API                       • Attention & Transformers
   • Multi-Provider Router                             • Causal Masking & RoPE
   • RAG, Memory, & Tool System                        • Backpropagation & AdamW
   • Autonomous Research Agents                        • Checkpointing & Generation
```

1. **Goal 1 — The AI Product**: A modern, modular, production-ready AI assistant featuring conversation management, streaming responses, model comparison arena, retrieval-augmented generation (RAG), sandboxed code execution, web search, and agentic workflows.
2. **Goal 2 — The LLM Laboratory**: A complete educational deep learning lab where we implement an autoregressive language model from scratch in PyTorch—starting with tiny CPU-friendly experiments and progressing through tokenization, multi-head attention, SFT, DPO, and quantization.

---

## 💻 Hardware Philosophy & Zero-Cost Policy

- **Target Hardware**: Designed specifically to run on everyday computers (tested on Intel Core i5-12450H CPU, 16GB RAM, integrated Intel UHD graphics).
- **Cost Policy**: **$0 / ₹0 development**. Paid services are never required. All core experiments run offline and free of charge using open weights, local models, or CPU-friendly educational implementations.

---

## 🏗️ Architecture Blueprint

```
USER
  │
  ▼
LIBRA WEB INTERFACE (Next.js / TypeScript / Tailwind CSS)
  │
  ▼
LIBRA CORE API (FastAPI)
  │
  ▼
MODEL ROUTER & PROVIDER ABSTRACTION
  │
  ├── Local Models (Ollama, Hugging Face Transformers)
  ├── Educational Libra Model (Trained from scratch in LLM Lab)
  └── Optional External APIs (OpenAI, Claude, Gemini, DeepSeek, Groq)
```

---

## 🚀 Quick Start (Phase 0)

### 1. Prerequisites
- Python 3.11+
- Node.js 18+

### 2. Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd Libra

# Initialize Python Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# Install Phase 0 Dependencies
pip install -e .[dev]
```

### 3. Run the Backend API
```bash
.\.venv\Scripts\uvicorn apps.backend.main:app --reload --port 8000
```
Visit `http://localhost:8000/api/v1/health` or `http://localhost:8000/docs` for interactive Swagger API documentation.

### 4. Run the Test Suite
```bash
.\.venv\Scripts\pytest -v tests/
```

---

## 🗺️ Phased Roadmap

- **Phase 0 (Current)**: Foundation, Project Skeleton, FastAPI, Next.js, Hardware Detection, Testing.
- **Phase 1**: Educational Tiny LLM (CPU decoder-only transformer).
- **Phase 2**: Educational and Production Tokenizer (BPE).
- **Phase 3**: Data Pipeline & Pre-training Corpora.
- **Phase 4**: Modern Transformer (RoPE, RMSNorm, SwiGLU).
- **Phase 5**: Training Engine & Learning-Rate Scheduler.
- **Phase 6**: Evaluation & Perplexity Benchmarking.
- **Phase 7–11**: Model Registry, Local Inference (Ollama/vLLM), Provider Adapters, Full Chat UI.
- **Phase 12–20**: Memory, RAG, Web Search, Deep Research, Sandboxed Code Execution, Agents.

---

## 📄 License

Apache 2.0 License — Free for educational, research, and personal use.
