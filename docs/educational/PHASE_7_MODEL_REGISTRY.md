# Educational Guide: Phase 7 — Model Registry & Hardware Sizing

Welcome to **Phase 7** of **Libra**!

In this phase, we constructed the **Model Registry**: the catalog and hardware governance engine that bridges the gap between our internal LLM laboratory and production conversational assistants.

---

## 1. WHAT Was Built?

1. **Standardized Model Metadata Schema** (`packages/models/registry.py`):
   - Formal dataclass capturing parameter count, context length, architecture, quantization, exact licenses, and hardware requirements.
2. **Automated Hardware Classifier** (`packages/models/registry.py`):
   - Introspects the host machine (Intel Core i5-12450H CPU, 16GB RAM, CPU-only).
   - Computes estimated memory consumption in gigabytes.
   - Automatically categorizes every model into one of four **Hardware Tiers**:
     - `CPU-friendly`: Models $\le 3.5\text{B}$ parameters in 4-bit quantization, or models requiring $\le 8\text{ GB}$ of RAM.
     - `GPU-required`: Models requiring high VRAM bandwidth or dedicated CUDA execution.
     - `large`: Models between $10\text{B}$ and $50\text{B}$ parameters.
     - `very-large`: Frontier models $\ge 50\text{B}$ parameters (e.g. 70B, 405B, 671B).
3. **Curated Open-Weight Catalog** (`packages/models/catalog.py`):
   - Curated entries for 16 models across Meta (Llama 3.2), DeepSeek (R1 Distill), Alibaba (Qwen 2.5), Google (Gemma 2), Mistral AI (Mistral 7B), Microsoft (Phi-3.5), NVIDIA (Nemotron-Mini), and our own Libra Lab models.
4. **FastAPI Model Registry API** (`apps/backend/api/v1/endpoints/models.py`):
   - Live endpoints for querying models, filtering by `cpu_friendly_only=true`, and inspecting detailed hardware verdicts.

---

## 2. WHY Do We Need a Model Registry?

### A. Preventing Out-Of-Memory (OOM) Crashes
If an inexperienced user or autonomous agent tries to run a 70-billion-parameter model on a 16GB laptop, the operating system will attempt to load 46+ GB of data into memory, causing severe disk thrashing (paging) and freezing the computer.
The Model Registry acts as a **hardware guardrail**: before any model is downloaded or run, it verifies whether the model will fit within the user's available physical RAM.

### B. The Truth About "Open Source" vs. "Open Weights"
In AI marketing, people often say *"Llama is open source."* **This is legally inaccurate.**
In Project Libra, we enforce strict intellectual property terminology:

| Classification | Meaning | Examples | Commercial Use? |
| :--- | :--- | :--- | :--- |
| **True Open Source (OSI)** | Free code, weights, and training details with zero usage restrictions. | **Qwen 2.5** (Apache-2.0), **Mistral 7B** (Apache-2.0), **DeepSeek-R1 Distill** (MIT) | Full commercial freedom |
| **Open Weights (Community)** | Weights are downloadable, but subject to acceptable use policies or user limits. | **Llama 3.2** (Meta license; requires special approval if >700M monthly users), **Gemma 2** (Google Terms of Use) | Conditional commercial use |
| **Proprietary / API-Only** | Weights are secret; accessible only via paid cloud API. | **GPT-4o**, **Claude 3.5 Sonnet**, **Gemini 1.5 Pro** | Pay-per-token API only |

---

## 3. HOW Does the Sizing Math Work?

### A. The Memory Footprint Formula
$$\text{Memory Required (GB)} \approx \left(\frac{\text{Parameters} \times \text{Bytes per Parameter}}{1024^3}\right) \times 1.25$$

The $1.25$ multiplier provides a **25% safety overhead** to store:
1. **The KV Cache**: Attention keys and values for all past tokens in the conversation.
2. **Context Activation Buffers**: Tensors computed at intermediate layers during forward passes.
3. **Runtime Engine Overhead**: Allocations for Ollama / PyTorch execution.

### B. The Power of Quantization (4-Bit vs. 16-Bit)
When models are trained in FP16 (16-bit floating point), every single parameter consumes **2 bytes (16 bits)**.
When quantized to **Q4_K_M (4-bit)**:
- Every parameter is compressed to **~0.55 bytes (~4.5 bits)**.
- Memory consumption drops by **~73%**!

**Comparison on Your 16GB RAM Laptop**:
- **Llama 3.2 1B (Q4)**: $1.23\text{B} \times 0.55 \approx 676\text{ MB} \times 1.25 \approx \mathbf{0.8\text{ GB RAM}}$ $\to$ **Lightning fast on CPU!**
- **DeepSeek-R1 1.5B (Q4)**: $1.78\text{B} \times 0.55 \approx 980\text{ MB} \times 1.25 \approx \mathbf{1.1\text{ GB RAM}}$ $\to$ **Smooth CPU reasoning!**
- **Llama 3.2 3B (Q4)**: $3.21\text{B} \times 0.55 \approx 1.76\text{ GB} \times 1.25 \approx \mathbf{2.1\text{ GB RAM}}$ $\to$ **Runs smoothly on CPU!**
- **Qwen 2.5 72B (Q4)**: $72.7\text{B} \times 0.55 \approx 40.0\text{ GB} \times 1.25 \approx \mathbf{46.5\text{ GB RAM}}$ $\to$ **Cannot run on 16GB RAM!**

---

## 4. How We Verified It

1. **Automated Pytest Suite** (`tests/models/test_model_registry.py`):
   - Verified 16 models in catalog with correct metadata.
   - Tested hardware tier classification math for 1.5B (CPU-friendly) vs. 72B (very-large).
   - Tested FastAPI `/api/v1/models` and `/api/v1/models?cpu_friendly_only=true` endpoints.
2. **CLI Inspection** (`scripts/run_phase7_registry_demo.py`):
   - Evaluated all models in 0.05s, dynamically recommending 14 CPU-friendly models for your Intel i5 processor.

---

## 5. NEXT: Phase 8 — Local Inference (Ollama)

In **Phase 8**, we will integrate **Local Inference with Ollama**:
- Interfacing with the local Ollama runtime.
- Streaming tokens in real-time on CPU with our CPU-friendly models (`llama3.2:1b`, `deepseek-r1:1.5b`, `qwen2.5:1.5b`).
- Documenting why **Ollama** is ideal for CPU whereas **vLLM** is deferred to future dedicated GPU hardware.
