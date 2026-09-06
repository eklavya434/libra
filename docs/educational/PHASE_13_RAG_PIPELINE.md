# Educational Guide: Phase 13 - Retrieval-Augmented Generation (RAG) & Vector Retrieval

Welcome to **Phase 13** of **Libra**!

In this phase, we built a complete, first-principles **Retrieval-Augmented Generation (RAG)** pipeline in `packages/rag/`, integrated it with our FastAPI backend (`apps/backend/api/v1/endpoints/rag.py`), and created a dedicated **Knowledge Base** frontend interface (`apps/frontend/src/components/KnowledgeBaseView.tsx`).

---

## 1. WHAT Was Built?

1. **Recursive Character Chunker (`packages/rag/chunking.py`)**:
   - Hierarchically splits long texts using structural separators (`\n\n` paragraphs $\to$ `\n` lines $\to$ `. ` sentences $\to$ ` ` words).
   - Enforces configurable `chunk_size` (e.g. 500 characters) and `chunk_overlap` (e.g. 80 characters) to ensure factual context is never fractured at arbitrary byte cutoffs.
   - Accurately tracks original source character spans (`char_start`, `char_end`) and token footprints.

2. **First-Principles Dense Embedder (`packages/rag/embeddings.py`)**:
   - `EducationalDenseEmbedder`: Transforms text into dense unit-norm normalized vectors ($D=128$) using sliding word/character n-grams and deterministic hash projections. Runs in microseconds on CPU with zero network calls and zero model weight downloads.
   - `OllamaEmbeddingProvider`: Adapter connecting to local Ollama embedding runtimes (`nomic-embed-text`, `all-minilm`) with automatic offline fallback.

3. **In-Memory Vector Store (`packages/rag/vector_store.py`)**:
   - Vector database engine built directly on NumPy matrix operations.
   - Exact cosine similarity search: computes dot products $S = E \cdot q$ in a single matrix-vector operation, exploiting the fact that unit L2-normalized vectors satisfy $\cos(u, v) = u \cdot v$.
   - Supports incremental document ingestion, chunk indexing, top-$k$ ranking, and cascading document purging.

4. **Prompt Synthesizer (`packages/rag/synthesizer.py`)**:
   - Injects retrieved context passages with document titles and relevance scores into a grounding prompt template.
   - Enforces strict anti-hallucination system instructions: *"Answer using ONLY the provided reference context."*

5. **FastAPI RAG Endpoints (`apps/backend/api/v1/endpoints/rag.py` & `chat.py`)**:
   - `POST /api/v1/rag/documents`: Ingest and vectorize new documents.
   - `GET /api/v1/rag/documents`: List indexed documents and chunk statistics.
   - `DELETE /api/v1/rag/documents/{doc_id}`: Purge document and remove its embeddings from matrix memory.
   - `POST /api/v1/rag/query`: Direct semantic search with top-$k$ rankings and augmented prompt generation.
   - `POST /api/v1/chat/completions`: Added `use_rag: bool` flag to automatically ground multi-turn chats with knowledge base context.

6. **Knowledge Base UI & Chat Grounding Toggle (`apps/frontend/`)**:
   - New **Knowledge Base (RAG)** workspace tab in `Sidebar.tsx`.
   - `KnowledgeBaseView.tsx`: Form to index new technical notes, interactive semantic search tester with real-time cosine score match percentages, and document management cards.
   - `ChatArea.tsx`: Added one-click **RAG Grounding** toggle button in the chat header.

---

## 2. WHY Do We Need RAG?

While pre-trained Large Language Models possess broad general reasoning capabilities, they suffer from three fundamental limitations:

1. **Knowledge Cutoff & Static Weights**:
   The model only knows what was present in its training dataset. Updating parametric memory requires retraining or fine-tuning, which is computationally expensive.
2. **Hallucination on Specific Facts**:
   When asked about domain-specific, private, or granular facts (such as API keys, specialized company guidelines, or recent papers), models generate plausible-sounding falsehoods.
3. **Context Window Limits**:
   One cannot simply paste an entire 300-page textbook into the prompt context window ($L_{\max}$). Chunking and vector retrieval surface only the 2–4 most relevant passages directly related to the user's question.

---

## 3. HOW The Math Operates

### A. Dense Feature Projection:
For an input string $T$, we extract word and character n-grams:
$$F(T) = \{\text{words}\} \cup \{\text{3-grams}\} \cup \{\text{4-grams}\}$$

Each feature $f_i$ is mapped deterministically to a dimension index $j = \text{hash}(f_i) \pmod D$ with a sign $s_i \in \{-1, +1\}$. The vector is normalized onto the unit hypersphere $\mathbb{S}^{D-1}$:
$$\vec{v} = \frac{\sum_{i} s_i \vec{e}_{j}}{\left\| \sum_i s_i \vec{e}_j \right\|_2}$$

### B. Cosine Similarity via Dot Product:
Cosine similarity measures the angle $\theta$ between two vectors:
$$\text{sim}(\vec{u}, \vec{v}) = \cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$

Because our embedder pre-normalizes all document vectors $\|\vec{d}_i\|_2 = 1$ and query vector $\|\vec{q}\|_2 = 1$, the denominator is identically $1$:
$$\text{sim}(\vec{d}_i, \vec{q}) = \vec{d}_i \cdot \vec{q}$$

For an entire document collection represented as matrix $E \in \mathbb{R}^{N \times D}$, similarity across all $N$ chunks is computed in a single vectorized NumPy operation:
$$\vec{s} = E \vec{q} \in \mathbb{R}^N$$

The top-$k$ indices are extracted via $\text{argsort}(\vec{s})$ in descending order.

---

## 4. Verification & Test Coverage

1. **RAG Unit & API Tests**:
   ```bash
   .\.venv\Scripts\pytest -v tests/rag/ tests/api/test_rag_endpoint.py
   ```
   *Result*: **17 passed** in 8.21s.

2. **Full Repository Pytest Suite**:
   ```bash
   .\.venv\Scripts\pytest -v tests/
   ```
   *Result*: **108 passed, 0 failed** in 18.44s.

3. **Frontend Production Build**:
   ```bash
   cd apps/frontend
   npm run build
   ```
   *Result*: Successfully compiled and statically optimized with 0 TypeScript errors.

4. **Zero-Cost & Quota Compliance**:
   - Zero external vector database servers or cloud embedding fees.
   - Complete local CPU execution with zero storage regression.
