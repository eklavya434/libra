# Educational Guide: Phase 14 — Advanced Hybrid Search, Reciprocal Rank Fusion & Re-Ranking

Welcome to **Phase 14** of Project Libra. In this phase, we evolve our Retrieval-Augmented Generation (RAG) system from a single-vector dense retriever into a production-grade **Two-Stage Hybrid Search & Re-Ranking Pipeline**.

---

## 1. The Dual-Retriever Problem: Why Hybrid Search?

In real-world applications, single-retriever systems suffer from fundamental blind spots:

| Metric / Capability | Dense Vector Retrieval | Sparse Lexical Retrieval (BM25) | Hybrid RAG (Dense + BM25) |
| :--- | :--- | :--- | :--- |
| **Semantic Synonyms** | ⭐⭐⭐ Exquisite ("car" $\approx$ "automobile") | ❌ Fails (vocabulary mismatch) | ⭐⭐⭐ High recall via dense |
| **Exact Technical Terms** | ❌ Weak (`i5-12450H`, `SwiGLU`, `WAL`) | ⭐⭐⭐ Exquisite (exact token match) | ⭐⭐⭐ Exact precision via BM25 |
| **Out-of-Vocabulary (OOV)** | ⚠️ Struggles or hashes poorly | ⭐⭐⭐ Matches exact character tokens | ⭐⭐⭐ Robust coverage |
| **Code Identifiers & Numbers**| ⚠️ Vector smearing across embeddings | ⭐⭐⭐ High IDF for rare identifiers | ⭐⭐⭐ Preserves exact matches |

By deploying **both** in parallel, we achieve the best of both worlds:
1. **Dense embeddings** find conceptually relevant passages even when phrasing differs.
2. **BM25 sparse index** catches exact model names, version numbers, error codes, and technical jargon.

---

## 2. Mathematical First Principles

### A. Okapi BM25 Sparse Search

Okapi BM25 is a non-linear term-matching function that addresses two major flaws in classic TF-IDF:
1. **Term Frequency Saturation**: In standard TF, a word appearing 100 times in a document scores 100 times higher than a word appearing once. In reality, relevance plateaus quickly. BM25 introduces asymptotic saturation controlled by $k_1$.
2. **Document Length Normalization**: Long documents naturally contain more words, which inflates term counts. BM25 penalizes excessively long documents with parameter $b$.

#### The Okapi BM25 Formula
$$\text{BM25}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $f(q_i, D)$ is the frequency of query term $q_i$ in chunk $D$.
- $|D|$ is the length of chunk $D$ in tokens.
- $\text{avgdl}$ is the average token length across all chunks in the knowledge base.
- $k_1 \in [1.2, 2.0]$ (default $1.5$): controls how quickly term frequency saturates.
- $b \in [0.0, 1.0]$ (default $0.75$): controls the degree of document length penalization.

#### Robertson-Spärck Jones Smoothed IDF
$$\text{IDF}(q_i) = \ln\left(1 + \frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)$$
Where $N$ is the total number of chunks and $n(q_i)$ is the number of chunks containing term $q_i$. The $+0.5$ and outer $+1.0$ guarantee that $\text{IDF}(q_i)$ is strictly positive for all terms, avoiding negative scores for ubiquitous words.

---

### B. Reciprocal Rank Fusion (RRF)

#### Why Not Simply Add Scores?
Dense cosine similarity produces scores strictly bounded in $[-1.0, 1.0]$. BM25 produces unbounded positive scores dependent on corpus size and term rarity (e.g. $0.5$ to $25.0+$). Linearly summing raw scores is mathematically invalid because the sparse scale completely dominates the dense scale.

#### The RRF Solution
**Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)** ignores raw numerical magnitudes and combines candidates strictly by their **ordinal ranking**:

$$\text{RRF}(d) = \sum_{m \in M} \frac{1}{k_{\text{rrf}} + r_m(d)}$$

Where:
- $M = \{\text{Dense}, \text{BM25}\}$ is the set of retrieval systems.
- $r_m(d)$ is the 1-based rank of chunk $d$ in system $m$.
- $k_{\text{rrf}}$ (standard default $60$) is a smoothing constant preventing top-ranked items from receiving an overwhelming advantage over items ranked slightly lower.

**Key Mathematical Property**: If chunk $A$ ranks #1 in Dense and #1 in BM25, its RRF score is:
$$\frac{1}{60 + 1} + \frac{1}{60 + 1} = \frac{2}{61} \approx 0.03279$$
If chunk $B$ ranks #1 in Dense but is completely absent from BM25's top list:
$$\frac{1}{60 + 1} + 0 = \frac{1}{61} \approx 0.01639$$
Items discovered by **both** systems automatically vault to the top of the combined pool!

---

### C. Multi-Factor Re-Ranking

Stage 1 (Retrieval) prioritizes **high recall** across the entire corpus, generating a candidate pool of $K_{\text{pool}} = \max(4 \cdot \text{top\_k}, 15)$ chunks.
Stage 2 (Re-ranking) applies a higher-precision multi-factor scorer to evaluate query-document alignment:

$$\text{Score}_{\text{rerank}} = w_{\text{prior}} S_{\text{prior}} + w_{\text{phrase}} S_{\text{phrase}} + w_{\text{coverage}} S_{\text{coverage}} + w_{\text{proximity}} S_{\text{proximity}}$$

1. **Exact Phrase Match ($S_{\text{phrase}}$)**: Checks for contiguous query word sequences in the chunk text, granting full bonus if the query appears verbatim.
2. **Query Keyword Coverage ($S_{\text{coverage}}$)**: Calculates the ratio of unique query terms present: $\frac{|\text{Terms}(Q) \cap \text{Terms}(D)|}{|\text{Terms}(Q)|}$.
3. **Span Proximity ($S_{\text{proximity}}$)**: Measures the distance between query terms in the text. Terms appearing adjacent or within a 15-word window receive high scores; scattered terms are discounted.
4. **Prior Score Anchor ($S_{\text{prior}}$)**: Incorporates min-max normalized initial retrieval score to maintain foundational relevance.

---

### D. Chunk Deduplication (Jaccard & MMR)

Sliding chunk windows often produce overlapping fragments from the same document section. Presenting duplicate paragraphs wastes token budget and reduces generation diversity.

1. **Greedy Jaccard Filtering**:
   $$J(A, B) = \frac{|\text{Tokens}(A) \cap \text{Tokens}(B)|}{|\text{Tokens}(A) \cup \text{Tokens}(B)|}$$
   If candidate $B$ has $J(A, B) \ge 0.70$ with an already accepted higher-ranked chunk $A$, candidate $B$ is excluded.
2. **Maximal Marginal Relevance (MMR)**:
   $$\text{MMR} = \operatorname{argmax}_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Score}(d_i) - (1 - \lambda) \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]$$
   Balancing relevance ($\lambda$) with novelty ($1 - \lambda$).

---

## 3. Architecture & Implementation Summary

- **`packages/rag/bm25.py`**: First-principles Okapi BM25 with suffix stemming and inverted index.
- **`packages/rag/fusion.py`**: Reciprocal Rank Fusion (RRF $k=60$) and min-max normalized score fusion.
- **`packages/rag/reranker.py`**: Heuristic multi-factor cross-scorer.
- **`packages/rag/deduplication.py`**: Jaccard threshold deduplication and MMR diversification.
- **`packages/rag/hybrid.py`**: Unified `HybridRetriever` orchestrating two-stage retrieval.
- **`apps/backend/api/v1/endpoints/rag.py`**: Updated `/api/v1/rag/query` with `mode`, RRF, reranking, and deduplication controls.
- **`apps/frontend/src/components/KnowledgeBaseView.tsx`**: Interactive UI for toggling Dense/BM25/Hybrid modes and inspecting score breakdowns.

---

## 4. Verification

- **RAG & API Test Suite**: **32 passed** (`pytest tests/rag/ tests/api/test_rag_endpoint.py`).
- **Full Test Suite**: **123 passed, 0 failed** in 31.86s (`pytest tests/`).
- **Next.js Production Build**: `npm run build` compiled with **0 errors**.
- **Interactive Script**: `scripts/run_phase14_hybrid_rag_demo.py` ran cleanly.
