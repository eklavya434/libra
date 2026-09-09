# Phase 48: Reasoning via Verifiable Search (Tree-Search Reasoning with PRMs & Best-of-N Guidance)

Project Libra — First-Principles Educational Laboratory & ChatGPT-like Assistant

---

## 1. WHAT: What Was Built

Phase 48 implements **Step-Level Verifiable Search (SV-Search)** for educational language models in Project Libra from mathematical first principles in PyTorch. It equips models with test-time compute scaling through intermediate step verification, early branch pruning, and automatic frontier backtracking.

Key Deliverables:
1. **Verifiable Search Engine (`packages/models/reasoning/verifiable_search.py`)**:
   - `SearchStepNode`: Represents a node in the reasoning tree, storing intermediate step content, depth, Process Reward Model (PRM) score, cumulative confidence, status (`accepted`, `pruned`, `terminal`), and rationale.
   - `VerifiableSearchEngine`: Implements beam/tree search over reasoning steps. Evaluates candidate steps with `ProcessRewardModel`, immediately detects arithmetic mismatches and logical contradictions, prunes flawed branches early, and backtracks to the next best candidate on the search frontier.
   - `VerifiableSearchResult`: Returns the verified solution trajectory, tree nodes and edges for visual graph exploration, step counts, pruned branch counts, and compute savings percentage.
2. **Quantitative Search Evaluator (`packages/evaluation/verifiable_search_eval.py`)**:
   - `VerifiableSearchEvaluator`: Quantitatively benchmarks:
     - Greedy Autoregressive Generation (no search, no backtracking).
     - Best-of-N Outcome-Only Search (ORM scoring on complete trajectories).
     - Step-Level Verifiable Search (PRM intermediate verification + early pruning).
     - Compares accuracy, token compute expenditure, early prune rate, and latency.
3. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/verifiable_search.py`)**:
   - `POST /api/v1/verifiable-search/solve`: Executes verifiable step search on reasoning prompts.
   - `POST /api/v1/verifiable-search/benchmark`: Runs side-by-side benchmark across search paradigms.
   - `GET /api/v1/verifiable-search/presets`: Provides educational problem sets (compound arithmetic, word problems, logic riddles).
4. **Interactive Next.js Reasoning UI (`apps/frontend/src/components/VerifiableSearchView.tsx`)**:
   - Interactive tree visualizer with color-coded nodes: emerald for verified steps ($r \ge \tau$), rose for pruned branches ($r < \tau$), and purple for terminal solutions.
   - Step detail inspector displaying intermediate PRM confidence, cumulative path confidence, and detected flaws.
   - Search parameters controls (beam width, prune threshold $\tau$, max depth).
   - Side-by-side paradigm benchmark table with test-time compute scaling telemetry.

---

## 2. WHY: Why It Exists & What Problem It Solves

### The Error Cascade in Autoregressive Generation
In standard autoregressive language generation, tokens and reasoning steps are produced strictly left-to-right without backtracking. In complex mathematical deduction or formal logic, an arithmetic error in Step 1 (e.g. $15 \times 4 = 55$ instead of $60$) corrupts every subsequent calculation. The model cannot recover, leading to guaranteed hallucinated answers.

### Limitations of Outcome Reward Models (ORMs)
Outcome Reward Models (e.g. Best-of-N in Phase 29) score only the *final complete trajectory* $r(s_T)$:
1. **Wasted Compute**: If a reasoning trajectory fails at Step 1, generating the remaining $T-1$ steps wastes massive FLOPs on a doomed path.
2. **Sparse Feedback**: An ORM provides zero credit assignment—it cannot indicate *which* step failed.
3. **False Positives**: An incorrect reasoning path may coincidentally arrive at the correct final number (e.g. two offsetting mistakes), misleading the model.

### The Power of Verifiable Search (PRMs)
Process Reward Models evaluate intermediate steps $s_t$:
- **Granular Credit Assignment**: Evaluates $r(s_t \mid s_{<t}) \in [0, 1]$ at every step.
- **Early Pruning**: Pruning a flaw at depth $d=1$ avoids exploring up to $B^{D-1}$ doomed sub-nodes.
- **Frontier Backtracking**: When a branch is pruned, search resumes from the next most promising prefix on the active priority queue.

---

## 3. HOW: Mathematical Foundations & Code Implementation

### A. Step Verification & PRM Formulation
For an intermediate reasoning step $s_t$ given context $s_{<t}$:
$$r(s_t) = P(\text{Step } t \text{ is mathematically and logically valid} \mid s_{<t})$$
The step is pruned if:
$$r(s_t) < \tau_{\text{prune}} \quad \text{or} \quad \text{Errors}(s_t) \ne \emptyset$$
where $\tau_{\text{prune}} = 0.55$ by default.

### B. Cumulative Path Confidence
The cumulative confidence of a reasoning path $\pi = (s_1, s_2, \dots, s_t)$ is computed as the soft geometric mean:
$$C(\pi) = \left( \prod_{i=1}^t \max(\epsilon, r(s_i)) \right)^{1/\sqrt{t}}$$
This prevents long correct reasoning chains from unfairly suffering from multiplicative decay while rewarding consistently sound reasoning.

### C. Frontier Search & Backtracking Algorithm
1. Initialize search frontier with root problem node $s_0$: $\mathcal{F} = \{s_0\}$.
2. Pop highest-scoring node $u = \text{argmax}_{n \in \mathcal{F}} C(n)$.
3. If $u$ is terminal (contains final answer) or reached `max_depth`, add to completed set.
4. Generate $B$ candidate next steps $\{c_1, \dots, c_B\}$.
5. For each candidate $c_i$:
   - Score with PRM: $r_i = \text{PRM}(c_i \mid \text{history}(u))$.
   - If $r_i < \tau_{\text{prune}}$: mark $c_i$ as **pruned**, increment prune counter.
   - If $r_i \ge \tau_{\text{prune}}$: mark $c_i$ as **accepted**, insert $c_i$ into $\mathcal{F}$.
6. If all candidates of $u$ were pruned, a **backtrack** is recorded; search seamlessly shifts to the next highest-scoring alternative node on $\mathcal{F}$.
7. Return optimal trajectory with highest cumulative confidence.

---

## 4. TEST: How It Was Verified

1. **Unit Test Suite (`pytest`)**:
   - `tests/models/test_verifiable_search.py`:
     - Verified `SearchStepNode` attribute serialization.
     - Verified compound arithmetic problem solving with automatic PRM error pruning and backtracking.
     - Verified custom generator step verification.
   - `tests/evaluation/test_verifiable_search_eval.py`:
     - Verified `VerifiableSearchEvaluator.evaluate_search` accuracy and prune metrics.
     - Verified `compare_search_methods` comparing Greedy vs Best-of-N vs Verifiable Search.
   - `tests/api/test_verifiable_search_endpoint.py`:
     - Verified `/presets`, `/solve`, and `/benchmark` API routes.
2. **Full Repository Regression**:
   - **546 / 546 unit tests passing** across all 48 phases on CPU.
3. **Static Analysis & Linting**:
   - `ruff check .` and `ruff format --check .`: 100% clean.
4. **Production Frontend Build**:
   - `npm run build`: Compiled 4/4 static pages cleanly with zero TypeScript errors.

---

## 5. NEXT: What the Next Phase Requires

The curriculum transitions to:
- **Phase 49: Self-Rewarding Language Models (Iterative DPO with LLM-as-a-Judge)**:
  - Policy acts simultaneously as reasoning generator and reward evaluator.
  - Iterative self-training loops without external reward models or human annotations.
  - LLM-as-a-Judge self-scoring with position bias mitigation.
