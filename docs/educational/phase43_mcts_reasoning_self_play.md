# Phase 43: Reinforcement Learning via Self-Play & Monte Carlo Tree Search (MCTS / LibraReason)

> **Project Libra — First-Principles Educational LLM Laboratory**
> *Dual-Track Architecture: Interactive AI Product & First-Principles Foundation Lab*

---

## 1. WHAT: System Architecture & Deliverables

Phase 43 implements **LibraReason**, a first-principles Monte Carlo Tree Search (MCTS) engine, Process Reward Model (PRM), and self-play preference pair synthesizer running 100% locally on CPU at zero cost ($0 / ₹0).

### Core Components Delivered:
1. **Process Reward Model (PRM) & Step-Level Verifier (`packages/models/reasoning/prm.py`)**:
   - Scores intermediate reasoning steps individually ($0.0 \le r_{\text{step}} \le 1.0$), superseding Outcome Reward Models (ORMs) that only judge final outputs.
   - Verifies arithmetic equations directly ($a \odot b = c$) with exact calculation checks.
   - Detects circular assertions, logical contradictions, and physical impossibility.
   - Identifies the `first_error_index` to facilitate early tree pruning before downstream compute is wasted.

2. **Monte Carlo Tree Search for Reasoning (`packages/models/reasoning/mcts.py`)**:
   - Organizes reasoning as a tree of thought states where each node represents an intermediate reasoning step.
   - 4-phase MCTS lifecycle:
     1. **Selection**: Polynomial Upper Confidence Trees (PUCT):
        $$\operatorname{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
     2. **Expansion**: Branches candidate reasoning actions for unvisited nodes.
     3. **Evaluation**: Obtains intermediate validity $r \in [0, 1]$ via the Process Reward Model.
     4. **Backpropagation**: Propagates value $V$ and updates visit counts $N(s)$ and action-values $Q(s)$ up to the root.
   - Optimal path extraction: greedily extracts the verified trajectory leading to the final solution.
   - Full tree graph serialization for visualization.

3. **Self-Play Trajectory Generator (`packages/models/reasoning/self_play.py`)**:
   - Generates reasoning problems across arithmetic, algebra, and geometry.
   - Synthesizes branching candidate trajectories from the same problem state.
   - Employs the PRM to assign the valid path as `chosen` and the defective path as `rejected`.
   - Generates step-level DPO preference datasets `(prompt, chosen, rejected)` with reward margin $\Delta r$.

4. **FastAPI Endpoints (`apps/backend/api/v1/endpoints/mcts_reasoning.py`)**:
   - `POST /api/v1/reasoning/mcts/search`: Full MCTS step search returning tree graph and optimal path.
   - `POST /api/v1/reasoning/mcts/prm/score`: Step-level PRM scoring and error identification.
   - `POST /api/v1/reasoning/mcts/self_play/generate`: Synthetic self-play DPO pair generator.
   - `GET /api/v1/reasoning/mcts/presets`: Educational reasoning problem templates.

5. **Interactive MCTS Tree Visualizer UI (`apps/frontend/src/components/MCTSTreeView.tsx`)**:
   - **Search Tree Graph**: Displays reasoning steps by depth levels, color-coded by PRM score (green = valid, amber = exploring, red = pruned).
   - **Step Inspector Drawer**: Displays PUCT metrics ($N$, $Q$, PRM score) and root-to-node trajectory.
   - **PRM Step Verifier**: Interactive step validator highlighting the critical failure step.
   - **Self-Play DPO Console**: Side-by-side chosen vs rejected viewer.
   - Integrated into [`Sidebar.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/components/Sidebar.tsx) and rendered in [`page.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/app/page.tsx).

---

## 2. WHY: Problem Space & Theoretical Motivation

### Outcome vs. Process Supervision
Traditional Reinforcement Learning from Human Feedback (RLHF) uses Outcome Reward Models (ORMs): the model is rewarded only if the final answer matches ground truth. This suffers from two major problems:
1. **False Positives (Lucky Guesses)**: A model can follow nonsensical or erroneous reasoning steps and happen upon the correct final answer by coincidence.
2. **Delayed Feedback & Credit Assignment**: When an answer is wrong, the ORM penalizes every step equally, failing to identify which specific step went astray.

**Process Reward Models (PRMs)** solve this by evaluating every intermediate step ($r_{\text{step}}$), providing dense supervision that allows tree search to prune defective branches immediately.

---

## 3. HOW: Mathematical & Algorithmic Implementation

### A. PUCT Exploration-Exploitation Tradeoff
At each tree decision point, action $a$ is selected according to:
$$a^* = \arg\max_{a} \left[ Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)} \right]$$
Where:
- $Q(s, a)$: Average empirical reward of all rollouts passing through action $a$.
- $c_{\text{puct}}$: Exploration constant ($1.414$).
- $P(s, a)$: Prior probability assigned to candidate step $a$.
- $N(s)$: Total visits to parent state $s$.
- $N(s, a)$: Total visits to action branch $a$.

### B. PRM Step Scoring Function
A candidate step $x_t$ conditioned on history $x_{<t}$ is scored:
$$r(x_t \mid x_{<t}) = \sigma \left( w_0 + w_{\text{arith}} \cdot \mathbb{I}_{\text{arith}}(x_t) - w_{\text{contra}} \cdot \mathbb{I}_{\text{contra}}(x_t) - w_{\text{rep}} \cdot \mathbb{I}_{\text{rep}}(x_t, x_{<t}) \right)$$
If $r(x_t \mid x_{<t}) < 0.55$, the step is rejected and downstream expansion from this node is halted.

---

## 4. TEST: Verification & Regression Results

1. **Phase 43 Test Suites**:
   - `tests/models/test_process_reward_model.py`: Verified arithmetic validation, contradiction detection, repetition penalty, and `first_error_index` identification (3 tests).
   - `tests/models/test_mcts_reasoning.py`: Verified PUCT calculation, tree search on Game of 24, and self-play DPO pair generation (3 tests).
   - `tests/api/test_mcts_reasoning_endpoint.py`: Verified presets, MCTS search, PRM score, and self-play endpoints (4 tests).
   - **All 10 Phase 43 tests passed in 14.77s**.

2. **Full Project Regression**:
   - **490 passed, 0 failed** across all 43 phases.

3. **Frontend Production Build**:
   - Next.js 14 production build compiled cleanly with zero errors.

---

## 5. NEXT: Upcoming Roadmap

With Phase 43 complete, Project Libra possesses test-time tree search, process supervision, and automated self-play dataset generation.
The curriculum transitions to:
- **Phase 44: Knowledge Distillation & Model Shrinking (Teacher-Student Logit Transfer)**:
  - Temperature-scaled Kullback-Leibler (KL) divergence loss.
  - Hard vs Soft label distillation.
  - Shrinking multi-head attention to student sub-networks with zero loss in task accuracy.
