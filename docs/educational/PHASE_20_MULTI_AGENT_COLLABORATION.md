# Phase 20: Multi-Agent Collaboration & Orchestration

Welcome to **Phase 20** of **Project Libra**.

In this phase, we evolve beyond single-agent architectures (ReAct, Plan-and-Solve, PAL) by implementing an autonomous, role-specialized **Multi-Agent Collaborative Team**:
1. **Role Specialization**: `Architect`, `Coder`, `Reviewer`, `Tester`, and `Coordinator` personas.
2. **Inter-Agent Message Bus & Shared Blackboard**: Structured communication and synchronized artifact state.
3. **Iterative Consensus & Adversarial Review Loop**: A self-auditing workflow where proposals are critiqued, verified in the `SafePythonSandbox`, and revised until multi-agent consensus is reached.

---

## 1. WHAT: Architecture & Components

```
                      User Task / Engineering Objective
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │      Shared Blackboard       │
                      │  - Task & Specification      │
                      │  - Candidate Code            │
                      │  - Test Assertions           │
                      │  - Review Notes & Verdict    │
                      │  - Sandbox Execution Trace   │
                      └──────────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  ArchitectAgent  │        │    CoderAgent    │        │  ReviewerAgent   │
│  - Requirements  │        │  - Synthesis     │        │  - Adversarial   │
│  - Signatures    │───────►│  - Revision      │◄──────►│    Code Audit    │
│  - Edge Cases    │        │  - Formatting    │        │  - Verdict       │
└──────────────────┘        └────────┬─────────┘        └──────────────────┘
                                     │
                                     ▼
                            ┌──────────────────┐
                            │   TesterAgent    │
                            │  - Unit Tests    │
                            │  - Safe Sandbox  │
                            │  - Verification  │
                            └────────┬─────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
            [ Consensus ]                     [ Revisions Needed ]
    Reviewer APPROVED + Tests PASSED       Reviewer REVISION or Tests FAILED
                    │                                 │
                    ▼                                 ▼
            Final Delivery                   Next Iteration Round
```

### Core Modules Implemented:

- **`packages/agents/message_bus.py`**:
  - `AgentRole`: `COORDINATOR`, `ARCHITECT`, `CODER`, `REVIEWER`, `TESTER`.
  - `MessageType`: `TASK`, `SPECIFICATION`, `CODE_PROPOSAL`, `CRITIQUE`, `TEST_RESULTS`, `CONSENSUS_APPROVAL`, `FINAL_DELIVERY`.
  - `AgentMessage`: Structured message schema with metadata, timestamps, and routing.
  - `SharedBlackboard`: Central state repository managing task, code artifacts, unit assertions, review notes, and message logs.

- **`packages/agents/multi_agent.py`**:
  - `SpecializedAgent`: Autonomous agent wrapping a dedicated role, system prompt, and LLM provider.
  - `CollaborativeTeam`: Orchestrator managing multi-agent rounds:
    - **Stage 1 (Planning)**: `ArchitectAgent` defines functional specs and interface contracts.
    - **Stage 2 (Implementation)**: `CoderAgent` synthesizes code meeting the spec.
    - **Stage 3 (Adversarial Review)**: `ReviewerAgent` audits code for logic bugs, security, and edge cases.
    - **Stage 4 (Sandbox Testing)**: `TesterAgent` generates assertions and executes them in `SafePythonSandbox`.
    - **Stage 5 (Consensus or Revision)**: If revisions are requested or tests fail, the Coder addresses feedback in the next round.
  - `TeamRound` & `TeamTrajectory`: Telemetry containers capturing per-round snapshots and full session history.

- **`apps/backend/api/v1/endpoints/teams.py`**:
  - `POST /api/v1/teams/collaborate`: Synchronous multi-agent collaboration returning `TeamTrajectory`.
  - `POST /api/v1/teams/collaborate/stream`: Real-time Server-Sent Events (SSE) streaming of dialogue and rounds.

---

## 2. WHY: Cognitive & Mathematical Foundations

### 1. The Confirmation Bias Trap in Single-Agent Systems
When a single language model generates code and is subsequently asked to evaluate its own output in the same prompt, it exhibits severe **confirmation bias**:
$$P(\text{spot bug} \mid \text{self-authored}) \ll P(\text{spot bug} \mid \text{adversarial reviewer})$$
By separating the generation role (`CoderAgent`) from the critique role (`ReviewerAgent`) with distinct system instructions, the reviewer is primed to search for flaws rather than defend the existing output.

### 2. Condorcet's Jury Theorem in Agent Ensembles
Condorcet's Jury Theorem states that if a group of independent decision-makers each have a probability $p > 0.5$ of reaching a correct decision, the probability $P_n$ of the majority vote being correct approaches 1 as the number of voters $n$ increases:
$$P_n = \sum_{k=\lfloor n/2 \rfloor + 1}^n \binom{n}{k} p^k (1-p)^{n-k} \xrightarrow{n \to \infty} 1$$
In software engineering tasks, requiring independent consensus across three distinct filters (**Specification Alignment**, **Adversarial Code Review**, and **Sandboxed Unit Test Execution**) drives the composite failure rate towards zero.

### 3. The Blackboard Architectural Pattern
Originating in early AI systems (Hearsay-II, 1980), the **Blackboard Pattern** decouples specialized problem solvers. Agents do not need to know the internal prompt or state of other agents; they communicate asynchronously by reading from and writing to a shared global workspace.

---

## 3. HOW: Under the Hood

### 1. The 4-Stage Iteration Cycle

In each round:
1. **Coder** reads the blackboard (including previous critique and test failures) and generates code inside a ```python ... ``` block.
2. **Reviewer** audits the code and terminates with either:
   - `VERDICT: APPROVED`
   - `VERDICT: REVISION_REQUESTED`
3. **Tester** generates unit assertions and runs `SafePythonSandbox.execute(candidate_code + assertions)`.
4. **Coordinator** evaluates the consensus condition:
   $$\text{Consensus} = (\text{Reviewer Approved}) \land (\text{Sandbox Tests Passed})$$
   - If `True`, the team halts with `AgentStatus.COMPLETED`.
   - If `False`, the round increments and the Coder receives the critique and error trace.

### 2. Sandboxed Isolation
All test assertions authored by the `TesterAgent` are executed strictly inside the OS-isolated, memory-limited `SafePythonSandbox` developed in Phase 16, preventing arbitrary system damage or hanging threads.

---

## 4. TEST: Verification Results

Verified across `tests/agents/test_multi_agent.py` and `tests/api/test_teams_endpoint.py`:
- `test_shared_blackboard_operations`: Message posting, filtering, dialogue formatting.
- `test_multi_agent_consensus_round_1`: Direct consensus convergence when code is clean.
- `test_multi_agent_reviewer_revision_loop`: Reviewer rejects division by zero -> Coder repairs in Round 2 -> Consensus.
- `test_multi_agent_test_failure_triggers_revision`: Test fails in sandbox -> Coder fixes logic in Round 2 -> Consensus.
- `test_multi_agent_max_rounds_exhaustion`: Clean circuit breaker termination when consensus cannot be reached.
- `test_multi_agent_streaming_events`: Real-time SSE event emission.
- `test_api_teams_collaborate`: REST endpoint verification.
- `test_api_teams_collaborate_stream`: Streaming REST endpoint verification.

All 8 tests pass in **6.50s**.

---

## 5. NEXT: Phase 21 Preview

With multi-agent collaboration operating reliably, **Phase 21** introduces **Dynamic Model Routing & Semantic Speculative Decoding**, where queries are classified by complexity, intent, and domain, dynamically routing simple tasks to tiny local models and complex tasks to frontier multi-agent ensembles.
