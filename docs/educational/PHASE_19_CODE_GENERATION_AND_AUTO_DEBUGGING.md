# Phase 19: Code Generation, Auto-Debugging & Program-Aided Language Models (PAL)

Welcome to **Phase 19** of **Project Libra**.

In this phase, we bridge the gap between generative language models and deterministic computation by implementing:
1. **Program-Aided Language Models (PAL)**: Offloading arithmetic, combinatorics, date/time logic, and symbolic computation to an isolated Python runtime.
2. **Test-Driven Auto-Debugging & Reflection Loop**: An autonomous self-healing software engineering harness that writes code, runs unit tests in the sandbox, diagnoses runtime crashes (`SyntaxError`, `IndexError`, `AssertionError`, `ZeroDivisionError`), and repairs the code through iterative reflection.

---

## 1. WHAT: Architecture & Components

```
User Task / Specification
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       Code / PAL Agent                      │
│   1. Few-shot prompt requesting structured Python solution  │
│   2. Extract clean Python code from markdown blocks         │
└─────────────────────────────────────┬───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    SafePythonSandbox (Phase 16)             │
│   - Process-isolated OS child process (subprocess.Popen)    │
│   - Windows Job Objects hard memory ceiling (64-128 MB)     │
│   - Preemptive timeout termination (proc.kill())            │
│   - Strict module allowlist (math, datetime, itertools...)  │
└───────────────────────┬─────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
    [ Success ]                   [ Exception ]
Code passes tests & assertions   SyntaxError / IndexError / AssertionError
         │                             │
         │                             ▼
         │             ┌───────────────────────────────┐
         │             │         AutoDebugger          │
         │             │  - Line-numbered code context │
         │             │  - Traceback & frame analysis │
         │             │  - Root-cause reflection      │
         │             │  - Synthesis of bug repair    │
         │             └───────────────┬───────────────┘
         │                             │
         │                             ▼ (Retry loop up to max_iterations)
         │                     Re-execute in Sandbox
         │                             │
         ▼                             ▼
   Final Answer Delivered / CodeTrajectory Summary
```

### Core Modules Implemented:

- **`packages/agents/pal.py`**:
  - `extract_code()`: Robust parser extracting Python code from ```` ```python ... ``` ````, generic ```` ``` ````, or raw scripts.
  - `normalize_pal_code()`: Normalizes PAL programs, ensuring `solution()` is invoked at top-level so results can be evaluated by the sandbox.
  - `PALAgent`: Translates natural language word problems into executable Python functions, executing them within `SafePythonSandbox` for deterministic answers.

- **`packages/agents/coder.py`**:
  - `DebugIteration`: Detailed per-step trace recording iteration number, candidate code, stdout, exception diagnostics, traceback, reflection, and fix explanation.
  - `CodeTrajectory`: Complete session container recording original prompt, assertions, initial code, final code, iteration history, success status, and duration.
  - `AutoDebugger`: Standalone test-driven debugging engine that consumes broken code + failed assertions, extracts line-level stack traces, and prompts the LLM for self-correction.
  - `CodeAgent`: High-level autonomous coding agent synthesizing code from specification, executing unit assertions, and auto-debugging failures until all tests pass.

- **`apps/backend/api/v1/endpoints/coder.py`**:
  - `POST /api/v1/coder/pal`: Solves word problems via program execution.
  - `POST /api/v1/coder/generate`: Generates code validated against unit assertions with auto-repair.
  - `POST /api/v1/coder/debug`: Diagnoses and repairs user-provided broken code snippets.

---

## 2. WHY: The Mathematical & Architectural Problem

### The Chain-of-Thought (CoT) Calculation Breakdown

Standard LLMs predict tokens based on conditional probability:
$$P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

When an LLM performs multi-digit arithmetic (e.g. $47 \times 89$ or $\binom{12}{5}$), it does not have an internal arithmetic register. Instead, it relies on pattern-matching tokens in its weights. As the number of reasoning steps increases, errors compound exponentially:
$$P(\text{correct}) = \prod_{k=1}^N P(\text{step } k \text{ correct})$$
If each arithmetic step has 95% accuracy, after 10 reasoning steps the final probability drops to $(0.95)^{10} \approx 59.8\%$.

### The PAL Solution (Gao et al., 2022)

Instead of asking the language model to *compute* the answer, we ask it to *write a Python program* that computes the answer. The LLM excels at symbolic formulation and translation, while the Python interpreter is 100% deterministic and exact:
$$P(\text{arithmetic error}) = 0$$

### The Auto-Debugging Self-Correction Loop

Language models frequently make minor syntax mistakes, off-by-one errors (`<` vs `<=`), or fail boundary test cases. Without execution feedback, these errors remain undetected.
By executing code in a sandboxed runtime, capturing the exact exception (`IndexError`, `AssertionError`) and stack frame (`File "<sandbox>", line 2`), and feeding this diagnostic trace back into the prompt, the model can reason about its own mistake and correct it.

---

## 3. HOW: Under the Hood

### 1. Stack Trace & Line Context Injection

When code crashes in `SafePythonSandbox`, `runner.py` captures the full Python traceback:
```python
Traceback (most recent call last):
  File "<sandbox>", line 2, in get_last
    return items[len(items)]
IndexError: list index out of range
```
The `AutoDebugger` formats the original code with line numbers:
```
  1 | def get_last(items: list):
  2 |     return items[len(items)]
```
And prompts the model:
> "Analyze what caused this error, explain your fix concisely, and provide the complete repaired Python code in a ```python ... ``` block."

### 2. Guardrails & Safety
- Code generation never bypasses the sandbox: all runs are executed inside `SafePythonSandbox`.
- Memory is strictly limited (Windows Job Objects / POSIX rlimit).
- Network is blackholed (`127.0.0.1:0`).
- Disallowed modules (`os`, `sys`, `subprocess`, `socket`) are blocked at static AST parse time.
- Infinite loops or hanging test cases are terminated preemptively via OS process termination (`proc.kill()`).

---

## 4. TEST: Verification Results

The Phase 19 test suite is verified in `tests/agents/` and `tests/api/`:
- `tests/agents/test_pal_agent.py`:
  - `test_extract_code_python_fence`: Verifies regex markdown block extraction.
  - `test_extract_code_raw_fallback`: Verifies fallback extraction on raw code with prose.
  - `test_normalize_pal_code`: Verifies `solution()` call appending and idempotency.
  - `test_pal_agent_combinatorics_calculation`: Exact computation of $\binom{12}{5} = 792$.
  - `test_pal_agent_date_arithmetic`: Exact computation of date offsets via `datetime`.
  - `test_pal_agent_execution_runtime_error`: Clean failure handling on unrecoverable code.
- `tests/agents/test_coder_agent.py`:
  - `test_code_generation_immediate_success`: Clean generation passing tests on attempt 1.
  - `test_auto_debugger_syntax_error_recovery`: Recovers from missing colon syntax error.
  - `test_auto_debugger_index_error_off_by_one`: Recovers from list index out of range.
  - `test_auto_debugger_assertion_failure_recovery`: Recovers from failed unit test assertion.
  - `test_auto_debugger_budget_exhaustion`: Graceful circuit-breaker termination when code is unfixable.
  - `test_code_agent_sandbox_security_rejection`: Sandbox blocks forbidden imports.
- `tests/api/test_coder_endpoint.py`:
  - `test_api_pal_endpoint`: `POST /api/v1/coder/pal`
  - `test_api_code_generate_endpoint`: `POST /api/v1/coder/generate`
  - `test_api_code_debug_endpoint`: `POST /api/v1/coder/debug`

All 20 tests pass in **7.77s**.

---

## 5. NEXT: Phase 20 Preview

With code generation, auto-debugging, and Program-Aided Language Models operating reliably, **Phase 20** introduces **Multi-Agent Collaboration & Orchestration**, where multiple specialized agents (e.g., Planner, Coder, Reviewer, Tester) communicate and collaborate to solve complex, multi-faceted engineering and research projects.
