# Phase 18: Multi-Step Autonomous Agent Loops (ReAct / Plan-and-Solve)

Welcome to **Phase 18** of Project Libra. In this phase, we graduate from single-step tool execution and schema decoding into **stateful multi-step autonomous agent systems**. We implement the classic **ReAct (Reasoning + Acting)** loop and the **Plan-and-Solve** deliberate architecture from mathematical and engineering first principles.

---

## 1. WHAT: The Architecture of Autonomous Agents in Libra

An autonomous LLM agent is an autoregressive loop that iteratively interacts with external environments through tools until a stopping condition is achieved:

```
+-------------------------------------------------------------+
|                        User Task                            |
|    "Calculate sqrt(144) * 5 and search latest Libra docs"   |
+-------------------------------------------------------------+
                               |
                               v
                       [AGENT REASONING]
                      +-----------------+
              +-----> | Thought (Chain) |
              |       +-----------------+
              |                |
              |                v
              |       +-----------------+
              |       | Action Selected |
              |       +-----------------+
              |                |
              |                v
     [UPDATE  |       +-----------------+
    SCRATCHPAD|       | Tool Execution  |
              |       +-----------------+
              |                |
              |                v
              |       +-----------------+
              +------ |   Observation   |
                      +-----------------+
                               |
                (When Final Answer is Reached)
                               v
                      +-----------------+
                      |  Final Answer   |
                      +-----------------+
```

### Supported Paradigms:
1. **`ReActAgent` (Reasoning + Acting)**:
   - Alternates `Thought:` (verbal reasoning and planning), `Action:` (invoking a specific tool with arguments), and `Observation:` (feedback from the tool execution).
   - Keeps an explicit scratchpad history allowing the agent to diagnose errors, reflect on partial results, and pivot dynamically.
   - Circuit breakers stop execution if a tool fails repeatedly (`max_tool_failures=3`).
   - Hard budget limit guards against runaway execution costs (`max_steps=10`, `timeout_sec=60.0`).
2. **`PlanAndSolveAgent`**:
   - **Stage 1 (Planner)**: Generates an explicit sequential execution plan (`ExecutionPlan`) with discrete milestones using `StructuredOutputGenerator`.
   - **Stage 2 (Solver)**: Steps through each milestone sequentially, executing tools when necessary and tracking accumulated findings.
   - **Stage 3 (Synthesis)**: Synthesizes all milestone findings into a cohesive, grounded response.
3. **Real-Time SSE Streaming**:
   - The UI subscribes to `POST /api/v1/agents/react/stream`.
   - Emits granular event streams: `thought`, `action`, `observation`, `final_answer`, and `finish` for live step progression.

---

## 2. WHY: Cognitive Synergies of Interleaving Reasoning & Acting

In standard "direct" generation, an LLM must emit an entire answer in one shot. If external facts or arithmetic calculations are required, errors compound uncontrollably.

In **ReAct**:
- **Reasoning grounds Actions**: The `Thought` step allows the model to state what it needs and why, formulating the exact input parameters for tools.
- **Actions ground Reasoning**: The `Observation` step brings real, non-hallucinated data back from external APIs, sandboxes, or search engines.
- **Trace Transparency**: Every step is preserved in `AgentTrajectory`, providing full auditability and explainability.

---

## 3. HOW: Step Parsing & Circuit Breakers

### Tool Parsing & Final Answer Extraction:
```python
if "Final Answer:" in raw_output:
    parts = raw_output.split("Final Answer:", 1)
    thought = parts[0].replace("Thought:", "").strip()
    final_answer = parts[1].strip()
    return complete_trajectory(thought, final_answer)

tool_calls = ToolCallParser.parse(raw_output)
if tool_calls:
    call = tool_calls[0]
    res = await registry.execute_tool_async(call.name, call.arguments)
    obs = str(res.output) if res.success else f"Error: {res.error}"
```

### Circuit Breakers & Budget Guards:
- **Consecutive Tool Errors**: If a tool fails 3 times consecutively, the circuit breaker trips and halts the loop with `AgentStatus.FAILED`.
- **Max Steps**: If `step_idx > max_steps`, the loop stops safely with a timeout or max steps exceeded error.
- **Wall-Clock Timeout**: Verified every iteration using `time.perf_counter()`.

---

## 4. TEST: Verification Results

The agent suite is verified in `tests/agents/`:
- `test_react_direct_answer`: Handles prompts without tools in 1 step.
- `test_react_multi_step_calculation`: Chains calculator tool execution and observation into final answer in 2 steps.
- `test_react_max_steps_budget_limit`: Halts infinite agent loops cleanly at step ceiling.
- `test_react_circuit_breaker`: Halts when tools fail consecutively.
- `test_react_streaming_events`: Confirms SSE step event delivery.
- `test_plan_and_solve_execution`: Decomposes multi-step tasks into milestones and synthesizes result.
- Full repository test pass: **205 passed in 29.11s**.

---

## 5. NEXT: Phase 19 Preview

With multi-step autonomous agent loops operating reliably, **Phase 19** introduces **Code Generation, Auto-Debugging & Program-Aided Language Models (PAL)**, where the agent writes, self-evaluates, debugs, and tests Python programs directly within the secure sandbox to solve analytical and mathematical tasks.
