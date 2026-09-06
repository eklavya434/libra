# Phase 16: Tool Use & Safe Sandbox Execution

Welcome to **Phase 16** of Project Libra. In this phase, we elevate Libra from a conversational text generator into an **agentic system capable of taking safe external actions**: calculating exact arithmetic, executing untrusted Python code in a kernel-enforced sandbox, searching the live web, and querying local knowledge base indexes.

---

## 1. WHAT: The Architecture of Tool Use in Libra

Phase 16 establishes the complete tool execution pipeline:

```
+-------------------------------------------------------------+
|                       User Message                          |
|         "What is sqrt(1764) and check the date?"            |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      LLM Generation                         |
|   Emits XML markup or OpenAI function calling JSON schema   |
|   <tool_call>                                               |
|     {"name": "calculator", "arguments": {"expression": ...}}|
|   </tool_call>                                              |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      ToolCallParser                         |
|   Extracts structured tool names, arguments & call IDs      |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       ToolRegistry                          |
|   Validates Pydantic schemas, dispatches execution          |
+-------------------------------------------------------------+
           |                   |                   |
           v                   v                   v
   +---------------+   +---------------+   +---------------+
   |CalculatorTool |   | WebSearchTool |   |SafePython-    |
   |AST Arithmetic |   |DuckDuckGo/Mock|   |Sandbox        |
   +---------------+   +---------------+   |Process-Level  |
                                           |Isolated OS    |
                                           |Child Process  |
                                           +---------------+
```

### Core Built-in Tools:
1. **`CalculatorTool`**: Evaluates mathematical and arithmetic expressions strictly using an Abstract Syntax Tree (AST) evaluator without `eval()`, preventing command injection while supporting power (`**`), trigonometric, logarithmic, and factorial functions.
2. **`SafePythonSandbox` & `PythonInterpreterTool`**: Executes dynamic Python code inside an isolated child process with:
   - **Static AST Import Allowlisting**: Rejects any imports outside `math`, `statistics`, `random`, `json`, `datetime`, `collections`, `itertools`, `re`, `string`, `decimal`, `fractions`.
   - **Kernel Memory Limit**: Windows Job Object enforcing memory caps (e.g. 64–128 MB), immediately aborting runaway memory bombs.
   - **Hard Preemptive Process Kill**: Unkillable thread loops (`while True: pass`) are terminated by the OS kernel (`proc.kill()`), eradicating lingering threads and GIL starvation.
   - **Filesystem Chroot Isolation**: Custom `safe_open()` strictly prevents path traversal outside an ephemeral temporary workspace directory.
   - **Subprocess Blocking**: Windows Job Object `ActiveProcessLimit = 1` blocks child processes from spawning secondary subprocesses.
   - **Network Blackholing**: Environment proxies redirected to `127.0.0.1:0`.
3. **`WebSearchTool`**: Fetches live web pages and summaries via DuckDuckGo without paid API keys.
4. **`KnowledgeBaseTool`**: Grounded document retrieval over local documents via Reciprocal Rank Fusion (BM25 + Dense embeddings).

---

## 2. WHY: The Danger of In-Process Thread Abandonment

In Python, threads **cannot be preemptively stopped or killed** from another thread. A naive timeout pattern:

```python
# DANGEROUS / FLAWED TIMEOUT PATTERN:
worker_thread = threading.Thread(target=_worker, daemon=True)
worker_thread.start()
worker_thread.join(timeout=0.3)
if worker_thread.is_alive():
    raise TimeoutError("Timed out")
```

When an adversarial snippet runs `while True: pass`:
1. `join(timeout=0.3)` times out.
2. The caller raises `TimeoutError`.
3. **The thread continues spinning in the background**.
4. In CPython, that runaway thread fiercely competes for the Global Interpreter Lock (GIL), starving other threads and halting subsequent operations or test suites.

### The Solution: True OS Process Isolation

By spawning a separate operating system child process (`subprocess.Popen([sys._base_executable, ...])`):
- `proc.communicate(timeout=timeout)` monitors execution.
- On `TimeoutExpired`, the host issues `proc.kill()`.
- The operating system kernel immediately invokes Win32 `TerminateProcess` or POSIX `SIGKILL`, destroying every thread in the child process.
- Zero leftover threads, zero CPU pegging, zero GIL contention.

---

## 3. HOW: AST Security & Job Object Enforcement

### AST Security Visitor:
Before bytecodes are ever created, the code is parsed into an AST:
```python
class SecurityVisitor(ast.NodeVisitor):
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden import: '{alias.name}'")
```

### Windows Job Object Limits:
```python
limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
limits.BasicLimitInformation.LimitFlags = (
    JOB_OBJECT_LIMIT_PROCESS_MEMORY |
    JOB_OBJECT_LIMIT_JOB_MEMORY |
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS |
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
)
limits.BasicLimitInformation.ActiveProcessLimit = 1
limits.ProcessMemoryLimit = 64 * 1024 * 1024
kernel32.SetInformationJobObject(job, 9, byref(limits), sizeof(limits))
kernel32.AssignProcessToJobObject(job, int(proc._handle))
```

---

## 4. TEST: Adversarial Verification

All 5 adversarial security boundaries are strictly tested and verified:
1. `test_adversarial_infinite_loop`: Hard timeout kills child in <0.5s without GIL stalling.
2. `test_adversarial_memory_bomb`: 64MB allocation triggers `MemoryError` without crashing the host.
3. `test_adversarial_filesystem_escape_relative`: `../../` path traversal denied with `PermissionError`.
4. `test_adversarial_filesystem_escape_absolute`: Access to `C:/Windows` denied.
5. `test_adversarial_disallowed_imports`: Imports of `socket`, `os`, `sys`, `subprocess`, `requests` statically rejected.
6. `test_adversarial_reflection_breakout`: Dunder calls `().__class__.__subclasses__()` rejected.
7. `test_legitimate_scoped_filesystem_io`: Safe file operations inside the sandbox execute cleanly.

---

## 5. NEXT: Phase 17 Preview

With robust tool execution and sandboxing complete, **Phase 17** introduces **Structured Outputs & Schema Enforcing Decoders**, ensuring model generations strictly adhere to JSON Schemas via CFG (Context-Free Grammar) guidance before advancing to Phase 18 (Multi-step Agentic Loops).
