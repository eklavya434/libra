# Educational Guide: Phase 15 — Real-Time SSE Streaming UX, Markdown Rendering & Telemetry

Welcome to **Phase 15** of Project Libra. In this phase, we bridge the gap between raw backend token streams and a ChatGPT-grade user interface: streaming Markdown parsing, syntax-highlighted code blocks with copy actions, abortable streams via `AbortController`, reasoning `<think>` block collapsing, and high-precision generation telemetry.

---

## 1. The Streaming UI Challenge: Why Custom Parsers?

When an LLM streams tokens over Server-Sent Events (SSE), tokens arrive in irregular chunks (e.g. `"` $\to$ `def ` $\to$ `foo():` $\to$ `\n    ret` $\to$ `urn 42`).

Traditional Markdown parsers (e.g., standard Markdown-it or Marked) assume **complete, static documents**. When fed in-flight streams, they encounter severe rendering pathologies:
1. **Unclosed Code Fences**: When ` ```python ` is received, the code block is unclosed until the entire response finishes. Standard parsers either treat the remaining document as raw text or fail to apply syntax highlighting until the final ` ``` ` arrives minutes later.
2. **Layout Shifts & Jitter**: Re-parsing the entire accumulating markdown document on every single token causes DOM thrashing, cursor jumping, and noticeable frame drops.
3. **Reasoning Artifacts**: Modern reasoning models (e.g. DeepSeek-R1, Qwen 2.5) emit internal chain-of-thought enclosed in `<think>...</think>`. Without specialized parsing, this internal monologue clutters the conversation window.

Our solution is a **first-principles streaming parser** designed for token accumulation.

---

## 2. Technical Architecture & Protocols

### A. Server-Sent Events (SSE) Protocol Mechanics

FastAPI serves chat completions via the standard SSE format (`text/event-stream`):

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"choices": [{"delta": {"content": "Hello"}}]}

data: {"choices": [{"delta": {"content": " world"}}]}

data: [DONE]
```

On the Next.js frontend (`apps/frontend/src/lib/api.ts`), we process the raw byte stream using the web standard `ReadableStreamDefaultReader`:

```typescript
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  // Process complete newline-separated lines...
}
```

---

### B. In-Flight Code Fence State Machine

Our parser treats the stream as a sequential state machine:

```
[Normal Text] ---> (encounter "```python") ---> [In-Code State]
      ^                                                |
      |                                                |
(encounter "```") <------------------------------------+
```

1. **Closed Fences**: When closing ` ``` ` arrives, the code block is marked `isComplete: true`.
2. **In-Flight Open Fences**: If the stream is still generating and the code fence is open, our parser immediately renders a live `<CodeBlock isStreaming={true} />` container with a pulsing "generating..." indicator and line numbers. The user sees formatted code evolving in real-time.

---

### C. First-Principles Syntax Highlighting

Rather than importing a 250 KB heavy library (like PrismJS or Highlight.js), [`CodeBlock.tsx`](file:///c:/Users/eklav/Desktop/Libra/apps/frontend/src/components/CodeBlock.tsx) tokenizes lines using a zero-dependency lexical regex pattern:

```typescript
const tokenRegex = /(\/\/[^\n]*|#[^\n]*|--[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b[a-zA-Z_]\w*\b|\b\d+(?:\.\d+)?\b|[^\s\w]+|\s+)/g;
```

Tokens are classified into distinct semantic categories:
- **Keywords**: `def`, `class`, `return`, `async`, `await`, `const`, `function` $\to$ Purple (`text-purple-400`).
- **Strings**: Single/double/backtick quoted spans $\to$ Emerald (`text-emerald-400`).
- **Numbers**: Integer and floating-point literals $\to$ Amber (`text-amber-400`).
- **Comments**: `#`, `//`, `--` $\to$ Muted Italic (`text-slate-500 italic`).
- **Built-in Types**: `int`, `str`, `list`, `dict`, `Promise` $\to$ Cyan (`text-cyan-400`).

---

### D. Stream Cancellation with AbortController

Users can stop generation mid-stream by clicking the **Stop** button (`Square` icon).

1. `ChatArea.tsx` initializes `abortControllerRef = new AbortController()`.
2. `fetch(url, { signal: controller.signal, ... })` wires the HTTP connection to the signal.
3. When the user clicks Stop (or presses Escape), `abortController.abort()` immediately fires:
   - The browser cancels the TCP/TLS connection.
   - The fetch promise rejects with `AbortError`.
   - Our handler intercepts `AbortError`, finalizes telemetry for all tokens received up to that exact millisecond, and gracefully ends generation without error popups.

---

### E. Generation Telemetry Mathematics

For every completion, we measure and display:
1. **Time to First Token (TTFT)**:
   $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_start}}$$
   Directly reflects backend queueing, model prompt processing, and TTFT latency.
2. **Token Generation Velocity**:
   $$V = \frac{N_{\text{tokens}}}{t_{\text{end}} - t_{\text{first\_token}}}$$
   Measures pure autoregressive decoding speed in **tokens per second (tok/s)**.

---

## 3. Verification & Quota

- **Next.js Production Build**: `npm run build` compiled with **0 errors**.
- **Pytest Suite**: **126 / 126 tests passed**.
- **Interactive Script**: `scripts/run_phase15_streaming_demo.py` ran with full telemetry output.
- **Zero Bundle Overhead**: Added **0 bytes** to `package.json` dependencies.
