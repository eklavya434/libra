# Phase 17: Structured Outputs & Grammar-Constrained Decoding

Welcome to **Phase 17** of Project Libra. In this phase, we bridge the gap between free-form autoregressive text generation and **guaranteed schema-compliant outputs**. We implement **Context-Free Grammar (CFG) and JSON Schema-constrained decoding** from mathematical first principles.

---

## 1. WHAT: The Architecture of Structured Outputs in Libra

Large Language Models are probabilistic next-token predictors. When tasked with emitting JSON or conforming to an API contract, standard unconstrained models frequently produce:
- Malformed syntax (unclosed braces, trailing commas).
- Incorrect data types (strings where integers are expected).
- Hallucinated or missing required schema keys.

```
+-------------------------------------------------------------+
|                     Target Pydantic Schema                  |
|          class UserProfile(BaseModel): user_id: int         |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                       SchemaCompiler                        |
|       Translates schema into JSON Schema + Constraints      |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               IncrementalJSONStateMachine (PDA)             |
|   Pushdown automaton evaluating prefix character-by-character|
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                ConstrainedLogitsProcessor                   |
|   Applies mask to raw logits: z'_t(v) = -inf for invalid v  |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                Guaranteed Valid JSON Generation             |
+-------------------------------------------------------------+
```

### Core Components:
1. **`IncrementalJSONStateMachine`**: A character-level Pushdown Automaton (PDA) tracking nested object and array contexts (`stack`), literal tokens (`true`, `false`, `null`), numbers, and strings. It determines:
   - `is_valid_prefix(prefix: str) -> bool`: Can this prefix be continued into valid JSON?
   - `get_allowed_characters() -> Set[str]`: Which characters are legally allowed next?
   - `is_complete() -> bool`: Has a complete, closed JSON root value been emitted?
2. **`SchemaCompiler` & `SchemaConstraint`**: Compiles Pydantic classes or JSON Schema dictionaries into:
   - OpenAI/Ollama `response_format={"type": "json_schema", ...}` payloads.
   - Validation methods (`validate_text()`) verifying required fields and types.
   - System prompts with `<json_schema>` tags for models without native schema support.
3. **`ConstrainedLogitsProcessor`**: Intercepts autoregressive token logits $z_t \in \mathbb{R}^{|\mathcal{V}|}$, evaluating whether each candidate token maintains a valid JSON prefix. Tokens that would break the JSON grammar receive $-\infty$ logits.
4. **`StructuredOutputGenerator`**: High-level orchestrator with a multi-turn **Self-Healing Reflection Loop**: if a model returns an invalid structure, the generator reflects the exact validation error back to the model for automatic repair.

---

## 2. WHY: The Mathematics of Logit Masking

During standard autoregressive decoding at step $t$:
$$P(w_t = v \mid w_{<t}) = \frac{\exp(z_t(v))}{\sum_{j \in \mathcal{V}} \exp(z_t(j))}$$

Every token in the vocabulary $\mathcal{V}$ has a non-zero probability of being sampled unless temperature is 0, and even at temperature 0, the greedy token may produce invalid syntax (e.g. emitting a letter inside a numeric field).

In **grammar-constrained decoding**, the state machine determines the valid candidate token subset:
$$\mathcal{V}_{\text{valid}}(w_{<t}) = \{ v \in \mathcal{V} \mid w_{<t} \circ \text{decode}(v) \text{ is a valid prefix} \}$$

We apply a hard logit mask:
$$z'_t(v) = \begin{cases} z_t(v) & \text{if } v \in \mathcal{V}_{\text{valid}}(w_{<t}) \\ -\infty & \text{otherwise} \end{cases}$$

Then:
$$P(w_t = v \mid w_{<t}) = \frac{\exp(z'_t(v))}{\sum_{j \in \mathcal{V}} \exp(z'_t(j))}$$

Since $\exp(-\infty) = 0$, the probability of sampling any token outside $\mathcal{V}_{\text{valid}}$ is **strictly zero**. The generated text is mathematically guaranteed to be a valid prefix at every step.

---

## 3. HOW: Pushdown Automaton State Transitions

The `IncrementalJSONStateMachine` maintains state transitions:

| State | Allowed Transitions | Transition Target |
| :--- | :--- | :--- |
| `START` | `{`, `[`, `"`, digits, `-`, `t`, `f`, `n` | `IN_OBJECT_START`, `ARRAY`, `STRING`, `NUMBER`, `LITERAL` |
| `IN_OBJECT_START` | `"` (key), `}` (empty object) | `IN_KEY_STRING`, `DONE` |
| `IN_KEY_STRING` | Any valid character, `\"` (escaped), `"` (end of key) | `AFTER_KEY` |
| `AFTER_KEY` | `:`, whitespace | `EXPECT_VALUE` |
| `EXPECT_VALUE` | `{`, `[`, `"`, digits, `-`, `t`, `f`, `n` | Nested value state |
| `EXPECT_COMMA_OR_END` | `,` (next item), `}` or `]` (close container) | `EXPECT_KEY` (object) / `EXPECT_VALUE` (array) or `DONE` |
| `DONE` | Whitespace, `<eos>` token | End of generation |

---

## 4. TEST: Verification Results

1. **State Machine Transitions**: Validates incremental feeding of complex nested objects, arrays, escaped strings, floats, and empty containers.
2. **Logit Masking**: Confirms that non-JSON tokens receive $-\infty$ and legal tokens retain original scores.
3. **Self-Healing Reflection**: Demonstrates that if an initial generation lacks a required key, the generator feeds the validation error back and successfully receives a corrected object on attempt 2.
4. **REST Endpoints**:
   - `POST /api/v1/structured/validate`: Validates raw text against schema.
   - `POST /api/v1/structured/generate`: Returns guaranteed structured output.

---

## 5. NEXT: Phase 18 Preview

With guaranteed structured outputs and safe sandboxed tool execution complete, **Phase 18** combines both capabilities into **Multi-Step Autonomous Agent Loops (ReAct / Plan-and-Solve)**, allowing Libra to reason, act, observe tool outputs, and iteratively reach complex goals.
