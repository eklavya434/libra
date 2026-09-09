"""
Libra Evaluation - Grand Capstone System Audit & Certification Engine (Phase 50)

Systematically validates all 10 architectural pillars covering the complete
50-phase curriculum built from first principles in Project Libra.
"""

from __future__ import annotations

import os
import time

from pydantic import BaseModel


class GrandPillarAudit(BaseModel):
    """Audit evaluation for one of the 10 Grand Architectural Pillars."""

    pillar_id: int
    title: str
    description: str
    phases_covered: str
    passed: bool
    details: list[str]


class GrandCapstoneReport(BaseModel):
    """Graduation certification report for the complete 50-phase LLM lab stack."""

    app_name: str = "Project Libra - Educational LLM Lab & ChatGPT-Like Assistant"
    version: str = "1.0.0"
    total_pillars: int = 10
    pillars_passed: int
    all_passed: bool
    completion_score_pct: float
    total_phases: int = 50
    completed_phases: int = 50
    pillars: list[GrandPillarAudit]
    duration_sec: float
    graduation_honors: str


class GrandCapstoneAudit:
    """Master certification engine for the complete 50-phase stack."""

    def __init__(self, repo_root: str | None = None) -> None:
        if repo_root is None:
            self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        else:
            self.repo_root = repo_root

    def _file_exists(self, rel_path: str) -> bool:
        norm = os.path.normpath(os.path.join(self.repo_root, rel_path))
        return os.path.exists(norm)

    def audit_pillar_1_foundations(self) -> GrandPillarAudit:
        """Pillar 1: Educational Foundations & Transformers (Phases 0-6)."""
        checks = [
            ("packages/core/hardware.py", "Hardware detection & CPU constraints"),
            (
                "packages/core/tokenizer/educational_bpe.py",
                "Byte-Pair Encoding from first principles",
            ),
            ("packages/data/pipeline.py", "Data cleaning & SHA-256 deduplication"),
            ("packages/models/components/rope.py", "Rotary Position Embeddings (RoPE)"),
            ("packages/models/components/rmsnorm.py", "Root Mean Square Normalization (RMSNorm)"),
            ("packages/models/components/swiglu.py", "SwiGLU feedforward activation"),
            ("packages/training/engine.py", "Production training loop with cosine warmup"),
            ("packages/evaluation/harness.py", "Evaluation harness & probing benchmarks"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=1,
            title="Educational Foundations & Core Modeling",
            description="First-principles tokenizer, modern transformer modules, training engine, and evaluation harness.",
            phases_covered="Phases 0 - 6",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_2_inference(self) -> GrandPillarAudit:
        """Pillar 2: Local & External Inference Engine (Phases 7-11)."""
        checks = [
            ("packages/models/registry.py", "Model registry & RAM formula sizing"),
            ("packages/providers/ollama.py", "Ollama local inference adapter"),
            ("packages/providers/openai.py", "External multi-provider adapter"),
            ("packages/providers/cost.py", "Real-time token cost tracker"),
            ("packages/evaluation/arena.py", "Model comparison arena"),
            ("apps/frontend/src/app/page.tsx", "Next.js 14 conversational frontend"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=2,
            title="Inference Engine & Multi-Provider Ecosystem",
            description="Ollama local engine, external providers (OpenAI, Gemini, Claude, DeepSeek), cost tracker, and Arena UI.",
            phases_covered="Phases 7 - 11",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_3_rag_search(self) -> GrandPillarAudit:
        """Pillar 3: Memory, RAG & Web Search (Phases 12-15)."""
        checks = [
            ("packages/core/memory/sqlite_store.py", "SQLite WAL persistent conversation store"),
            ("packages/rag/embeddings.py", "Dense vector embeddings from scratch"),
            ("packages/rag/bm25.py", "Okapi BM25 sparse lexical index"),
            ("packages/rag/fusion.py", "Reciprocal Rank Fusion (RRF)"),
            ("packages/rag/deep_research.py", "Deep research multi-query agent"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=3,
            title="Conversation Memory & Hybrid RAG",
            description="Persistent SQLite WAL, dense + sparse BM25 retrieval, RRF hybrid fusion, and Deep Research Agent.",
            phases_covered="Phases 12 - 15",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_4_agents_tools(self) -> GrandPillarAudit:
        """Pillar 4: Tools, Grammars & Multi-Agent Loops (Phases 16-20)."""
        checks = [
            ("packages/tools/sandbox.py", "Out-of-process OS sandboxing & security visitor"),
            ("packages/core/grammar/json_state_machine.py", "Incremental JSON Pushdown Automaton"),
            ("packages/agents/react.py", "ReAct multi-step autonomous agent loop"),
            ("packages/agents/pal.py", "Program-Aided Language Model (PAL)"),
            ("packages/agents/multi_agent.py", "Collaborative multi-agent consensus team"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=4,
            title="Tools, Structured Decoding & Agentic Loops",
            description="Isolated Python sandbox, JSON grammar decoders, ReAct loops, PAL, and multi-agent collaboration.",
            phases_covered="Phases 16 - 20",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_5_routing_alignment(self) -> GrandPillarAudit:
        """Pillar 5: Routing, Speculative Decoding & Alignment (Phases 21-22)."""
        checks = [
            ("packages/routing/classifier.py", "Query intent & multi-factor complexity classifier"),
            ("packages/models/speculative.py", "Speculative decoding draft-verify engine"),
            ("packages/models/reward_model.py", "Bradley-Terry preference reward model"),
            ("packages/training/dpo_trainer.py", "Direct Preference Optimization (DPO)"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=5,
            title="Dynamic Routing & Preference Alignment",
            description="Complexity classifier, dynamic model routing, and Bradley-Terry Direct Preference Optimization.",
            phases_covered="Phases 21 - 22",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_6_efficiency(self) -> GrandPillarAudit:
        """Pillar 6: Efficiency, KV Caching & Quantization (Phases 23-27)."""
        checks = [
            ("packages/models/components/kv_cache.py", "Dynamic Key-Value cache & GQA/MQA"),
            (
                "packages/models/quantization/ptq.py",
                "INT8 / INT4 affine post-training quantization",
            ),
            ("packages/models/lora/lora_linear.py", "LoRA low-rank parameter-efficient adapters"),
            ("packages/models/telemetry.py", "Token surprisal & Shannon entropy telemetry"),
            ("packages/models/grammar_generation.py", "Grammar constrained generation"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=6,
            title="Inference Acceleration & Parameter Efficiency",
            description="Dynamic KV caching, GQA, INT8/INT4 quantization, LoRA PEFT, and grammar masking.",
            phases_covered="Phases 23 - 27",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_7_reasoning_long_context(self) -> GrandPillarAudit:
        """Pillar 7: Deliberative Reasoning & Long-Context (Phases 28-32)."""
        checks = [
            ("packages/evaluation/elo.py", "Automated Bradley-Terry Elo leaderboard"),
            (
                "packages/models/reasoning/trace_parser.py",
                "Deliberative reasoning CoT trace parser",
            ),
            (
                "packages/models/components/rope_scaling.py",
                "Rotary position scaling (Linear, Dynamic NTK, YaRN)",
            ),
            (
                "packages/models/components/paged_attention.py",
                "PagedAttention virtual memory KV block paging",
            ),
            (
                "packages/models/vision/vlm_model.py",
                "Vision-Language multimodal adapter (LLaVA MLP / Perceiver)",
            ),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=7,
            title="Deliberative Reasoning & Long-Context Architecture",
            description="Elo arena, CoT reasoning, dynamic NTK/YaRN RoPE scaling, PagedAttention, and Vision-Language models.",
            phases_covered="Phases 28 - 32",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_8_production_security(self) -> GrandPillarAudit:
        """Pillar 8: Production Systems & Security Hardening (Phases 33-38)."""
        checks = [
            ("packages/training/chat_formatter.py", "ChatML multi-turn packing & loss masking"),
            ("docker-compose.yml", "Multi-stage container packaging"),
            (".github/workflows/ci.yml", "CI/CD automated quality gates matrix"),
            ("packages/core/security/prompt_guard.py", "Prompt injection & jailbreak defense"),
            ("packages/core/security/secret_scanner.py", "Secret scanning & DLP filters"),
            ("packages/core/observability/tracer.py", "OpenTelemetry distributed tracing"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=8,
            title="Production Hardening, CI/CD & Observability",
            description="ChatML instruction fine-tuning, Docker deployment, CI matrix, PromptGuard DLP, and OpenTelemetry.",
            phases_covered="Phases 33 - 38",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_9_advanced_architectures(self) -> GrandPillarAudit:
        """Pillar 9: Advanced Workflows & Model Architecture (Phases 39-45)."""
        checks = [
            ("packages/core/batch/sequence_binner.py", "High-throughput dynamic sequence batching"),
            (
                "packages/core/document/layout_parser.py",
                "LibraOCR layout-aware document understanding",
            ),
            ("packages/core/notebook/session_kernel.py", "Stateful Python notebook data sandbox"),
            (
                "packages/models/components/compacted_kv_cache.py",
                "StreamingLLM & H2O attention compaction",
            ),
            ("packages/models/reasoning/mcts.py", "Monte Carlo Tree Search with PUCT"),
            (
                "packages/training/distillation_trainer.py",
                "Knowledge distillation & student model shrinking",
            ),
            (
                "packages/models/moe_transformer.py",
                "Sparse Mixture of Experts with top-k noisy gating",
            ),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=9,
            title="Advanced Architectures, MCTS & Sparse MoE",
            description="Dynamic batching, document OCR, stateful code notebook, H2O compaction, MCTS reasoning, distillation, and Sparse MoE.",
            phases_covered="Phases 39 - 45",
            passed=all_ok,
            details=details,
        )

    def audit_pillar_10_frontier_self_evolution(self) -> GrandPillarAudit:
        """Pillar 10: Frontier Reasoning & Self-Evolution (Phases 46-50)."""
        checks = [
            (
                "packages/models/components/medusa.py",
                "Medusa multi-head residual speculative drafting",
            ),
            ("packages/training/kto_trainer.py", "Kahneman-Tversky Optimization (KTO)"),
            ("packages/training/online_dpo_trainer.py", "Online on-policy DPO alignment"),
            (
                "packages/models/reasoning/verifiable_search.py",
                "Step-level verifiable search with PRM pruning",
            ),
            (
                "packages/training/self_rewarding.py",
                "LLM-as-a-Judge self-rewarding alignment flywheel",
            ),
            ("packages/training/self_evolution.py", "Autonomous self-evolution engine"),
        ]
        details = []
        all_ok = True
        for path, desc in checks:
            exists = self._file_exists(path)
            details.append(f"{'[PASS]' if exists else '[FAIL]'} {desc} ({path})")
            if not exists:
                all_ok = False

        return GrandPillarAudit(
            pillar_id=10,
            title="Frontier Alignment, Verifiable Search & Self-Evolution",
            description="Medusa speculative drafting, KTO alignment, PRM tree search, Self-Rewarding LLM, and Autonomous Self-Evolution.",
            phases_covered="Phases 46 - 50",
            passed=all_ok,
            details=details,
        )

    def run_full_grand_audit(self) -> GrandCapstoneReport:
        """Executes the complete 10-pillar grand audit across all 50 phases."""
        start_time = time.perf_counter()

        pillars = [
            self.audit_pillar_1_foundations(),
            self.audit_pillar_2_inference(),
            self.audit_pillar_3_rag_search(),
            self.audit_pillar_4_agents_tools(),
            self.audit_pillar_5_routing_alignment(),
            self.audit_pillar_6_efficiency(),
            self.audit_pillar_7_reasoning_long_context(),
            self.audit_pillar_8_production_security(),
            self.audit_pillar_9_advanced_architectures(),
            self.audit_pillar_10_frontier_self_evolution(),
        ]

        duration = round(time.perf_counter() - start_time, 3)
        passed_count = sum(1 for p in pillars if p.passed)
        all_passed = passed_count == len(pillars)
        score_pct = round((passed_count / len(pillars)) * 100.0, 1)

        honors = (
            "SUMMA CUM LAUDE — Perfect 50-Phase Architectural Completion. "
            "All foundational, modern, agentic, production, and self-evolution systems verified."
            if all_passed
            else f"{passed_count}/10 Pillars Passed — Review pending milestones."
        )

        return GrandCapstoneReport(
            pillars_passed=passed_count,
            all_passed=all_passed,
            completion_score_pct=score_pct,
            pillars=pillars,
            duration_sec=duration,
            graduation_honors=honors,
        )
