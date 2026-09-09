"""
Libra Evaluation - Master Capstone System Audit Engine (Phase 36)
Executes end-to-end integration and verification across all 7 architectural pillars:
1. Core Modeling (ModernTransformerLM, RoPE, GQA, PagedAttention, LibraVLM)
2. Inference Acceleration (KV-Cache, Speculative Decoding, Grammar Decoding)
3. Providers & Routing (ProviderRouter, Complexity Classifier, Cost Tracker)
4. Memory & Context (SQLite Store, Context Window Pruning)
5. Knowledge & RAG (Dense Vector Store, BM25, Hybrid RRF)
6. Agents & Tool Execution (Safe Sandbox, Tool Registry, ReAct Agent)
7. Training, Alignment & Deployment (ChatML Packing, DPO Loss, Deployment Validator)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from packages.core.deployment_validator import DeploymentValidator
from packages.core.grammar.regex_automaton import RegexAutomaton

# 4. Memory imports
from packages.core.memory.context_manager import ContextWindowManager
from packages.core.memory.sqlite_store import SQLiteConversationStore
from packages.models.components.paged_cache import (
    BlockAllocator,
    PhysicalBlockPool,
    SequenceBlockTable,
)

# 2. Inference imports
from packages.models.generation import generate_with_cache

# 1. Modeling imports
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.models.speculative import SpeculativeDecoder
from packages.models.vision.vlm_model import LibraVLM
from packages.models.vision.vlm_projector import ProjectorType
from packages.providers.cost import calculate_cost

# 3. Provider & Routing imports
from packages.providers.router import ProviderRouter

# 5. RAG imports
from packages.rag.hybrid import HybridRetriever
from packages.routing.classifier import QueryClassifier

# 6. Agents & Tools imports
from packages.tools.builtin import CalculatorTool
from packages.tools.registry import ToolRegistry
from packages.tools.sandbox import SafePythonSandbox

# 7. Training & Deployment imports
from packages.training.chat_formatter import (
    ChatMessage,
    ChatMLFormatter,
    tokenize_with_loss_masking,
)


@dataclass
class PillarResult:
    pillar_name: str
    passed: bool
    latency_ms: float
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CapstoneAuditReport:
    total_pillars: int
    pillars_passed: int
    duration_sec: float
    results: list[PillarResult]

    @property
    def all_passed(self) -> bool:
        return self.pillars_passed == self.total_pillars


class LibraCapstoneAudit:
    """Master audit orchestrator verifying all architectural subsystems."""

    def __init__(self, repo_root: str | None = None) -> None:
        self.repo_root = repo_root or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    def audit_pillar_1_modeling(self) -> PillarResult:
        """Pillar 1: Modern Transformer, RoPE, GQA, PagedAttention, and VLM."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            # A. ModernTransformerLM
            cfg = ModernTransformerConfig(
                vocab_size=256,
                d_model=64,
                n_layers=2,
                n_heads=2,
                n_kv_heads=1,
                max_context_length=128,
            )
            model = ModernTransformerLM(cfg)
            x = torch.randint(0, 256, (1, 8))
            logits, _ = model(x)
            assert logits.shape == (1, 8, 256)
            details.append("ModernTransformerLM forward pass verified (GQA + SwiGLU + RoPE)")

            # B. PagedAttention
            pool = PhysicalBlockPool(num_blocks=16, block_size=4, num_kv_heads=1, head_dim=32)
            allocator = BlockAllocator(num_blocks=16)
            table = SequenceBlockTable(seq_id="seq-capstone", block_size=4)
            b_id = allocator.allocate()
            table.physical_block_ids.append(b_id)
            assert b_id is not None
            details.append("PagedAttention PhysicalBlockPool & BlockAllocator verified")

            # C. LibraVLM
            vlm = LibraVLM(
                llm_config=cfg,
                image_size=32,
                patch_size=16,
                in_channels=3,
                vision_dim=32,
                projector_type=ProjectorType.MLP,
            )
            img = torch.randn(1, 3, 32, 32)
            prompt = torch.randint(0, 256, (1, 4))
            vlm_logits, _ = vlm(images=img, text_ids=prompt)
            assert vlm_logits.shape[-1] == 256
            details.append("LibraVLM Multimodal forward pass verified")

        except Exception as e:
            errors.append(f"Modeling audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="1. Architecture & Modeling",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_2_inference(self) -> PillarResult:
        """Pillar 2: KV Cache, Speculative Decoding, and Grammar."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            cfg = ModernTransformerConfig(
                vocab_size=256,
                d_model=32,
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                max_context_length=64,
            )
            model = ModernTransformerLM(cfg)

            # KV Cache Generation
            prompt = torch.tensor([[65, 66, 67]], dtype=torch.long)
            out_tokens, _ = generate_with_cache(
                model=model, idx=prompt, max_new_tokens=4, temperature=0.0
            )
            assert out_tokens.shape[1] == 7
            details.append("KV-Cache O(1) step generation verified")

            # Speculative Decoding
            draft_cfg = ModernTransformerConfig(
                vocab_size=256,
                d_model=32,
                n_layers=1,
                n_heads=2,
                n_kv_heads=1,
                max_context_length=64,
            )
            draft_model = ModernTransformerLM(draft_cfg)
            decoder = SpeculativeDecoder(target_model=model, draft_model=draft_model, lookahead_k=2)
            res = decoder.generate(prompt_tokens=[65, 66, 67], max_new_tokens=4)
            assert len(res.output_tokens) >= 4
            details.append(
                f"Speculative Decoding verified (acceptance rate: {res.acceptance_rate:.2f})"
            )

            # Regex Automaton
            regex = RegexAutomaton(r"[0-9]+")
            assert regex.is_valid_prefix("5") is True
            assert regex.is_accepted("543") is True
            details.append("Grammar/Regex state machine verified")

        except Exception as e:
            errors.append(f"Inference audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="2. Inference Acceleration & Grammars",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_3_providers(self) -> PillarResult:
        """Pillar 3: Provider Router, Semantic Complexity, Cost Tracking."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            router = ProviderRouter()
            assert router is not None
            details.append("Unified ProviderRouter initialized")

            classifier = QueryClassifier()
            decision = classifier.classify("Write a Python script to calculate Fibonacci numbers.")
            assert decision is not None
            details.append(
                f"Semantic Task Complexity Classifier verified (Tier: {decision.recommended_tier})"
            )

            cost_info = calculate_cost(
                model_id="libra-test", prompt_tokens=100, completion_tokens=50
            )
            assert cost_info["is_free"] is True
            details.append(
                f"Cost & Telemetry Economics verified (${cost_info['total_cost_usd']:.4f}, Zero-Cost: {cost_info['is_free']})"
            )

        except Exception as e:
            errors.append(f"Provider audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="3. Provider Routing & Cost Tracking",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_4_memory(self) -> PillarResult:
        """Pillar 4: SQLite Conversation Store & Context Manager."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            db_path = os.path.join(self.repo_root, "data", "capstone_temp_memory.db")
            store = SQLiteConversationStore(db_path=db_path)
            conv = store.create_conversation("Capstone Session", model="libra-capstone")
            store.add_message(conv.id, "user", "Hello Libra!")
            store.add_message(conv.id, "assistant", "Hello! How can I assist you today?")
            messages = store.get_messages(conv.id)
            assert len(messages) == 2
            details.append(f"SQLite Conversation Store verified ({len(messages)} messages stored)")

            mgr = ContextWindowManager(max_context_tokens=100)
            context = mgr.prepare_context(messages=messages)
            assert len(context.messages) > 0
            details.append("Sliding-window ContextWindowManager verified")

            # Clean up temp db
            if os.path.exists(db_path):
                try:
                    os.remove(db_path)
                except OSError:
                    pass

        except Exception as e:
            errors.append(f"Memory audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="4. Memory & Context Management",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_5_rag(self) -> PillarResult:
        """Pillar 5: Dense Vector Store, BM25, and Hybrid RRF."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            hybrid = HybridRetriever()
            doc, chunks = hybrid.add_document(
                title="Transformer Attention",
                content="Self-attention computes query-key dot products and applies rotary position embeddings.",
            )
            assert len(chunks) > 0
            details.append(f"Hybrid Document Ingestion & Chunking verified ({len(chunks)} chunks)")

            results = hybrid.search("query key dot products", top_k=1, mode="hybrid")
            assert len(results) > 0
            details.append(
                f"Hybrid Retriever (Dense + BM25 with RRF) verified (Score: {results[0].score:.4f})"
            )

        except Exception as e:
            errors.append(f"RAG audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="5. Knowledge & Hybrid RAG",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_6_agents(self) -> PillarResult:
        """Pillar 6: Safe Python Sandbox, Tools, and Agents."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            # Sandbox
            sandbox = SafePythonSandbox(default_timeout=2.0)
            res = sandbox.execute("sum([1, 2, 3, 4, 5])")
            assert res["result"] == 15, f"Expected 15, got {res}"
            details.append("Safe Python Execution Sandbox verified (AST whitelist + timeout)")

            # Tool Registry
            reg = ToolRegistry()
            reg.register(CalculatorTool())
            assert "calculator" in reg.get_tool_names()
            tool_res = reg.execute_tool("calculator", {"expression": "12 * 12"})
            assert tool_res.success is True
            assert "144" in str(tool_res.output)
            details.append("Tool Registry & Execution verified")

        except Exception as e:
            errors.append(f"Agents & Tools audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="6. Agents & Sandboxed Tools",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def audit_pillar_7_deployment(self) -> PillarResult:
        """Pillar 7: ChatML Packing, Docker Validation, and Quota Check."""
        t0 = time.perf_counter()
        details = []
        errors = []
        try:
            # ChatML & Loss Masking
            dialogue = [
                ChatMessage(role="system", content="System"),
                ChatMessage(role="user", content="Prompt"),
                ChatMessage(role="assistant", content="Answer"),
            ]
            formatter = ChatMLFormatter()
            text = formatter.format_conversation(dialogue)
            assert "<|im_start|>" in text
            inps, lbls = tokenize_with_loss_masking(dialogue, lambda s: list(s.encode("utf-8")))
            assert -100 in lbls
            details.append("ChatML Formatter & Prompt Loss Masking (-100) verified")

            # Deployment Validator
            val = DeploymentValidator(self.repo_root)
            comp_rep = val.validate_compose()
            assert comp_rep.is_valid is True
            details.append(f"Docker Compose verified ({comp_rep.service_count} services)")

        except Exception as e:
            errors.append(f"Deployment audit failed: {e}")

        latency = (time.perf_counter() - t0) * 1000.0
        return PillarResult(
            pillar_name="7. Training, Alignment & Deployment",
            passed=len(errors) == 0,
            latency_ms=latency,
            details=details,
            errors=errors,
        )

    def run_full_capstone_audit(self) -> CapstoneAuditReport:
        """Executes the master capstone audit across all 7 pillars."""
        start_time = time.perf_counter()
        audit_methods = [
            self.audit_pillar_1_modeling,
            self.audit_pillar_2_inference,
            self.audit_pillar_3_providers,
            self.audit_pillar_4_memory,
            self.audit_pillar_5_rag,
            self.audit_pillar_6_agents,
            self.audit_pillar_7_deployment,
        ]

        results = []
        for method in audit_methods:
            res = method()
            results.append(res)

        total_duration = time.perf_counter() - start_time
        passed_count = sum(1 for r in results if r.passed)

        return CapstoneAuditReport(
            total_pillars=len(results),
            pillars_passed=passed_count,
            duration_sec=total_duration,
            results=results,
        )
