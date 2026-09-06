"""
Libra Providers - Local Educational Transformer Provider

Enables chatting with the neural networks built in our LLM Laboratory
directly through the unified BaseProvider interface!
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from typing import Any, Optional

import torch

from packages.core.tokenizer.educational_bpe import EducationalBPETokenizer
from packages.models.generation import generate
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.prompt_template import PromptTemplate


class LocalTransformerProvider(BaseProvider):
    """Serves locally trained PyTorch transformer checkpoints for inference."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_engine_model.pt",
        config_path: str = "configs/models/tiny_modern_tied.yaml",
        tokenizer_path: str = "data/tokenized/libra_educational_bpe.json",
        device: str = "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device(device)
        self._model: Optional[ModernTransformerLM] = None
        self._tokenizer: Optional[EducationalBPETokenizer] = None

    @property
    def name(self) -> str:
        return "libra_lab"

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_text": True,
            "supports_vision": False,
            "supports_tools": False,
            "supports_reasoning": False,
            "supports_embeddings": False,
            "supports_streaming": True,
        }

    def _ensure_loaded(self) -> tuple[ModernTransformerLM, EducationalBPETokenizer]:
        """Lazy loader for model weights and tokenizer."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        # 1. Load Tokenizer
        if os.path.exists(self.tokenizer_path):
            self._tokenizer = EducationalBPETokenizer.load(self.tokenizer_path)
        else:
            # Fallback training on default corpus
            corpus_path = "data/raw/educational_science_corpus.txt"
            if os.path.exists(corpus_path):
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus = f.read()
            else:
                corpus = "Project Libra educational transformer."
            tok = EducationalBPETokenizer()
            tok.train(corpus, num_merges=40)
            self._tokenizer = tok

        # 2. Load Model Architecture
        if os.path.exists(self.config_path):
            config = ModernTransformerConfig.from_yaml(self.config_path)
        else:
            config = ModernTransformerConfig(vocab_size=512, d_model=128, n_heads=4, n_layers=2)

        model = ModernTransformerLM(config).to(self.device)

        # 3. Load Checkpoint Weights
        if os.path.exists(self.checkpoint_path):
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

        model.eval()
        self._model = model
        return self._model, self._tokenizer

    async def health(self) -> dict[str, Any]:
        has_checkpoint = os.path.exists(self.checkpoint_path)
        return {
            "status": "online",
            "provider": "libra_lab",
            "checkpoint_loaded": has_checkpoint,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
        }

    async def list_models(self) -> list[ModelMetadata]:
        return [
            ModelMetadata(
                id="libra-llama-tied",
                name="Libra Modern Llama (Local Lab)",
                provider="libra_lab",
                architecture="Decoder-only RoPE + SwiGLU",
                context_length=256,
                parameter_count="468K",
                quantization="FP32",
                is_local=True,
                requires_gpu=False,
                hardware_tier="cpu-friendly",
                capabilities=self.capabilities(),
            )
        ]

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "libra-llama-tied",
        temperature: float = 0.7,
        max_tokens: int = 64,
        top_k: int = 40,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_obj, tokenizer = self._ensure_loaded()
        prompt_text = PromptTemplate.format_plain(messages)
        prompt_ids = tokenizer.encode(prompt_text)

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids = generate(
            model=model_obj,
            idx=input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        new_tokens = generated_ids[0, len(prompt_ids) :].tolist()
        output_text = tokenizer.decode(new_tokens)

        # Stop sequences cleanup
        for stop_seq in PromptTemplate.get_stop_sequences("plain"):
            if stop_seq in output_text:
                output_text = output_text.split(stop_seq)[0]

        return {
            "id": f"libra-lab-{int(time.time())}",
            "provider": "libra_lab",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text.strip()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(new_tokens),
                "total_tokens": len(prompt_ids) + len(new_tokens),
            },
        }

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "libra-llama-tied",
        temperature: float = 0.7,
        max_tokens: int = 64,
        top_k: int = 40,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        # Full autoregressive generation then simulated streaming token-by-token
        result = await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            **kwargs,
        )
        content = result["choices"][0]["message"]["content"]
        words = content.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == 0 else " " + word
            yield chunk

    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        # Dummy or placeholder embeddings for local model
        return [[0.0] * 128 for _ in texts]
