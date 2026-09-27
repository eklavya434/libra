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
from packages.models.generation import decode_tokens, encode_string, generate
from packages.models.modern_config import ModernTransformerConfig
from packages.models.modern_transformer import ModernTransformerLM
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.errors import ProviderOfflineError
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

    def _ensure_loaded(self) -> tuple[ModernTransformerLM, Optional[EducationalBPETokenizer]]:
        """Lazy loader for model weights and tokenizer.

        Raises ``ProviderOfflineError`` (never a silent random-weight output) when
        the trained checkpoint required for real inference is absent.
        """
        if self._model is not None:
            return self._model, self._tokenizer

        if not os.path.exists(self.checkpoint_path):
            raise ProviderOfflineError(
                message=(
                    f"No trained Libra checkpoints found at '{self.checkpoint_path}'. "
                    "This model is only available on a machine where the educational "
                    "training pipeline has been run. Run a Phase 4/5 training job, or "
                    "select a cloud/Ollama/mock model instead."
                ),
                provider="libra_lab",
                status_code=503,
            )

        # 1. Load Tokenizer (optional: only used to surface BPE metadata)
        if os.path.exists(self.tokenizer_path):
            self._tokenizer = EducationalBPETokenizer.load(self.tokenizer_path)
        else:
            self._tokenizer = None

        # 2. Load Model Architecture (a matching config is required for real weights)
        if not os.path.exists(self.config_path):
            raise ProviderOfflineError(
                message=(
                    f"No architecture config found at '{self.config_path}'. The trained "
                    "checkpoint cannot be used without the config that produced it."
                ),
                provider="libra_lab",
                status_code=503,
            )
        config = ModernTransformerConfig.from_yaml(self.config_path)
        model = ModernTransformerLM(config).to(self.device)

        # 3. Load Checkpoint Weights
        try:
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
        except RuntimeError as e:
            raise ProviderOfflineError(
                message=(
                    f"Checkpoint '{self.checkpoint_path}' is incompatible with the "
                    f"configured architecture: {e}"
                ),
                provider="libra_lab",
                status_code=503,
            ) from e

        model.eval()
        self._model = model
        return self._model, self._tokenizer

    def is_ready(self) -> bool:
        """True only when a real trained checkpoint exists on disk."""
        return os.path.exists(self.checkpoint_path) and os.path.exists(self.config_path)

    async def health(self) -> dict[str, Any]:
        ready = self.is_ready()
        return {
            "status": "online" if ready else "offline",
            "provider": "libra_lab",
            "checkpoint_loaded": ready,
            "checkpoint_path": self.checkpoint_path,
            "device": str(self.device),
            "detail": (
                None
                if ready
                else "No trained checkpoint. Run the training pipeline or configure a cloud/Ollama provider."
            ),
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
        model_obj, _ = self._ensure_loaded()
        prompt_text = PromptTemplate.format_plain(messages)
        # The training pipeline (TextDataset) encodes raw UTF-8 bytes as ids 0..255
        # (no BPE). Encode the prompt the same way so inference ids match training.
        prompt_ids = encode_string(prompt_text)

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids = generate(
            model=model_obj,
            idx=input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        new_tokens = [t % 256 for t in generated_ids[0, len(prompt_ids) :].tolist()]
        output_text = decode_tokens(new_tokens)

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
        raise NotImplementedError(
            "LocalTransformerProvider does not implement embeddings; "
            "use an embedding-capable provider (e.g. OpenAI/Gemini) for RAG."
        )
