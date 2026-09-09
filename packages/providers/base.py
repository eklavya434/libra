"""
Libra Providers - Unified Model Provider Abstraction
Decouples application logic from specific model runtimes (Ollama, HF, OpenAI, Gemini, etc.).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelMetadata:
    id: str
    name: str
    provider: str
    architecture: str
    context_length: int
    parameter_count: str | None = None
    license: str = "Unknown"
    is_local: bool = False
    requires_gpu: bool = False
    hardware_tier: str = "cpu-light"  # cpu-light, cpu-medium, cpu-large, gpu-required
    quantization: str = "FP16"
    capabilities: dict[str, bool] = field(
        default_factory=lambda: {
            "supports_text": True,
            "supports_vision": False,
            "supports_tools": False,
            "supports_reasoning": False,
            "supports_embeddings": False,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "parameter_count": self.parameter_count,
            "license": self.license,
            "is_local": self.is_local,
            "requires_gpu": self.requires_gpu,
            "hardware_tier": self.hardware_tier,
            "quantization": self.quantization,
            "capabilities": self.capabilities,
        }


class BaseProvider(ABC):
    """Abstract Base Class defining the contract for any LLM Provider in Libra."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique identifier for this provider."""

    @abstractmethod
    async def list_models(self) -> list[ModelMetadata]:
        """Lists all models offered by this provider."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sends a non-streaming chat completion request."""

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streams response tokens asynchronously."""

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Checks provider reachability and credentials."""

    @abstractmethod
    async def embeddings(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generates vector embeddings for the provided list of texts."""

    @abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Declares overarching capabilities supported by the provider adapter."""
