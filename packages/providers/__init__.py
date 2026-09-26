import os

from dotenv import load_dotenv

from packages.core.network import enable_ipv4_preference

if os.path.exists("/etc/secrets/.env"):
    load_dotenv("/etc/secrets/.env")
load_dotenv()
enable_ipv4_preference()

from packages.providers.anthropic import AnthropicProvider
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.cost import ModelPricing, calculate_cost, get_model_pricing
from packages.providers.deepseek import DeepSeekProvider
from packages.providers.errors import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderOfflineError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
)
from packages.providers.gemini import GeminiProvider
from packages.providers.generic_openai import GenericOpenAIProvider, OpenRouterProvider
from packages.providers.groq import GroqProvider
from packages.providers.huggingface import HuggingFaceProvider
from packages.providers.kimi import KimiProvider
from packages.providers.local_transformer import LocalTransformerProvider
from packages.providers.mock import MockProvider
from packages.providers.nvidia import NvidiaProvider
from packages.providers.ollama import OllamaProvider
from packages.providers.openai import OpenAIProvider
from packages.providers.prompt_template import PromptTemplate
from packages.providers.router import ProviderRouter, get_router
from packages.providers.structured import StructuredOutputGenerator, StructuredResult
from packages.providers.vllm_stub import VLLMProvider

__all__ = [
    "BaseProvider",
    "ModelMetadata",
    "StructuredOutputGenerator",
    "StructuredResult",
    "MockProvider",
    "NvidiaProvider",
    "OllamaProvider",
    "HuggingFaceProvider",
    "LocalTransformerProvider",
    "VLLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "DeepSeekProvider",
    "GroqProvider",
    "KimiProvider",
    "OpenRouterProvider",
    "GenericOpenAIProvider",
    "PromptTemplate",
    "ProviderRouter",
    "get_router",
    "calculate_cost",
    "get_model_pricing",
    "ModelPricing",
    "ProviderError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderQuotaExceededError",
    "ModelNotFoundError",
    "ProviderOfflineError",
]
