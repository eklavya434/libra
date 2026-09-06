"""
Tests for StructuredOutputGenerator and self-healing repair loop (packages/providers/structured.py)
"""

import pytest
from pydantic import BaseModel, Field
from packages.providers.base import BaseProvider, ModelMetadata
from packages.providers.router import ProviderRouter
from packages.providers.structured import (
    StructuredOutputGenerator,
    extract_json_from_text,
)


class CityStats(BaseModel):
    city: str = Field(..., description="City name")
    population: int = Field(..., description="Population count")


def test_extract_json_from_markdown():
    markdown_text = """
Here is the city information requested:
```json
{
    "city": "Tokyo",
    "population": 14000000
}
```
Hope this helps!
"""
    extracted = extract_json_from_text(markdown_text)
    assert '"city": "Tokyo"' in extracted
    assert "```" not in extracted


class FlakyMockProvider(BaseProvider):
    """Mock provider that intentionally fails on attempt 1 with bad schema, then fixes on attempt 2."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "flaky"

    async def list_models(self) -> list[ModelMetadata]:
        return []

    async def chat(self, messages, model=None, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            # Missing required field 'population'
            return {
                "choices": [{"message": {"content": '{"city": "Paris"}'}}]
            }
        else:
            # Valid response
            return {
                "choices": [
                    {"message": {"content": '{"city": "Paris", "population": 2161000}'}}
                ]
            }

    async def stream(self, messages, model=None, **kwargs):
        yield ""

    async def health(self):
        return {"status": "healthy"}

    async def embeddings(self, texts, model=None):
        return [[0.0] * 16]

    def capabilities(self):
        return {"supports_text": True}


@pytest.mark.asyncio
async def test_self_healing_repair_loop():
    flaky_provider = FlakyMockProvider()
    router = ProviderRouter()
    router._providers["flaky"] = flaky_provider

    generator = StructuredOutputGenerator(router=router)

    result = await generator.generate(
        prompt="Tell me about Paris",
        schema=CityStats,
        model_id="flaky",
        provider_name="flaky",
        max_retries=2,
    )

    assert result.success is True
    assert result.attempts == 2
    assert isinstance(result.data, CityStats)
    assert result.data.city == "Paris"
    assert result.data.population == 2161000
    assert result.error is None
