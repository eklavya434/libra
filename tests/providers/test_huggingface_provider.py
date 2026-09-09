import pytest

from packages.providers.huggingface import HuggingFaceProvider


@pytest.mark.asyncio
async def test_huggingface_provider_health_and_capabilities():
    provider = HuggingFaceProvider()
    health = await provider.health()
    assert health["provider"] == "huggingface"
    assert health["status"] == "online"
    assert "transformers_version" in health

    caps = provider.capabilities()
    assert caps["supports_text"] is True
    assert caps["supports_streaming"] is True

    models = await provider.list_models()
    assert len(models) >= 1
    assert models[0].provider == "huggingface"
