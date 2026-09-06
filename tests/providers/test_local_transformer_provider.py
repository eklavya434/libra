import pytest
from packages.providers.local_transformer import LocalTransformerProvider


@pytest.mark.asyncio
async def test_local_transformer_provider_chat():
    provider = LocalTransformerProvider()
    health = await provider.health()
    assert health["provider"] == "libra_lab"
    assert "status" in health

    models = await provider.list_models()
    assert len(models) == 1
    assert models[0].id == "libra-llama-tied"

    # Chat execution test
    response = await provider.chat(
        messages=[{"role": "user", "content": "Gravity is"}],
        model="libra-llama-tied",
        max_tokens=8,
    )

    assert "choices" in response
    assert len(response["choices"]) > 0
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert response["usage"]["completion_tokens"] > 0
