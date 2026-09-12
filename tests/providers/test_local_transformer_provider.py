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


@pytest.mark.asyncio
async def test_local_transformer_output_has_no_control_characters():
    """Inference must use the same raw-byte encoding as the training pipeline.

    Previously the prompt was encoded with EducationalBPETokenizer (byte ids
    offset by +4) while training used raw byte ids 0..255, so generated text
    decoded into control-char paddling (e.g. '\\x1c'). Output must be plain text.
    """
    import os

    provider = LocalTransformerProvider()
    if not os.path.exists(provider.checkpoint_path):
        pytest.skip(
            f"Trained checkpoint {provider.checkpoint_path} not present (CI runner environment)"
        )

    response = await provider.chat(
        messages=[{"role": "user", "content": "Gravity is a force"}],
        model="libra-llama-tied",
        max_tokens=20,
        temperature=0.8,
    )
    content = response["choices"][0]["message"]["content"]
    assert content, "expected non-empty local model output"
    control_chars = [c for c in content if ord(c) < 32 and c not in "\n\r\t"]
    assert not control_chars, f"garbage control chars in output: {content!r}"
