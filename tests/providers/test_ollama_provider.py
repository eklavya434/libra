import pytest
from packages.providers.ollama import OllamaProvider


def test_ollama_options_builder():
    provider = OllamaProvider(base_url="http://127.0.0.1:11434")
    opts = provider._build_options(
        temperature=0.4,
        max_tokens=256,
        top_p=0.85,
        top_k=50,
        stop=["<stop>", "\nEnd"],
        seed=42,
    )

    assert opts["temperature"] == 0.4
    assert opts["num_predict"] == 256
    assert opts["top_p"] == 0.85
    assert opts["top_k"] == 50
    assert opts["stop"] == ["<stop>", "\nEnd"]
    assert opts["seed"] == 42


@pytest.mark.asyncio
async def test_ollama_offline_health_check():
    # Points to a non-existent port to test offline detection
    provider = OllamaProvider(base_url="http://127.0.0.1:59999")
    report = await provider.health()

    assert report["status"] == "offline"
    assert report["connected"] is False
    assert "guidance" in report


@pytest.mark.asyncio
async def test_ollama_offline_chat_raises_connection_error():
    provider = OllamaProvider(base_url="http://127.0.0.1:59999")
    with pytest.raises(ConnectionError):
        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            model="llama3.2:1b",
        )
