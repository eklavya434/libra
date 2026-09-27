"""Integrity guarantees for the local educational transformer provider (ISSUE_MATRIX #1).

The provider must never fabricate a "successful" response from random weights.
Inference is only legal when a real trained checkpoint **and** the architecture
config that produced it both exist on disk; otherwise it must fail loudly with
``ProviderOfflineError`` (HTTP 503) and report ``is_ready() is False``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from packages.providers.errors import ProviderOfflineError
from packages.providers.local_transformer import LocalTransformerProvider

REPO_CONFIG = Path("configs/models/tiny_modern_tied.yaml")


@pytest.mark.asyncio
async def test_missing_checkpoint_never_yields_fabricated_output():
    """No checkpoint -> ProviderOfflineError, not random-weight text."""
    provider = LocalTransformerProvider(
        checkpoint_path="definitely/absent/checkpoint.pt",
        config_path="definitely/absent/config.yaml",
    )

    assert provider.is_ready() is False

    with pytest.raises(ProviderOfflineError) as excinfo:
        await provider.chat(messages=[{"role": "user", "content": "Explain gravity"}])
    error = excinfo.value
    assert error.provider == "libra_lab"
    assert error.status_code == 503
    assert "checkpoint" in error.message.lower()
    # The message must tell the user what to do, not pretend to be a model.
    assert "training" in error.message.lower()


@pytest.mark.asyncio
async def test_missing_config_is_offline_even_with_checkpoint_on_disk(tmp_path):
    """A checkpoint without its architecture config is unusable, so it is offline."""
    checkpoint = tmp_path / "best_engine_model.pt"
    torch.save({"model_state_dict": {}}, checkpoint)

    provider = LocalTransformerProvider(
        checkpoint_path=str(checkpoint),
        config_path=str(tmp_path / "absent_config.yaml"),
    )

    assert provider.is_ready() is False
    health = await provider.health()
    assert health["status"] == "offline"
    assert health["checkpoint_loaded"] is False

    with pytest.raises(ProviderOfflineError) as excinfo:
        await provider.chat(messages=[{"role": "user", "content": "hi"}])
    assert "config" in excinfo.value.message.lower()


@pytest.mark.asyncio
async def test_incompatible_checkpoint_is_offline_not_a_wrong_architecture(tmp_path):
    """A checkpoint whose keys do not match the config must fail honestly."""
    if not os.path.exists(REPO_CONFIG):
        pytest.skip(f"{REPO_CONFIG} not present in this checkout")

    checkpoint = tmp_path / "best_engine_model.pt"
    torch.save({"model_state_dict": {"not_a_real_layer": torch.zeros(2, 2)}}, checkpoint)

    provider = LocalTransformerProvider(
        checkpoint_path=str(checkpoint),
        config_path=str(REPO_CONFIG),
    )

    with pytest.raises(ProviderOfflineError) as excinfo:
        await provider.chat(messages=[{"role": "user", "content": "hi"}], max_tokens=1)
    assert "incompatible" in excinfo.value.message.lower()


@pytest.mark.asyncio
async def test_health_reports_disk_truth_for_the_default_paths():
    """``health()`` must describe the real filesystem state, not an assumption."""
    provider = LocalTransformerProvider()
    health = await provider.health()

    ready = provider.is_ready()
    assert health["checkpoint_loaded"] is ready
    assert health["status"] == ("online" if ready else "offline")
    assert health["provider"] == "libra_lab"
    if ready:
        assert health["detail"] is None
    else:
        assert health["detail"], "offline health must explain why it is offline"


@pytest.mark.asyncio
async def test_stream_also_refuses_when_offline():
    """Streaming must not be a loophole that bypasses the offline guard."""
    provider = LocalTransformerProvider(
        checkpoint_path="definitely/absent/checkpoint.pt",
        config_path="definitely/absent/config.yaml",
    )

    with pytest.raises(ProviderOfflineError):
        async for _ in provider.stream(messages=[{"role": "user", "content": "hi"}]):
            pass
