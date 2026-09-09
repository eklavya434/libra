from fastapi.testclient import TestClient

from apps.backend.main import app
from packages.core.hardware import HardwareReport
from packages.models.catalog import get_default_registry
from packages.models.registry import (
    HardwareClassifier,
    HardwareTier,
)

client = TestClient(app)


def test_registry_catalog_completeness():
    registry = get_default_registry()
    assert registry.count() >= 10

    # Ensure key required model families are present
    model_ids = {m.model_id for m in registry.list_models()}
    assert "llama3.2:1b" in model_ids
    assert "llama3.2:3b" in model_ids
    assert "deepseek-r1:1.5b" in model_ids
    assert "qwen2.5:1.5b" in model_ids
    assert "gemma2:2b" in model_ids
    assert "mistral:7b-instruct-v0.3" in model_ids
    assert "phi3.5:3.8b" in model_ids
    assert "nemotron-mini:4b" in model_ids
    assert "libra-llama-tied" in model_ids


def test_hardware_tier_classification():
    # Mock system specs: 16 GB RAM CPU without CUDA
    sys_info = HardwareReport(
        os="Windows",
        os_version="10.0",
        architecture="x86_64",
        cpu_model="Intel Core i5-12450H",
        cpu_physical_cores=8,
        cpu_logical_cores=12,
        ram_total_gb=16.0,
        ram_available_gb=8.0,
        disk_total_gb=477.0,
        disk_free_gb=74.0,
        has_cuda=False,
        gpu_name=None,
        device_tier="cpu-only",
    )

    # 1.5B Q4 Model -> CPU-Friendly
    tier_1b, ram_1b, can_run_1b, verdict_1b = HardwareClassifier.classify(
        param_count=1_500_000_000,
        quantization="Q4_K_M",
        system_info=sys_info,
    )
    assert tier_1b == HardwareTier.CPU_FRIENDLY
    assert ram_1b < 2.0  # ~1.0 GB RAM
    assert can_run_1b is True
    assert "smoothly on CPU" in verdict_1b

    # 72B Q4 Model -> Very Large, Cannot run on 16GB
    tier_72b, ram_72b, can_run_72b, verdict_72b = HardwareClassifier.classify(
        param_count=72_000_000_000,
        quantization="Q4_K_M",
        system_info=sys_info,
    )
    assert tier_72b == HardwareTier.VERY_LARGE
    assert ram_72b > 40.0  # > 40 GB RAM
    assert can_run_72b is False
    assert "Exceeds system RAM" in verdict_72b


def test_registry_filtering():
    registry = get_default_registry()

    # Filter by provider
    ollama_models = registry.list_models(provider="ollama")
    assert len(ollama_models) >= 8
    assert all(m.provider == "ollama" for m in ollama_models)

    # Filter CPU friendly
    cpu_friendly = registry.get_cpu_friendly_models()
    assert len(cpu_friendly) >= 5
    assert all(m.can_run_locally for m in cpu_friendly)
    assert all(m.hardware_tier == HardwareTier.CPU_FRIENDLY for m in cpu_friendly)


def test_api_models_endpoint_extended():
    # Base list
    resp = client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 10

    # Filter by cpu_friendly_only
    resp_cpu = client.get("/api/v1/models?cpu_friendly_only=true")
    assert resp_cpu.status_code == 200
    cpu_data = resp_cpu.json()
    assert cpu_data["count"] < data["count"]
    for m in cpu_data["models"]:
        assert m["hardware_tier"] == "CPU-friendly"
        assert m["can_run_locally"] is True

    # Get specific model
    resp_single = client.get("/api/v1/models/llama3.2:1b")
    assert resp_single.status_code == 200
    m_data = resp_single.json()
    assert m_data["model_id"] == "llama3.2:1b"
    assert m_data["organization"] == "Meta"
    assert m_data["license_type"] == "open_weights_community"

    # Non-existent model returns 404
    resp_404 = client.get("/api/v1/models/non-existent-model-xyz")
    assert resp_404.status_code == 404
