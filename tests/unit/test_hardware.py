"""
Unit tests for Libra Hardware Introspection module.
"""

from packages.core.hardware import HardwareReport, detect_hardware


def test_detect_hardware_structure():
    hw = detect_hardware()
    assert isinstance(hw, HardwareReport)
    assert hw.cpu_physical_cores >= 1
    assert hw.cpu_logical_cores >= hw.cpu_physical_cores
    assert hw.ram_total_gb > 0.0
    assert hw.ram_available_gb > 0.0
    assert hw.disk_total_gb > 0.0
    assert hw.disk_free_gb > 0.0
    assert hw.device_tier in ["cuda-accelerated", "cpu-large", "cpu-medium", "cpu-light"]


def test_detect_hardware_dict_serialization():
    hw = detect_hardware()
    d = hw.to_dict()
    assert "cpu_model" in d
    assert "cpu_physical_cores" in d
    assert "ram_total_gb" in d
    assert "disk_free_gb" in d
    assert "has_cuda" in d
    assert "device_tier" in d
