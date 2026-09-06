"""
Libra Core - Hardware Introspection & Safeguards
Detects CPU, RAM, Disk, and GPU capabilities safely without crashes.
"""

import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass
class HardwareReport:
    os: str
    os_version: str
    architecture: str
    cpu_model: str
    cpu_physical_cores: int
    cpu_logical_cores: int
    ram_total_gb: float
    ram_available_gb: float
    disk_total_gb: float
    disk_free_gb: float
    has_cuda: bool
    gpu_name: str | None
    device_tier: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_hardware() -> HardwareReport:
    """Inspects the local machine and returns a structured HardwareReport."""
    os_name = platform.system()
    os_ver = platform.version()
    arch = platform.machine()
    cpu_model = platform.processor() or "Unknown CPU"

    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1

    mem = psutil.virtual_memory()
    ram_total_gb = round(mem.total / (1024**3), 2)
    ram_available_gb = round(mem.available / (1024**3), 2)

    # Disk usage on the current working drive
    disk = shutil.disk_usage(".")
    disk_total_gb = round(disk.total / (1024**3), 2)
    disk_free_gb = round(disk.free / (1024**3), 2)

    # GPU / CUDA detection
    has_cuda = False
    gpu_name = None

    # Optional check: try torch if installed
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
    except (ImportError, AttributeError):
        pass

    if not has_cuda and os_name == "Windows":
        try:
            res = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            gpus = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            if gpus:
                gpu_name = ", ".join(gpus)
        except (subprocess.SubprocessError, OSError):
            gpu_name = None

    # Classify device tier for safe LLM selection
    if has_cuda:
        device_tier = "cuda-accelerated"
    elif ram_available_gb >= 24:
        device_tier = "cpu-large"
    elif ram_available_gb >= 12:
        device_tier = "cpu-medium"
    else:
        device_tier = "cpu-light"

    return HardwareReport(
        os=os_name,
        os_version=os_ver,
        architecture=arch,
        cpu_model=cpu_model,
        cpu_physical_cores=physical_cores,
        cpu_logical_cores=logical_cores,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        disk_total_gb=disk_total_gb,
        disk_free_gb=disk_free_gb,
        has_cuda=has_cuda,
        gpu_name=gpu_name,
        device_tier=device_tier,
    )


if __name__ == "__main__":
    report = detect_hardware()
    import json

    print(json.dumps(report.to_dict(), indent=2))
