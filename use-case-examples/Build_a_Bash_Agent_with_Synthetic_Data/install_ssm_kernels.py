"""Install ABI-matched Mamba wheels for the pinned Torch/CUDA environment."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KernelPackage:
    distribution: str
    import_name: str
    version: str
    release_tag: str
    release_base_url: str
    probe_code: str


KERNEL_PACKAGES = (
    KernelPackage(
        distribution="causal-conv1d",
        import_name="causal_conv1d",
        version="1.6.1",
        release_tag="v1.6.1.post4",
        release_base_url=("https://github.com/Dao-AILab/causal-conv1d/releases/download"),
        probe_code=("from causal_conv1d import causal_conv1d_fn, causal_conv1d_update"),
    ),
    KernelPackage(
        distribution="mamba-ssm",
        import_name="mamba_ssm",
        version="2.3.1",
        release_tag="v2.3.1",
        release_base_url="https://github.com/state-spaces/mamba/releases/download",
        probe_code=(
            "from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn; "
            "from mamba_ssm.ops.triton.ssd_combined import "
            "mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined; "
            "from mamba_ssm.ops.triton.selective_state_update import "
            "selective_state_update"
        ),
    ),
)


def probe_environment(torch: Any) -> dict[str, str]:
    """Return the tags used by the projects' official release wheels."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The CUDA kernel installer supports Linux only.")
    machine = platform.machine().lower()
    platform_tag = {
        "x86_64": "linux_x86_64",
        "amd64": "linux_x86_64",
        "aarch64": "linux_aarch64",
        "arm64": "linux_aarch64",
    }.get(machine)
    if platform_tag is None:
        raise RuntimeError(f"No published Mamba wheels for architecture {machine!r}.")

    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        raise RuntimeError("A CUDA-enabled PyTorch build is required.")
    cuda_major = str(cuda_version).split(".", 1)[0]
    if cuda_major not in {"12", "13"}:
        raise RuntimeError(f"No pinned release wheels for CUDA {cuda_version}.")

    torch_parts = torch.__version__.split("+", 1)[0].split(".")
    torch_major_minor = ".".join(torch_parts[:2])
    if torch_major_minor != "2.10":
        raise RuntimeError(f"This tutorial pins Torch 2.10, but found {torch.__version__}.")

    return {
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch": torch_major_minor,
        "cuda": cuda_major,
        "abi": str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(),
        "platform": platform_tag,
    }


def wheel_url(package: KernelPackage, environment: dict[str, str]) -> str:
    filename = (
        f"{package.import_name}-{package.version}"
        f"+cu{environment['cuda']}torch{environment['torch']}"
        f"cxx11abi{environment['abi']}-{environment['python_tag']}"
        f"-{environment['python_tag']}-{environment['platform']}.whl"
    )
    return f"{package.release_base_url}/{package.release_tag}/{filename}"


def _uv_install(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required; install it before running this helper.")
    return subprocess.run(
        [uv, "pip", "install", "--python", sys.executable, *arguments],
        check=False,
        text=True,
    )


def _is_usable(package: KernelPackage) -> bool:
    """Probe the exact kernel symbols in a fresh interpreter."""
    result = subprocess.run(
        [sys.executable, "-c", package.probe_code],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def install_package(package: KernelPackage, environment: dict[str, str]) -> None:
    """Try the official wheel first, then compile against the active Torch ABI."""
    try:
        installed_version = importlib.metadata.version(package.distribution)
    except importlib.metadata.PackageNotFoundError:
        installed_version = None
    if installed_version == package.version and _is_usable(package):
        print(f"{package.distribution} {package.version} kernels are already usable")
        return

    url = wheel_url(package, environment)
    print(f"Installing {package.distribution} from {url}")
    result = _uv_install(["--no-deps", "--reinstall", url])
    if result.returncode == 0 and _is_usable(package):
        return

    print(f"No compatible prebuilt {package.distribution} wheel; building from source.")
    result = _uv_install(
        [
            "--no-build-isolation",
            "--no-deps",
            "--reinstall",
            f"{package.distribution}=={package.version}",
        ]
    )
    if result.returncode != 0 or not _is_usable(package):
        raise RuntimeError(f"Could not install {package.distribution}.")


def main() -> None:
    import torch

    environment = probe_environment(torch)
    for package in KERNEL_PACKAGES:
        install_package(package, environment)
    print("Mamba CUDA kernels are ready.")


if __name__ == "__main__":
    main()
