import subprocess
import sys
from unittest.mock import Mock, patch

from install_ssm_kernels import KERNEL_PACKAGES, _is_usable, wheel_url


def test_pinned_kernel_wheel_urls_match_torch_cuda_and_python_tags():
    environment = {
        "python_tag": "cp312",
        "torch": "2.10",
        "cuda": "12",
        "abi": "TRUE",
        "platform": "linux_x86_64",
    }
    urls = [wheel_url(package, environment) for package in KERNEL_PACKAGES]
    assert urls[0].endswith(
        "/v1.6.1.post4/causal_conv1d-1.6.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )
    assert urls[1].endswith(
        "/v2.3.1/mamba_ssm-2.3.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
    )


@patch("install_ssm_kernels.subprocess.run")
def test_kernel_probe_uses_a_fresh_interpreter_and_exact_symbols(run: Mock):
    run.return_value.returncode = 0

    assert _is_usable(KERNEL_PACKAGES[1])

    run.assert_called_once_with(
        [sys.executable, "-c", KERNEL_PACKAGES[1].probe_code],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert "mamba_chunk_scan_combined" in KERNEL_PACKAGES[1].probe_code
    assert "selective_state_update" in KERNEL_PACKAGES[1].probe_code
