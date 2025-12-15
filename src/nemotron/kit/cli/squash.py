# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Container squash utilities for Slurm execution.

Handles converting Docker images to squash files on remote clusters
using enroot. Uses deterministic naming to avoid re-squashing existing images.
"""

from __future__ import annotations

import re
from typing import Any


def container_to_sqsh_name(container: str) -> str:
    """Convert container image name to deterministic squash filename.

    Replaces any characters that can't be used in filenames with underscores.

    Args:
        container: Docker image name (e.g., "nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")

    Returns:
        Safe squash filename (e.g., "nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh")

    Examples:
        >>> container_to_sqsh_name("nvcr.io/nvidian/nemo:25.11-nano-v3.rc2")
        'nvcr_io_nvidian_nemo_25_11_nano_v3_rc2.sqsh'
        >>> container_to_sqsh_name("rayproject/ray:nightly-extra-py312-cpu")
        'rayproject_ray_nightly_extra_py312_cpu.sqsh'
    """
    # Replace any non-alphanumeric characters (except underscore) with underscore
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", container)
    # Collapse multiple underscores into one
    safe_name = re.sub(r"_+", "_", safe_name)
    # Strip leading/trailing underscores
    safe_name = safe_name.strip("_")
    return f"{safe_name}.sqsh"


def check_sqsh_exists(tunnel: Any, remote_path: str) -> bool:
    """Check if a squash file exists on the remote cluster.

    Args:
        tunnel: nemo-run SSHTunnel instance
        remote_path: Full path to the squash file

    Returns:
        True if file exists, False otherwise
    """
    result = tunnel.run(f"test -f {remote_path} && echo exists", hide=True, warn=True)
    return result.ok and "exists" in result.stdout
