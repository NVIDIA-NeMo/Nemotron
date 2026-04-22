#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "omni3/stage1_rl/build"
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Build the shared Omni RL container."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


CACHE_DIR = Path(os.environ.get("NEMOTRON_CACHE_DIR", Path.home() / ".cache" / "nemotron"))


def main() -> None:
    """Build and export the shared Omni RL image as an OCI archive."""
    here = Path(__file__).parent
    tag = "nemotron/omni3-rl:latest"
    output = CACHE_DIR / "containers" / "omni3-rl.tar"
    output.parent.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        ["podman", "build", "-f", str(here / "Dockerfile"), "-t", tag, str(here)]
    )
    subprocess.check_call(
        ["podman", "save", "--format", "oci-archive", "-o", str(output), tag]
    )
    print(f"Built: oci-archive://{output}")


if __name__ == "__main__":
    main()
