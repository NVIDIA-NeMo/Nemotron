#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "omni3/_smoke_test_stage/build"
#
# [tool.runspec.run]
# launch = "direct"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Placeholder build stage for omni3 CLI smoke testing."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    here = Path(__file__).parent
    cache_dir = Path(os.environ.get("NEMOTRON_CACHE_DIR", Path.home() / ".cache" / "nemotron"))
    output = cache_dir / "containers" / "omni3-smoke-test.tar"
    output.parent.mkdir(parents=True, exist_ok=True)

    tag = "nemotron/omni3-smoke-test:latest"
    subprocess.check_call(["podman", "build", "-f", str(here / "Dockerfile"), "-t", tag, str(here)])
    subprocess.check_call(["podman", "save", "--format", "oci-archive", "-o", str(output), tag])
    print(f"Built: oci-archive://{output}")


if __name__ == "__main__":
    main()
