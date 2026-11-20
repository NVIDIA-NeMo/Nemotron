# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""nemo_runspec - Bridge layer for PEP 723 ``[tool.runspec]`` metadata.

Parses declarative metadata from recipe scripts and provides tooling
to convert it into nemo-run constructs.

Usage::

    from nemo_runspec import parse

    SPEC = parse("src/nemotron/recipes/nano3/stage0_pretrain/train.py")
    print(SPEC.name)        # "nano3/pretrain"
    print(SPEC.image)       # "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
    print(SPEC.config_dir)  # Path to config directory
"""

from nemo_runspec._models import Runspec, RunspecConfig, RunspecResources, RunspecRun
from nemo_runspec._parser import parse


def __getattr__(name: str):
    # The metadata parser and uv bootstrap helpers use only the standard
    # library. Load artifact tracking dependencies only when requested.
    if name == "ManifestTracker":
        from nemo_runspec.manifest_tracker import ManifestTracker

        globals()[name] = ManifestTracker
        return ManifestTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "parse",
    "Runspec",
    "RunspecRun",
    "RunspecConfig",
    "RunspecResources",
    "ManifestTracker",
]
