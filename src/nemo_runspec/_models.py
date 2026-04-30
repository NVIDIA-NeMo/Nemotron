# Copyright (c) Nemotron Contributors
# SPDX-License-Identifier: MIT

"""Runspec dataclasses - purely declarative metadata for recipe scripts.

These frozen dataclasses represent the [tool.runspec] TOML block that lives
inside PEP 723 inline script metadata.  They describe *what* a script is and
needs (identity, container, launch method, resources) without any
implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RunspecRun:
    """How to launch the script.

    ``cmd`` is ``None`` when the script's ``[tool.runspec.run]`` block does
    not declare one — in that case the **backend** chooses the actual
    invocation based on ``launch`` (e.g. ``torchrun`` wrapping with the right
    nproc/node-rank for distributed multi-process training, or a bare
    subprocess locally). A non-None value is treated as a verbatim author
    override and forwarded as-is; ``{script}`` and ``{config}`` placeholders
    are still expanded.
    """

    launch: str = "torchrun"  # "torchrun" | "ray" | "direct"
    cmd: str | None = None
    workdir: str | None = None  # e.g., "/opt/nemo-rl"


@dataclass(frozen=True)
class RunspecConfig:
    """Where configs live, relative to the script file."""

    dir: str = "./config"
    default: str = "default"
    format: str = "omegaconf"  # "omegaconf" | "yaml" | "json"


@dataclass(frozen=True)
class RunspecResources:
    """Default resource requirements."""

    nodes: int = 1
    gpus_per_node: int = 8


@dataclass(frozen=True)
class Runspec:
    """Parsed [tool.runspec] metadata from a recipe script.

    This is the top-level object returned by ``parse()``.
    """

    schema: str = "1"
    docs: str = ""
    name: str = ""
    image: str | None = None
    setup: str = ""
    run: RunspecRun = field(default_factory=RunspecRun)
    config: RunspecConfig = field(default_factory=RunspecConfig)
    resources: RunspecResources = field(default_factory=RunspecResources)
    env: dict[str, str] = field(default_factory=dict)
    script_path: Path = field(default_factory=lambda: Path("."))

    @property
    def config_dir(self) -> Path:
        """Absolute path to the config directory (resolved relative to script)."""
        return self.script_path.parent / self.config.dir
