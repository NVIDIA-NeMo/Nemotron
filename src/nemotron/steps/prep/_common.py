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

"""Shared helpers for data-prep steps."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


def resolve_blend_path(cfg: dict[str, Any], *, step_dir: Path, default_name: str = "blend_tiny.json") -> Path:
    """Resolve a local data blend path, falling back to the step's bundled data."""
    data_dir = step_dir / "data"
    raw = cfg.get("blend_path")
    if raw:
        candidate = Path(str(raw)).expanduser()
        candidates = (
            [candidate]
            if candidate.is_absolute()
            else [
                Path.cwd() / candidate,
                step_dir / candidate,
                data_dir / candidate.name,
            ]
        )
        for path in candidates:
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(f"blend_path {raw!r} not found locally or in {data_dir}")

    default = data_dir / default_name
    if not default.exists():
        raise FileNotFoundError(
            f"No blend_path configured and no built-in blend at {default}. Set blend_path: in your config."
        )
    return default.resolve()


def resolve_output_dir(value: str | os.PathLike[str]) -> str:
    """Make local output paths absolute before steps move cwd to scratch."""
    text = os.fspath(value)
    if "://" in text:
        return text
    return str(Path(text).expanduser().resolve())


def chdir_to_scratch(prefix: str) -> Path:
    """Move cwd to an empty scratch dir so Ray does not package the repo tree."""
    scratch = Path(tempfile.mkdtemp(prefix=prefix))
    os.chdir(scratch)
    return scratch


def config_dataclass(cls: type[T], block: object | None) -> T | None:
    """Instantiate a stage config dataclass from an optional YAML block."""
    if block is None:
        return None
    if not isinstance(block, dict):
        raise TypeError(f"{cls.__name__} config must be a mapping, got {type(block).__name__}")
    return cls(**block)
