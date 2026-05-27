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

"""Internal pyproject.toml helpers.

Used by execution helpers (``execute_uv_local``) and the remote
``run_uv`` wrapper to synthesize a temporary pyproject.toml that
excludes container-provided packages (torch, flash-attn, …) from UV
dependency resolution.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path


def _quote_toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_toml_value(value) -> str:
    """Format the small TOML subset used by stage pyproject metadata."""
    if isinstance(value, str):
        return _quote_toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_format_toml_value(val)}" for key, val in value.items())
        return "{ " + items + " }"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _normalize_source_paths(value, stage_dir: Path):
    """Convert relative path sources to absolute paths inside source entries."""
    if isinstance(value, list):
        return [_normalize_source_paths(item, stage_dir) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = dict(value)
    if "path" in normalized:
        source_path = Path(normalized["path"])
        if not source_path.is_absolute():
            source_path = (stage_dir / source_path).resolve()
        normalized["path"] = str(source_path)
    return normalized


def _write_temp_pyproject(
    pyproject_data: dict, stage_dir: Path, exclude_deps: list[str]
) -> Path:
    """Write a temporary pyproject.toml with container exclude-dependencies."""
    temp_dir = Path(tempfile.mkdtemp())
    buf = io.StringIO()

    # [project]
    proj = pyproject_data["project"]
    buf.write("[project]\n")
    buf.write(f'name = "{proj["name"]}"\n')
    buf.write(f'version = "{proj["version"]}"\n')
    buf.write(f'requires-python = "{proj["requires-python"]}"\n')
    buf.write("dependencies = [\n")
    for dep in proj.get("dependencies", []):
        buf.write(f'  "{dep}",\n')
    buf.write("]\n\n")

    # [project.optional-dependencies]
    optional_deps = proj.get("optional-dependencies", {})
    if optional_deps:
        buf.write("[project.optional-dependencies]\n")
        for extra, deps in optional_deps.items():
            buf.write(f"{extra} = [\n")
            for dep in deps:
                buf.write(f'  "{dep}",\n')
            buf.write("]\n")
        buf.write("\n")

    uv = pyproject_data.get("tool", {}).get("uv", {})

    # [tool.uv]
    buf.write("[tool.uv]\n")
    for key, value in uv.items():
        if key in {
            "exclude-dependencies",
            "extra-build-dependencies",
            "index",
            "sources",
        }:
            continue
        buf.write(f"{key} = {_format_toml_value(value)}\n")

    combined_exclude = list(
        dict.fromkeys([*uv.get("exclude-dependencies", []), *exclude_deps])
    )
    buf.write("exclude-dependencies = [\n")
    for dep in combined_exclude:
        buf.write(f'  "{dep}",\n')
    buf.write("]\n\n")

    # [tool.uv.sources] — convert relative paths to absolute
    if "sources" in uv:
        buf.write("[tool.uv.sources]\n")
        for key, value in uv["sources"].items():
            normalized = _normalize_source_paths(value, stage_dir)
            buf.write(f"{key} = {_format_toml_value(normalized)}\n")
        buf.write("\n")

    # [[tool.uv.index]]
    for index in uv.get("index", []):
        buf.write("[[tool.uv.index]]\n")
        for key, value in index.items():
            buf.write(f"{key} = {_format_toml_value(value)}\n")
        buf.write("\n")

    # [tool.uv.extra-build-dependencies]
    if "extra-build-dependencies" in uv:
        buf.write("[tool.uv.extra-build-dependencies]\n")
        for key, deps in uv["extra-build-dependencies"].items():
            deps_str = "[" + ", ".join(f'"{d}"' for d in deps) + "]"
            buf.write(f"{key} = {deps_str}\n")
        buf.write("\n")

    (temp_dir / "pyproject.toml").write_text(buf.getvalue())
    return temp_dir
