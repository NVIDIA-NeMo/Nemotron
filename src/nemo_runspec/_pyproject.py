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

    uv = pyproject_data.get("tool", {}).get("uv", {})

    # [tool.uv.sources] — convert relative paths to absolute
    if "sources" in uv:
        buf.write("[tool.uv.sources]\n")
        for key, value in uv["sources"].items():
            if "path" in value:
                source_path = Path(value["path"])
                if not source_path.is_absolute():
                    source_path = (stage_dir / source_path).resolve()
                buf.write(f'{key} = {{ path = "{source_path}" }}\n')
        buf.write("\n")

    # [tool.uv.extra-build-dependencies]
    if "extra-build-dependencies" in uv:
        buf.write("[tool.uv.extra-build-dependencies]\n")
        for key, deps in uv["extra-build-dependencies"].items():
            deps_str = "[" + ", ".join(f'"{d}"' for d in deps) + "]"
            buf.write(f"{key} = {deps_str}\n")
        buf.write("\n")

    # [tool.uv]
    buf.write("[tool.uv]\n")
    if "override-dependencies" in uv:
        buf.write("override-dependencies = [\n")
        for dep in uv["override-dependencies"]:
            buf.write(f'  "{dep}",\n')
        buf.write("]\n")
    buf.write("exclude-dependencies = [\n")
    for dep in exclude_deps:
        buf.write(f'  "{dep}",\n')
    buf.write("]\n")

    (temp_dir / "pyproject.toml").write_text(buf.getvalue())
    return temp_dir
