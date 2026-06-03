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

"""Shared fixtures for static step manifest validation tests."""

from pathlib import Path
import tomllib

import pytest
import yaml


@pytest.fixture
def steps_root() -> Path:
    """Return the root directory that contains step manifests and index files."""
    return Path(__file__).resolve().parents[2] / "src" / "nemotron" / "steps"


@pytest.fixture
def patterns_root(steps_root: Path) -> Path:
    """Return the directory that contains cross-cutting pattern markdown files."""
    return steps_root / "patterns"


@pytest.fixture
def types_toml(steps_root: Path) -> dict:
    """Return the parsed types.toml document."""
    with (steps_root / "types.toml").open("rb") as handle:
        return tomllib.load(handle)


@pytest.fixture
def all_step_tomls(steps_root: Path) -> list[Path]:
    """Return all step.toml manifests under the steps root."""
    return sorted(steps_root.rglob("step.toml"))


@pytest.fixture
def all_step_pys(steps_root: Path) -> list[Path]:
    """Return all step.py scripts colocated with a step.toml manifest."""
    return sorted(p.parent / "step.py" for p in steps_root.rglob("step.toml") if (p.parent / "step.py").exists())


@pytest.fixture
def all_step_configs(steps_root: Path) -> list[Path]:
    """Return all config/*.yaml files under any step directory."""
    return sorted(p for p in steps_root.rglob("config/*.yaml"))


@pytest.fixture
def all_pattern_files(patterns_root: Path) -> list[Path]:
    """Return all pattern markdown files under the patterns root."""
    return sorted(patterns_root.glob("*.md"))


@pytest.fixture
def all_pattern_ids(all_pattern_files: list[Path]) -> list[str]:
    """Return every pattern id extracted from YAML frontmatter."""

    def load_pattern_id(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{path}: missing YAML frontmatter opening delimiter"
        remainder = text[4:]
        assert "\n---\n" in remainder, f"{path}: missing YAML frontmatter closing delimiter"
        frontmatter_text, _ = remainder.split("\n---\n", 1)
        frontmatter = yaml.safe_load(frontmatter_text) or {}
        assert isinstance(frontmatter, dict), f"{path}: YAML frontmatter must be a mapping"
        return str(frontmatter.get("id", "")).strip()

    return [load_pattern_id(path) for path in all_pattern_files]
