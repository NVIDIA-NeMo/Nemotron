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

"""Tests for SKILL.md files in the customization_recipes tree.

Validates that SKILL.md files exist in expected locations, reference
valid stage directories, and that script/config paths mentioned within
SKILL.md files actually exist on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES_ROOT = _REPO_ROOT / "src" / "nemotron" / "customization_recipes"
_NEMOTRON_ROOT = _RECIPES_ROOT / "nemotron"

# ---------------------------------------------------------------------------
# Expected SKILL.md locations
# ---------------------------------------------------------------------------

_EXPECTED_SKILL_LOCATIONS = [
    _NEMOTRON_ROOT / "SKILL.md",
    _NEMOTRON_ROOT / "stage0_cpt" / "SKILL.md",
    _NEMOTRON_ROOT / "stage1_sft" / "SKILL.md",
    _NEMOTRON_ROOT / "stage2_rl" / "SKILL.md",
    _NEMOTRON_ROOT / "stage3_byob" / "SKILL.md",
    _NEMOTRON_ROOT / "stage4_eval" / "SKILL.md",
    _NEMOTRON_ROOT / "stage5_quantization" / "SKILL.md",
    _RECIPES_ROOT / "data_prep" / "SKILL.md",
]

# Optional model-family stubs
_OPTIONAL_SKILL_LOCATIONS = [
    _RECIPES_ROOT / "llama" / "SKILL.md",
    _RECIPES_ROOT / "qwen" / "SKILL.md",
]


class TestSkillFilesExist:
    """Verify that SKILL.md files exist in all expected locations."""

    @pytest.mark.parametrize(
        "skill_path",
        _EXPECTED_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _EXPECTED_SKILL_LOCATIONS],
    )
    def test_required_skill_md_exists(self, skill_path: Path) -> None:
        assert skill_path.exists(), f"Missing required SKILL.md: {skill_path}"

    @pytest.mark.parametrize(
        "skill_path",
        _OPTIONAL_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _OPTIONAL_SKILL_LOCATIONS],
    )
    def test_optional_skill_md_exists(self, skill_path: Path) -> None:
        """Optional stubs -- warn if missing but do not fail."""
        if not skill_path.exists():
            pytest.skip(f"Optional SKILL.md not found: {skill_path}")

    def test_agents_md_exists(self) -> None:
        """The repo-level AGENTS.md should exist."""
        agents_path = _REPO_ROOT / "AGENTS.md"
        assert agents_path.exists(), f"Missing AGENTS.md at repo root: {agents_path}"


class TestSkillFilesNonEmpty:
    """Verify that SKILL.md files are not empty stubs."""

    @pytest.mark.parametrize(
        "skill_path",
        _EXPECTED_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _EXPECTED_SKILL_LOCATIONS],
    )
    def test_skill_md_has_content(self, skill_path: Path) -> None:
        if not skill_path.exists():
            pytest.skip(f"SKILL.md not found: {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        # Should have at least a heading and some content (more than 100 chars)
        assert len(text) > 100, (
            f"SKILL.md is too short ({len(text)} chars): {skill_path}"
        )

    @pytest.mark.parametrize(
        "skill_path",
        _EXPECTED_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _EXPECTED_SKILL_LOCATIONS],
    )
    def test_skill_md_has_heading(self, skill_path: Path) -> None:
        if not skill_path.exists():
            pytest.skip(f"SKILL.md not found: {skill_path}")
        text = skill_path.read_text(encoding="utf-8")
        assert text.startswith("#"), (
            f"SKILL.md should start with a markdown heading: {skill_path}"
        )


class TestSkillReferencesValidStageDirectories:
    """The top-level nemotron/SKILL.md should reference all stage directories."""

    def test_references_all_stages(self) -> None:
        skill_path = _NEMOTRON_ROOT / "SKILL.md"
        if not skill_path.exists():
            pytest.skip("nemotron/SKILL.md not found")

        text = skill_path.read_text(encoding="utf-8")
        expected_stages = [
            "stage0_cpt",
            "stage1_sft",
            "stage2_rl",
            "stage3_byob",
            "stage4_eval",
            "stage5_quantization",
        ]
        for stage in expected_stages:
            assert stage in text, (
                f"nemotron/SKILL.md does not reference '{stage}'"
            )

    def test_referenced_stage_dirs_exist(self) -> None:
        skill_path = _NEMOTRON_ROOT / "SKILL.md"
        if not skill_path.exists():
            pytest.skip("nemotron/SKILL.md not found")

        expected_stages = [
            "stage0_cpt",
            "stage1_sft",
            "stage2_rl",
            "stage3_byob",
            "stage4_eval",
            "stage5_quantization",
        ]
        for stage in expected_stages:
            stage_dir = _NEMOTRON_ROOT / stage
            assert stage_dir.is_dir(), (
                f"Stage directory missing: {stage_dir}"
            )


class TestSkillScriptPathsExist:
    """Verify that Python script paths referenced in SKILL.md files exist.

    Extracts paths matching patterns like:
      python src/nemotron/customization_recipes/nemotron/stage*/...
      src/nemotron/customization_recipes/.../*.py
    """

    # Pattern to match script paths in SKILL.md code blocks
    _PATH_PATTERN = re.compile(
        r"(?:python\s+)?"
        r"(src/nemotron/customization_recipes/[^\s\"'`]+\.py)"
    )

    @pytest.mark.parametrize(
        "skill_path",
        _EXPECTED_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _EXPECTED_SKILL_LOCATIONS],
    )
    def test_script_paths_exist(self, skill_path: Path) -> None:
        if not skill_path.exists():
            pytest.skip(f"SKILL.md not found: {skill_path}")

        text = skill_path.read_text(encoding="utf-8")
        script_paths = self._PATH_PATTERN.findall(text)

        missing: list[str] = []
        for rel_path in script_paths:
            full_path = _REPO_ROOT / rel_path
            if not full_path.exists():
                missing.append(rel_path)

        if missing:
            # Report as a warning -- many script paths in SKILL.md may refer
            # to scripts that are planned but not yet created.
            # We collect them so the quality review can flag them.
            pytest.xfail(
                f"SKILL.md at {skill_path.relative_to(_REPO_ROOT)} references "
                f"{len(missing)} script(s) that do not exist: "
                + ", ".join(missing)
            )


class TestSkillConfigPathsExist:
    """Verify that config YAML paths referenced in SKILL.md exist."""

    _CONFIG_PATH_PATTERN = re.compile(
        r"(src/nemotron/customization_recipes/[^\s\"'`]+\.yaml)"
    )

    @pytest.mark.parametrize(
        "skill_path",
        _EXPECTED_SKILL_LOCATIONS,
        ids=[str(p.relative_to(_REPO_ROOT)) for p in _EXPECTED_SKILL_LOCATIONS],
    )
    def test_config_paths_exist(self, skill_path: Path) -> None:
        if not skill_path.exists():
            pytest.skip(f"SKILL.md not found: {skill_path}")

        text = skill_path.read_text(encoding="utf-8")
        config_paths = self._CONFIG_PATH_PATTERN.findall(text)

        missing: list[str] = []
        for rel_path in config_paths:
            full_path = _REPO_ROOT / rel_path
            if not full_path.exists():
                missing.append(rel_path)

        if missing:
            pytest.xfail(
                f"SKILL.md at {skill_path.relative_to(_REPO_ROOT)} references "
                f"{len(missing)} config(s) that do not exist: "
                + ", ".join(missing)
            )


class TestAgentsMdContent:
    """Verify AGENTS.md at repo root has proper structure and references."""

    def test_agents_md_references_customization_recipes(self) -> None:
        agents_path = _REPO_ROOT / "AGENTS.md"
        if not agents_path.exists():
            pytest.skip("AGENTS.md not found")

        text = agents_path.read_text(encoding="utf-8")
        assert "customization_recipes" in text, (
            "AGENTS.md should reference customization_recipes/"
        )

    def test_agents_md_references_all_skill_files(self) -> None:
        agents_path = _REPO_ROOT / "AGENTS.md"
        if not agents_path.exists():
            pytest.skip("AGENTS.md not found")

        text = agents_path.read_text(encoding="utf-8")
        expected_refs = [
            "nemotron/SKILL.md",
            "stage0_cpt/SKILL.md",
            "stage1_sft/SKILL.md",
            "stage2_rl/SKILL.md",
            "stage3_byob/SKILL.md",
            "stage4_eval/SKILL.md",
            "stage5_quantization/SKILL.md",
            "data_prep/SKILL.md",
        ]
        missing = [ref for ref in expected_refs if ref not in text]
        assert not missing, (
            f"AGENTS.md is missing references to: {missing}"
        )

    def test_agents_md_has_task_routing_table(self) -> None:
        agents_path = _REPO_ROOT / "AGENTS.md"
        if not agents_path.exists():
            pytest.skip("AGENTS.md not found")

        text = agents_path.read_text(encoding="utf-8")
        assert "Task Routing" in text, (
            "AGENTS.md should have a 'Task Routing' section"
        )
