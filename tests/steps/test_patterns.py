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

"""Tier 1 static validation for cross-cutting pattern markdown files."""

from __future__ import annotations

from pathlib import Path

from nemotron.steps.index import (
    VALID_PATTERN_CONFIDENCE,
    discover_patterns,
    discover_steps,
    generate_patterns_md,
)
import pytest
import yaml


REQUIRED_PATTERN_FIELDS = {"id", "title", "tags", "triggers", "confidence"}


def _load_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path}: missing YAML frontmatter opening delimiter"

    remainder = text[4:]
    assert "\n---\n" in remainder, f"{path}: missing YAML frontmatter closing delimiter"

    frontmatter_text, _ = remainder.split("\n---\n", 1)
    try:
        frontmatter = yaml.safe_load(frontmatter_text) or {}
    except yaml.YAMLError as exc:
        pytest.fail(f"{path}: invalid YAML frontmatter ({exc})")

    assert isinstance(frontmatter, dict), f"{path}: YAML frontmatter must be a mapping"
    return frontmatter


def test_all_pattern_markdown_files_have_valid_yaml_frontmatter(
    all_pattern_files: list[Path],
) -> None:
    assert all_pattern_files, "No pattern markdown files found under src/nemotron/steps/patterns"

    for pattern_path in all_pattern_files:
        _load_frontmatter(pattern_path)


def test_pattern_frontmatter_has_required_fields(all_pattern_files: list[Path]) -> None:
    for pattern_path in all_pattern_files:
        frontmatter = _load_frontmatter(pattern_path)
        missing = sorted(field for field in REQUIRED_PATTERN_FIELDS if not frontmatter.get(field))
        assert not missing, f"{pattern_path}: missing required frontmatter fields {missing}"



def test_pattern_ids_match_filenames(all_pattern_files: list[Path]) -> None:
    for pattern_path in all_pattern_files:
        frontmatter = _load_frontmatter(pattern_path)
        expected_id = pattern_path.stem
        assert frontmatter.get("id") == expected_id, (
            f"{pattern_path}: frontmatter id {frontmatter.get('id')!r} does not match filename {expected_id!r}"
        )



def test_pattern_ids_are_unique(all_pattern_ids: list[str], all_pattern_files: list[Path]) -> None:
    seen_ids: dict[str, Path] = {}

    for pattern_path, pattern_id in zip(all_pattern_files, all_pattern_ids, strict=True):
        assert pattern_id not in seen_ids, (
            f"Duplicate pattern id {pattern_id!r}: {seen_ids[pattern_id]} and {pattern_path}"
        )
        seen_ids[pattern_id] = pattern_path



def test_pattern_step_references_are_valid_step_ids(
    steps_root: Path,
    all_pattern_files: list[Path],
) -> None:
    known_step_ids = {step.id for step in discover_steps(steps_root)}

    for pattern_path in all_pattern_files:
        frontmatter = _load_frontmatter(pattern_path)
        steps = frontmatter.get("steps", [])
        assert isinstance(steps, list), f"{pattern_path}: frontmatter steps must be a list"
        for step_id in steps:
            assert step_id in known_step_ids, (
                f"{pattern_path}: steps references unknown step id {step_id!r}"
            )



def test_pattern_confidence_values_are_valid(all_pattern_files: list[Path]) -> None:
    for pattern_path in all_pattern_files:
        frontmatter = _load_frontmatter(pattern_path)
        confidence = frontmatter.get("confidence")
        assert confidence in VALID_PATTERN_CONFIDENCE, (
            f"{pattern_path}: confidence must be one of {sorted(VALID_PATTERN_CONFIDENCE)}, got {confidence!r}"
        )



def test_patterns_index_is_up_to_date(steps_root: Path) -> None:
    patterns_index_path = steps_root / "PATTERNS.md"
    assert patterns_index_path.exists(), "src/nemotron/steps/PATTERNS.md does not exist"

    discovered_patterns = discover_patterns(steps_root / "patterns")
    expected = generate_patterns_md(discovered_patterns)
    actual = patterns_index_path.read_text(encoding="utf-8")

    assert actual == expected, "src/nemotron/steps/PATTERNS.md is out of date"
