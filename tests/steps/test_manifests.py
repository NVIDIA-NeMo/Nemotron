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

"""Tier 1 static validation for step manifests."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _defined_type_names(types_toml: dict) -> set[str]:
    if isinstance(types_toml.get("types"), dict):
        return set(types_toml["types"])
    return {name for name, value in types_toml.items() if isinstance(value, dict) and name != "convert_to"}


def _reference_paths(reference: dict) -> list[str]:
    paths: list[str] = []
    for value in reference.values():
        if isinstance(value, str):
            paths.append(value)
        elif isinstance(value, list):
            paths.extend(item for item in value if isinstance(item, str))
    return paths


def _resolve_reference_path(repo_root: Path, raw_path: str) -> Path | None:
    if "://" in raw_path:
        return None

    path_text = raw_path.split("#", 1)[0].strip()
    if not path_text:
        return None

    path = Path(path_text)
    if path.is_absolute():
        return path

    workspace_root = repo_root.parents[1]
    candidates = [
        repo_root / path,
        repo_root.parent / path,
        workspace_root / path,
    ]
    candidates.extend(base / path for base in workspace_root.iterdir() if base.is_dir())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def test_all_step_manifests_parse_as_valid_toml(all_step_tomls: list[Path]) -> None:
    assert all_step_tomls, "No step.toml files found under src/nemotron/steps"

    for manifest_path in all_step_tomls:
        try:
            _load_toml(manifest_path)
        except tomllib.TOMLDecodeError as exc:
            pytest.fail(f"{manifest_path}: invalid TOML ({exc})")


def test_all_step_manifests_have_required_fields(all_step_tomls: list[Path]) -> None:
    for manifest_path in all_step_tomls:
        data = _load_toml(manifest_path)
        step = data.get("step")

        assert isinstance(step, dict), f"{manifest_path}: missing [step] table"
        assert step.get("id"), f"{manifest_path}: missing step.id"
        assert step.get("name"), f"{manifest_path}: missing step.name"
        assert step.get("category"), f"{manifest_path}: missing step.category"


def test_all_consumed_and_produced_types_exist(
    all_step_tomls: list[Path],
    types_toml: dict,
) -> None:
    known_types = _defined_type_names(types_toml)

    for manifest_path in all_step_tomls:
        data = _load_toml(manifest_path)
        for section_name in ("consumes", "produces"):
            for entry in data.get(section_name, []):
                type_name = entry.get("type")
                assert type_name in known_types, (
                    f"{manifest_path}: {section_name} references unknown type {type_name!r}"
                )


def test_reference_paths_point_to_real_files(all_step_tomls: list[Path]) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    for manifest_path in all_step_tomls:
        data = _load_toml(manifest_path)
        reference = data.get("reference", {})
        if not isinstance(reference, dict):
            continue

        for raw_path in _reference_paths(reference):
            resolved = _resolve_reference_path(repo_root, raw_path)
            if resolved is None:
                continue
            assert resolved.exists(), f"{manifest_path}: missing reference path {raw_path!r}"


def test_step_ids_are_unique(all_step_tomls: list[Path]) -> None:
    seen_ids: dict[str, Path] = {}

    for manifest_path in all_step_tomls:
        step_id = _load_toml(manifest_path)["step"]["id"]
        assert step_id not in seen_ids, (
            f"Duplicate step id {step_id!r}: {seen_ids[step_id]} and {manifest_path}"
        )
        seen_ids[step_id] = manifest_path


def test_category_directory_matches_manifest_category(
    all_step_tomls: list[Path],
    steps_root: Path,
) -> None:
    for manifest_path in all_step_tomls:
        manifest = _load_toml(manifest_path)
        category_from_path = manifest_path.relative_to(steps_root).parts[0]
        category_from_manifest = manifest["step"]["category"]
        assert category_from_path == category_from_manifest, (
            f"{manifest_path}: directory category {category_from_path!r} does not match "
            f"step.category {category_from_manifest!r}"
        )
