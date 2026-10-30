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

"""Tier 1 static validation for types.toml."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _type_definitions(types_toml: dict) -> dict[str, dict]:
    if isinstance(types_toml.get("types"), dict):
        return {
            name: value
            for name, value in types_toml["types"].items()
            if isinstance(value, dict)
        }
    return {
        name: value
        for name, value in types_toml.items()
        if isinstance(value, dict) and name != "convert_to"
    }


def _convert_to_step_ids(types_toml: dict) -> set[str]:
    step_ids: set[str] = set()

    top_level_convert_to = types_toml.get("convert_to")
    if isinstance(top_level_convert_to, dict):
        for value in top_level_convert_to.values():
            if isinstance(value, str):
                step_ids.add(value)
            elif isinstance(value, dict):
                step = value.get("step")
                if isinstance(step, str):
                    step_ids.add(step)

    for type_def in _type_definitions(types_toml).values():
        convert_to = type_def.get("convert_to")
        if not isinstance(convert_to, dict):
            continue
        for value in convert_to.values():
            if isinstance(value, str):
                step_ids.add(value)
            elif isinstance(value, dict):
                step = value.get("step")
                if isinstance(step, str):
                    step_ids.add(step)

    return step_ids


def _all_used_type_names(all_step_tomls: list[Path]) -> set[str]:
    used_types: set[str] = set()
    for manifest_path in all_step_tomls:
        data = _load_toml(manifest_path)
        for section_name in ("consumes", "produces"):
            for entry in data.get(section_name, []):
                type_name = entry.get("type")
                if isinstance(type_name, str):
                    used_types.add(type_name)
    return used_types


def test_types_toml_parses_as_valid_toml(steps_root: Path) -> None:
    types_path = steps_root / "types.toml"
    try:
        _load_toml(types_path)
    except tomllib.TOMLDecodeError as exc:
        pytest.fail(f"{types_path}: invalid TOML ({exc})")


def test_is_a_relationships_have_no_cycles(types_toml: dict) -> None:
    type_defs = _type_definitions(types_toml)
    state: dict[str, str] = {}

    def visit(type_name: str, stack: list[str]) -> None:
        current_state = state.get(type_name)
        if current_state == "visiting":
            cycle = " -> ".join([*stack, type_name])
            pytest.fail(f"Cycle detected in is_a relationships: {cycle}")
        if current_state == "visited":
            return

        state[type_name] = "visiting"
        parents = type_defs[type_name].get("is_a", [])
        if isinstance(parents, str):
            parents = [parents]

        for parent in parents:
            assert parent in type_defs, f"Unknown is_a parent {parent!r} referenced by {type_name!r}"
            visit(parent, [*stack, type_name])

        state[type_name] = "visited"

    for type_name in type_defs:
        visit(type_name, [])


def test_convert_to_values_reference_valid_step_ids(
    types_toml: dict,
    all_step_tomls: list[Path],
) -> None:
    known_step_ids = {
        _load_toml(manifest_path)["step"]["id"]
        for manifest_path in all_step_tomls
    }

    for step_id in _convert_to_step_ids(types_toml):
        assert step_id in known_step_ids, f"types.toml convert_to references unknown step id {step_id!r}"


def test_every_type_used_in_step_manifests_exists(
    types_toml: dict,
    all_step_tomls: list[Path],
) -> None:
    defined_types = set(_type_definitions(types_toml))
    used_types = _all_used_type_names(all_step_tomls)
    missing_types = sorted(used_types - defined_types)

    assert not missing_types, f"Types used by step manifests are missing from types.toml: {missing_types}"
