#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/env/env_toml"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "lepton"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

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

"""Generate a maintainable env profile TOML file from a compact YAML template."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_runspec.config.loader import load_config
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    parse_config_and_overrides,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "lepton.yaml"
DEFAULT_OUTPUT = "env.lepton.toml"
RESERVED_SECTIONS = {"wandb", "cli", "cache", "artifacts"}


def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    config = apply_hydra_overrides(load_config(Path(config_path)), overrides)
    cfg = OmegaConf.to_container(config, resolve=False)

    output_path = Path(str(cfg.get("output_path", DEFAULT_OUTPUT))).expanduser()
    force = bool(cfg.get("force", False))
    sections = cfg.get("sections") or {}
    if not isinstance(sections, dict) or not sections:
        raise ValueError("`sections` must be a non-empty mapping of env profile sections")

    rendered = render_env_toml(sections, preamble=str(cfg.get("preamble", "") or ""))
    parsed = validate_rendered_toml(rendered, sections, cfg.get("checks") or {})

    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; set force=true to overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")

    profile_count = len([name for name in parsed if name not in RESERVED_SECTIONS])
    print(f"Wrote {output_path} with {profile_count} profile section(s)")


def render_env_toml(sections: dict[str, Any], *, preamble: str = "") -> str:
    lines: list[str] = []
    if preamble.strip():
        lines.extend(preamble.rstrip().splitlines())
        lines.append("")

    for name, values in sections.items():
        if not isinstance(values, dict):
            raise TypeError(f"Section {name!r} must be a mapping")
        lines.append(f"[{name}]")
        for key, value in values.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def validate_rendered_toml(
    rendered: str,
    source_sections: dict[str, Any],
    checks: dict[str, Any],
) -> dict[str, Any]:
    parsed = tomllib.loads(rendered)
    for name, values in parsed.items():
        parent = values.get("extends") if isinstance(values, dict) else None
        if parent and parent not in parsed:
            raise ValueError(f"Profile {name!r} extends missing profile {parent!r}")

    for profile_name, min_nodes in (checks.get("recommended_min_nodes") or {}).items():
        if profile_name not in source_sections:
            continue
        nodes = _resolved_profile_value(profile_name, parsed, "nodes")
        if nodes is not None and int(nodes) < int(min_nodes):
            print(
                f"Warning: {profile_name} uses nodes={nodes}; "
                f"recommended minimum is {min_nodes}",
                file=sys.stderr,
            )

    required = checks.get("required_profiles") or []
    missing = [name for name in required if name not in parsed]
    if missing:
        raise ValueError(f"Missing required profile(s): {', '.join(missing)}")
    return parsed


def _resolved_profile_value(
    profile_name: str,
    profiles: dict[str, Any],
    key: str,
    seen: set[str] | None = None,
) -> Any:
    seen = seen or set()
    if profile_name in seen or profile_name not in profiles:
        return None
    seen.add(profile_name)
    profile = profiles[profile_name]
    if key in profile:
        return profile[key]
    parent = profile.get("extends")
    if parent:
        return _resolved_profile_value(str(parent), profiles, key, seen)
    return None


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(item, dict) for item in value):
            return "[\n" + ",\n".join(f"    {_inline_table(item)}" for item in value) + ",\n]"
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return _inline_table(value)
    raise TypeError(f"Unsupported TOML value {value!r} ({type(value).__name__})")


def _inline_table(values: dict[str, Any]) -> str:
    items = []
    for key, value in values.items():
        if value is None:
            continue
        items.append(f"{key} = {_toml_value(value)}")
    return "{ " + ", ".join(items) + " }"


if __name__ == "__main__":
    main()
