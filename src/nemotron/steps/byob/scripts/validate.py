# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static validator for the BYOB skill package."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

SHARED_REQUIRED_FILES = (
    "README.md",
    "adapter.py",
    "scripts/run.py",
    "scripts/runtime.py",
    "scripts/validate.py",
    "runtime/benchmark_families/base.py",
    "runtime/benchmark_families/registry.py",
    "references/STEP.md",
    "references/guide.md",
    "references/benchmark-schema.md",
    "references/new-family-checklist.md",
    "references/quality-and-filtering.md",
    "patterns/index.yaml",
    "eval/golden_cases.yaml",
    "eval/skill_cases.yaml",
)

FAMILY_REQUIRED_FILES = (
    "step.toml",
    "step.py",
    "config/default.yaml",
    "config/tiny.yaml",
    "config/translate.yaml",
)

FAMILY_RUNTIME_REQUIRED_FILES = (
    "family.py",
    "pipeline.py",
)


def validate_skill_dir(skill_dir: Path) -> list[str]:
    """Return validation errors for a BYOB skill directory."""
    errors: list[str] = []
    for rel_path in SHARED_REQUIRED_FILES:
        if not (skill_dir / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")

    families = _discover_families(skill_dir)
    if not families:
        errors.append("no benchmark family found")

    for family in families:
        for family_path in FAMILY_REQUIRED_FILES:
            rel_path = f"{family}/{family_path}"
            if not (skill_dir / rel_path).exists():
                errors.append(f"missing required file: {rel_path}")

        for runtime_path in FAMILY_RUNTIME_REQUIRED_FILES:
            rel_path = f"runtime/benchmark_families/{family}/{runtime_path}"
            if not (skill_dir / rel_path).exists():
                errors.append(f"missing required file: {rel_path}")

        for config_name in ("default.yaml", "tiny.yaml", "translate.yaml"):
            rel_path = f"{family}/config/{config_name}"
            path = skill_dir / rel_path
            if path.exists():
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    errors.append(f"{rel_path} must parse to a YAML mapping")

    index_path = skill_dir / "patterns" / "index.yaml"
    if index_path.exists():
        index_data = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        for pattern in index_data.get("patterns", []):
            pattern_id = pattern.get("id")
            if pattern_id and not (skill_dir / "patterns" / f"{pattern_id}.md").exists():
                errors.append(f"patterns/index.yaml references missing pattern {pattern_id!r}")

    return errors


def _discover_families(skill_dir: Path) -> list[str]:
    """Discover family names from launch and runtime package directories."""
    families: set[str] = set()

    if skill_dir.exists():
        for child in skill_dir.iterdir():
            if child.is_dir() and any((child / marker).exists() for marker in ("step.toml", "step.py", "config")):
                families.add(child.name)

    runtime_root = skill_dir / "runtime" / "benchmark_families"
    if runtime_root.exists():
        for child in runtime_root.iterdir():
            if child.is_dir() and any((child / marker).exists() for marker in FAMILY_RUNTIME_REQUIRED_FILES):
                families.add(child.name)

    return sorted(families)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the BYOB skill package")
    parser.add_argument("--skill-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    errors = validate_skill_dir(args.skill_dir)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print("BYOB skill assets are valid")


if __name__ == "__main__":
    main()
