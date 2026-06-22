"""Static validator for the BYOB skill package."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

REQUIRED_FILES = (
    "README.md",
    "step.toml",
    "adapter.py",
    "step.py",
    "scripts/run.py",
    "scripts/runtime.py",
    "scripts/validate.py",
    "runtime/benchmark_families/base.py",
    "runtime/benchmark_families/registry.py",
    "runtime/benchmark_families/mcq/family.py",
    "runtime/benchmark_families/mcq/pipeline.py",
    "config/default.yaml",
    "config/tiny.yaml",
    "config/translate.yaml",
    "references/STEP.md",
    "references/guide.md",
    "references/benchmark-schema.md",
    "references/new-family-checklist.md",
    "references/quality-and-filtering.md",
    "patterns/index.yaml",
    "eval/golden_cases.yaml",
    "eval/skill_cases.yaml",
)


def validate_skill_dir(skill_dir: Path) -> list[str]:
    """Return validation errors for a BYOB skill directory."""
    errors: list[str] = []
    for rel_path in REQUIRED_FILES:
        if not (skill_dir / rel_path).exists():
            errors.append(f"missing required file: {rel_path}")

    skill_path = skill_dir / "README.md"
    if skill_path.exists():
        frontmatter = _load_frontmatter(skill_path)
        if frontmatter.get("name") != skill_dir.name:
            errors.append("README.md name must match the directory name")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(frontmatter.get("name", ""))):
            errors.append("README.md name must use lowercase letters, numbers, and hyphens")
        if not frontmatter.get("description"):
            errors.append("README.md description is required")
        if not frontmatter.get("when_to_use"):
            errors.append("README.md when_to_use is required")

    for rel_path in ("config/default.yaml", "config/tiny.yaml", "config/translate.yaml"):
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


def _load_frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        return {}
    frontmatter_text, _ = text[4:].split("\n---\n", 1)
    data = yaml.safe_load(frontmatter_text) or {}
    return data if isinstance(data, dict) else {}


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
