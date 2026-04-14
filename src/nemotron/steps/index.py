from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import tomllib

import yaml


DEFAULT_STEPS_ROOT = Path(__file__).resolve().parent
DEFAULT_PATTERNS_DIR = DEFAULT_STEPS_ROOT / "patterns"
CATEGORY_TITLES = {
    "benchmark": "Benchmarking",
    "convert": "Conversion",
    "curate": "Data Curation",
    "eval": "Evaluation",
    "prep": "Data Preparation",
    "pretrain": "Pretraining",
    "rl": "Reinforcement Learning",
    "sft": "Supervised Fine-Tuning",
    "synth": "Synthetic Data Generation",
    "translate": "Translation",
}
VALID_PATTERN_CONFIDENCE = {"high", "medium", "experimental"}


@dataclass(frozen=True)
class ArtifactRef:
    type: str
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class ParameterDef:
    name: str
    description: str = ""
    default: object | None = None
    choices: tuple[object, ...] = ()


@dataclass(frozen=True)
class StepInfo:
    id: str
    name: str
    category: str
    description: str
    path: Path
    stack: str | None = None
    difficulty: str | None = None
    tags: tuple[str, ...] = ()
    consumes: tuple[ArtifactRef, ...] = ()
    produces: tuple[ArtifactRef, ...] = ()
    parameters: tuple[ParameterDef, ...] = ()


@dataclass(frozen=True)
class PatternInfo:
    id: str
    title: str
    tags: tuple[str, ...]
    triggers: tuple[str, ...]
    steps: tuple[str, ...]
    confidence: str
    path: Path


def discover_steps(steps_root: Path | None = None) -> list[StepInfo]:
    """Walk the step tree and parse every step.toml manifest."""
    root = (steps_root or DEFAULT_STEPS_ROOT).resolve()
    steps: list[StepInfo] = []

    for manifest_path in sorted(root.rglob("step.toml")):
        data = _load_toml(manifest_path)
        steps.append(_parse_step_info(manifest_path, data))

    return sorted(steps, key=lambda step: (step.category, step.id, step.path.as_posix()))


def discover_patterns(patterns_dir: Path | None = None) -> list[PatternInfo]:
    """Walk the patterns tree and parse Markdown files with YAML frontmatter."""
    root = (patterns_dir or DEFAULT_PATTERNS_DIR).resolve()
    if not root.exists():
        return []

    patterns: list[PatternInfo] = []
    for pattern_path in sorted(root.glob("*.md")):
        frontmatter, _ = _split_frontmatter(pattern_path)
        patterns.append(_parse_pattern_info(pattern_path, frontmatter))

    return sorted(patterns, key=lambda pattern: (pattern.id, pattern.path.as_posix()))


def generate_steps_md(steps_root: Path | None = None) -> str:
    """Render a Markdown catalog of all discovered steps."""
    root = (steps_root or DEFAULT_STEPS_ROOT).resolve()
    steps = discover_steps(root)

    lines = ["# Available Steps", ""]
    if not steps:
        lines.append("No steps discovered yet.")
        return "\n".join(lines) + "\n"

    grouped: dict[str, list[StepInfo]] = defaultdict(list)
    for step in steps:
        grouped[step.category].append(step)

    for category in sorted(grouped):
        title = CATEGORY_TITLES.get(category)
        heading = f"## {category} — {title}" if title else f"## {category}"
        lines.extend([heading, ""])

        category_steps = grouped[category]
        include_stack = any(step.stack for step in category_steps)
        include_difficulty = any(step.difficulty for step in category_steps)

        headers = ["Step"]
        if include_stack:
            headers.append("Stack")
        if include_difficulty:
            headers.append("Difficulty")
        headers.extend(["Description", "Consumes", "Produces"])

        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        for step in category_steps:
            rel_path = step.path.relative_to(root).as_posix()
            row = [f"[{step.id}]({rel_path}/)"]
            if include_stack:
                row.append(step.stack or "-")
            if include_difficulty:
                row.append(step.difficulty or "-")
            row.extend([
                _escape_pipes(step.description),
                _format_artifacts(step.consumes),
                _format_artifacts(step.produces),
            ])
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_patterns_md(
    patterns: list[PatternInfo],
    output_path: Path | None = None,
) -> str:
    """Render a Markdown catalog of all discovered patterns and optionally write it."""
    root = patterns[0].path.parent.parent if patterns else (output_path.parent if output_path else DEFAULT_STEPS_ROOT).resolve()
    lines = ["# Available Patterns", ""]

    if not patterns:
        lines.append("No patterns discovered yet.")
        content = "\n".join(lines) + "\n"
        if output_path is not None:
            output_path.write_text(content, encoding="utf-8")
        return content

    lines.extend(
        [
            "| ID | Title | Tags | Triggers | Confidence |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for pattern in patterns:
        rel_path = pattern.path.relative_to(root).as_posix()
        lines.append(
            "| "
            + " | ".join(
                [
                    f"[{pattern.id}]({rel_path})",
                    _escape_pipes(pattern.title),
                    _escape_pipes(", ".join(pattern.tags) or "-"),
                    _escape_pipes("<br>".join(pattern.triggers) or "-"),
                    pattern.confidence,
                ]
            )
            + " |"
        )

    content = "\n".join(lines).rstrip() + "\n"
    if output_path is not None:
        output_path.write_text(content, encoding="utf-8")
    return content


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _parse_step_info(manifest_path: Path, data: dict) -> StepInfo:
    step_data = data.get("step", {})
    category = str(step_data.get("category") or manifest_path.parent.parent.name)
    default_id = f"{category}/{manifest_path.parent.name}"

    parameters = tuple(
        ParameterDef(
            name=str(entry.get("name", "")),
            description=str(entry.get("description", "")),
            default=entry.get("default"),
            choices=tuple(entry.get("choices", ()) or ()),
        )
        for entry in _table_list(data.get("parameters"))
    )

    return StepInfo(
        id=str(step_data.get("id") or default_id),
        name=str(step_data.get("name") or manifest_path.parent.name.replace("_", " ").replace("-", " ").title()),
        category=category,
        description=str(step_data.get("description") or step_data.get("summary") or ""),
        path=manifest_path.parent.resolve(),
        stack=_maybe_str(step_data.get("stack")),
        difficulty=_maybe_str(step_data.get("difficulty")),
        tags=tuple(str(tag) for tag in (step_data.get("tags") or ())),
        consumes=_parse_artifacts(data.get("consumes"), default_required=True),
        produces=_parse_artifacts(data.get("produces"), default_required=True),
        parameters=parameters,
    )


def _parse_pattern_info(pattern_path: Path, frontmatter: dict) -> PatternInfo:
    return PatternInfo(
        id=str(frontmatter.get("id", "")).strip(),
        title=str(frontmatter.get("title", "")).strip(),
        tags=_as_str_tuple(frontmatter.get("tags")),
        triggers=_as_str_tuple(frontmatter.get("triggers")),
        steps=_as_str_tuple(frontmatter.get("steps")),
        confidence=str(frontmatter.get("confidence", "")).strip(),
        path=pattern_path.resolve(),
    )


def _split_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter opening delimiter")

    remainder = text[4:]
    if "\n---\n" not in remainder:
        raise ValueError(f"{path}: missing YAML frontmatter closing delimiter")

    frontmatter_text, body = remainder.split("\n---\n", 1)
    data = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML frontmatter must parse to a mapping")
    return data, body


def _parse_artifacts(raw_entries: object, *, default_required: bool) -> tuple[ArtifactRef, ...]:
    return tuple(
        ArtifactRef(
            type=str(entry.get("type", "")),
            description=str(entry.get("description", "")),
            required=bool(entry.get("required", default_required)),
        )
        for entry in _table_list(raw_entries)
    )


def _table_list(value: object) -> list[dict]:
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    return []


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _format_artifacts(artifacts: tuple[ArtifactRef, ...]) -> str:
    if not artifacts:
        return "-"

    rendered = []
    for artifact in artifacts:
        label = artifact.type
        if not artifact.required:
            label = f"{label} (optional)"
        rendered.append(label)
    return ", ".join(rendered)


def _escape_pipes(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip() or "-"


def _maybe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate src/nemotron/steps index markdown files")
    parser.add_argument("--steps-root", type=Path, default=DEFAULT_STEPS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_STEPS_ROOT / "STEPS.md")
    parser.add_argument("--patterns-dir", type=Path, default=DEFAULT_PATTERNS_DIR)
    parser.add_argument("--patterns-output", type=Path, default=DEFAULT_STEPS_ROOT / "PATTERNS.md")
    args = parser.parse_args()

    steps_content = generate_steps_md(args.steps_root)
    args.output.write_text(steps_content, encoding="utf-8")

    patterns = discover_patterns(args.patterns_dir)
    generate_patterns_md(patterns, args.patterns_output)


if __name__ == "__main__":
    main()
