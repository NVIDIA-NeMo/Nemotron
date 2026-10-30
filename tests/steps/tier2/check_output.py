"""Deterministic structural checks for generated Tier 2 project outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import re

import yaml


STAGE_DIR_PATTERN = re.compile(r"^\d{2}_[A-Za-z0-9_]+$")
TRAINING_STAGE_KEYWORDS = ("sft", "train", "rl", "pretrain")


@dataclass(frozen=True)
class CheckResult:
    """Single structural validation result."""

    ok: bool
    check: str
    detail: str
    path: str | None = None


def check_project(project_dir: Path, expected: dict) -> list[CheckResult]:
    """Run deterministic checks against a generated project directory."""
    project_dir = project_dir.resolve()
    allow_missing = bool(expected.get("allow_missing"))

    if not project_dir.exists():
        if allow_missing:
            return [_pass("project_exists", "project output intentionally omitted")]
        return [_fail("project_exists", f"project directory not found: {project_dir}", path=str(project_dir))]

    results: list[CheckResult] = []
    must_have_files = expected.get("must_have_files", [])
    must_have_stages = expected.get("must_have_stages", [])

    for relative_path in must_have_files:
        target = project_dir / relative_path
        results.append(
            _pass("required_file", f"found {relative_path}", path=relative_path)
            if target.exists()
            else _fail("required_file", f"missing required file: {relative_path}", path=relative_path)
        )

    stages_root = project_dir / "stages"
    for stage_name in must_have_stages:
        stage_dir = stages_root / stage_name
        results.append(
            _pass("required_stage", f"found stage {stage_name}", path=str(stage_dir))
            if stage_dir.exists()
            else _fail("required_stage", f"missing stage directory: {stage_name}", path=str(stage_dir))
        )

    if stages_root.exists():
        for stage_dir in sorted(path for path in stages_root.iterdir() if path.is_dir()):
            if STAGE_DIR_PATTERN.match(stage_dir.name):
                results.append(_pass("stage_name", f"stage directory name is valid: {stage_dir.name}", path=str(stage_dir)))
            else:
                results.append(
                    _fail(
                        "stage_name",
                        f"stage directory does not follow NN_name convention: {stage_dir.name}",
                        path=str(stage_dir),
                    )
                )

    for py_file in sorted(project_dir.rglob("*.py")):
        try:
            ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError as exc:
            results.append(_fail("python_syntax", f"invalid Python syntax: {exc}", path=str(py_file)))
        else:
            results.append(_pass("python_syntax", "valid Python syntax", path=str(py_file)))

    yaml_files = sorted([*project_dir.rglob("*.yaml"), *project_dir.rglob("*.yml")])
    parsed_yaml: dict[Path, object] = {}
    for yaml_file in yaml_files:
        try:
            parsed_yaml[yaml_file] = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            results.append(_fail("yaml_parse", f"invalid YAML: {exc}", path=str(yaml_file)))
        else:
            results.append(_pass("yaml_parse", "valid YAML", path=str(yaml_file)))

    for relative_path, assertions in expected.get("config_assertions", {}).items():
        config_path = _resolve_config_path(project_dir, relative_path)
        if not config_path.exists():
            results.append(
                _fail(
                    "config_assertions",
                    f"config file for assertions not found: {relative_path}",
                    path=str(config_path),
                )
            )
            continue

        config_data = parsed_yaml.get(config_path)
        if config_data is None and config_path not in parsed_yaml:
            try:
                config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                results.append(
                    _fail("config_assertions", f"could not parse config for assertions: {exc}", path=str(config_path))
                )
                continue

        for assertion in assertions:
            if _yaml_assertion_matches(config_data, assertion):
                results.append(
                    _pass(
                        "config_assertion",
                        f"matched assertion {assertion!r}",
                        path=str(config_path),
                    )
                )
            else:
                results.append(
                    _fail(
                        "config_assertion",
                        f"config does not satisfy assertion {assertion!r}",
                        path=str(config_path),
                    )
                )

    if expected.get("no_relative_data_paths"):
        for yaml_file in yaml_files:
            text = yaml_file.read_text(encoding="utf-8")
            if "../" in text:
                results.append(
                    _fail(
                        "no_relative_data_paths",
                        "relative parent-directory path found in YAML",
                        path=str(yaml_file),
                    )
                )
            else:
                results.append(
                    _pass(
                        "no_relative_data_paths",
                        "no relative parent-directory paths found",
                        path=str(yaml_file),
                    )
                )

    results.append(_check_pipeline_stages_match(project_dir))

    if expected.get("must_have_tiny_config") and stages_root.exists():
        for stage_dir in sorted(path for path in stages_root.iterdir() if path.is_dir() and _is_training_stage(path)):
            tiny_config = stage_dir / "config" / "tiny.yaml"
            results.append(
                _pass("tiny_config", "training stage has tiny.yaml", path=str(tiny_config))
                if tiny_config.exists()
                else _fail("tiny_config", "training stage is missing config/tiny.yaml", path=str(tiny_config))
            )

    return results


def _check_pipeline_stages_match(project_dir: Path) -> CheckResult:
    pipeline_path = project_dir / "pipeline.py"
    stages_root = project_dir / "stages"

    if not pipeline_path.exists():
        return _fail("pipeline_stage_match", "pipeline.py not found", path=str(pipeline_path))
    if not stages_root.exists():
        return _fail("pipeline_stage_match", "stages/ directory not found", path=str(stages_root))

    actual_stages = {path.name for path in stages_root.iterdir() if path.is_dir()}
    declared_stages = _extract_pipeline_stages(pipeline_path)
    if not declared_stages:
        return _fail("pipeline_stage_match", "could not discover stage names from pipeline.py", path=str(pipeline_path))

    missing = sorted(actual_stages - declared_stages)
    extra = sorted(declared_stages - actual_stages)
    if missing or extra:
        detail_parts: list[str] = []
        if missing:
            detail_parts.append(f"missing from pipeline.py: {missing}")
        if extra:
            detail_parts.append(f"declared but not on disk: {extra}")
        return _fail("pipeline_stage_match", "; ".join(detail_parts), path=str(pipeline_path))

    return _pass("pipeline_stage_match", "pipeline.py stage list matches stage directories", path=str(pipeline_path))


def _extract_pipeline_stages(pipeline_path: Path) -> set[str]:
    source = pipeline_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(pipeline_path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "STAGES" for target in node.targets):
            continue
        stages = _literal_stage_names(node.value)
        if stages:
            return stages

    return set(re.findall(r"\b\d{2}_[A-Za-z0-9_]+\b", source))


def _literal_stage_names(value: ast.AST) -> set[str]:
    if not isinstance(value, (ast.List, ast.Tuple)):
        return set()

    stages: set[str] = set()
    for element in value.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            stages.add(element.value)
            continue
        if isinstance(element, (ast.List, ast.Tuple)) and element.elts:
            first = element.elts[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                stages.add(first.value)
    return stages


def _resolve_config_path(project_dir: Path, relative_path: str) -> Path:
    candidate = project_dir / relative_path
    if candidate.exists() or relative_path.startswith("stages/"):
        return candidate
    return project_dir / "stages" / relative_path


def _yaml_assertion_matches(config_data: object, assertion: object) -> bool:
    if isinstance(assertion, dict):
        key_path = assertion.get("path")
        expected_value = assertion.get("equals")
        if isinstance(key_path, str):
            return _get_by_path(config_data, key_path.split(".")) == expected_value
        return False

    if isinstance(assertion, str):
        if ":" in assertion:
            key, raw_value = assertion.split(":", 1)
            key = key.strip()
            expected_value = yaml.safe_load(raw_value.strip())
            return _contains_key_value(config_data, key, expected_value)
        return _contains_key(config_data, assertion.strip())

    return False


def _contains_key_value(data: object, key: str, expected_value: object) -> bool:
    if isinstance(data, dict):
        for current_key, value in data.items():
            if current_key == key and value == expected_value:
                return True
            if _contains_key_value(value, key, expected_value):
                return True
    elif isinstance(data, list):
        return any(_contains_key_value(item, key, expected_value) for item in data)
    return False


def _contains_key(data: object, key: str) -> bool:
    if isinstance(data, dict):
        if key in data:
            return True
        return any(_contains_key(value, key) for value in data.values())
    if isinstance(data, list):
        return any(_contains_key(item, key) for item in data)
    return False


def _get_by_path(data: object, parts: list[str]) -> object:
    current = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_training_stage(stage_dir: Path) -> bool:
    stage_name = stage_dir.name.split("_", 1)[-1].lower()
    if any(keyword in stage_name for keyword in TRAINING_STAGE_KEYWORDS):
        return True
    return (stage_dir / "train.py").exists()


def _pass(check: str, detail: str, *, path: str | None = None) -> CheckResult:
    return CheckResult(ok=True, check=check, detail=detail, path=path)


def _fail(check: str, detail: str, *, path: str | None = None) -> CheckResult:
    return CheckResult(ok=False, check=check, detail=detail, path=path)
