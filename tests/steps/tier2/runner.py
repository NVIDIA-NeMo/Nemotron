"""Tier 2 runner for golden-case structural validation."""

from __future__ import annotations

from pathlib import Path
import argparse
import tomllib

from .check_output import check_project
from .plan_graph_checker import check_plan_graph, count_plan_stages, extract_mermaid_block, extract_plan_steps


DEFAULT_CASES_ROOT = Path(__file__).resolve().parent / "cases"
DEFAULT_BASELINES_ROOT = Path(__file__).resolve().parent / "baselines"


def load_case(case_name_or_path: str, *, cases_root: Path = DEFAULT_CASES_ROOT) -> tuple[Path, dict]:
    """Load a golden case TOML by stem or explicit path."""
    case_path = Path(case_name_or_path)
    if not case_path.exists():
        filename = case_name_or_path if case_name_or_path.endswith(".toml") else f"{case_name_or_path}.toml"
        case_path = cases_root / filename

    with case_path.open("rb") as handle:
        data = tomllib.load(handle)

    return case_path.resolve(), data


def load_baseline(
    case_name: str,
    *,
    baselines_root: Path = DEFAULT_BASELINES_ROOT,
) -> tuple[str | None, Path | None]:
    """Load stubbed plan/project outputs from baselines/<case>/."""
    baseline_dir = baselines_root / case_name
    plan_path = baseline_dir / "plan.md"
    project_dir = baseline_dir / "project"

    plan_md = plan_path.read_text(encoding="utf-8") if plan_path.exists() else None
    return plan_md, (project_dir if project_dir.exists() else None)


def validate_case(
    case_data: dict,
    *,
    plan_md: str | None,
    project_dir: Path | None,
) -> list[str]:
    """Run the currently implemented structural checks for one case."""
    errors: list[str] = []
    expected = case_data.get("expected", {})

    errors.extend(_validate_plan(plan_md, expected))
    errors.extend(_validate_project(project_dir, expected))

    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Tier 2 structural validation for a golden case")
    parser.add_argument("--case", required=True, help="Case stem or path to a case TOML")
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES_ROOT)
    parser.add_argument("--plan-md", type=Path, help="Optional override for plan markdown")
    parser.add_argument("--project-dir", type=Path, help="Optional override for generated project directory")
    args = parser.parse_args(argv)

    case_path, case_data = load_case(args.case, cases_root=args.cases_root)
    case_name = str(case_data.get("case", {}).get("name") or case_path.stem)

    baseline_plan_md, baseline_project_dir = load_baseline(case_name, baselines_root=args.baselines_root)
    plan_md = args.plan_md.read_text(encoding="utf-8") if args.plan_md else baseline_plan_md
    project_dir = args.project_dir.resolve() if args.project_dir else baseline_project_dir

    errors = validate_case(case_data, plan_md=plan_md, project_dir=project_dir)

    if errors:
        print(f"[FAIL] {case_name} ({len(errors)} issue(s))")
        for error in errors:
            print(f" - {error}")
        return 1

    print(f"[PASS] {case_name}")
    return 0


def _validate_plan(plan_md: str | None, expected: dict) -> list[str]:
    plan_expectations = expected.get("plan", {})
    step_expectations = expected.get("steps", {})

    if not plan_expectations and not step_expectations:
        return []
    if plan_md is None:
        return ["Missing baseline plan.md for plan validation"]

    errors: list[str] = []
    has_mermaid = extract_mermaid_block(plan_md) is not None

    expected_has_mermaid = plan_expectations.get("has_mermaid_diagram")
    if expected_has_mermaid is True and not has_mermaid:
        errors.append("Expected a Mermaid diagram in the plan, but none was found")
    if expected_has_mermaid is False and has_mermaid:
        errors.append("Did not expect a Mermaid diagram in the plan, but one was found")

    if has_mermaid:
        stage_count = count_plan_stages(plan_md)
        bounds = plan_expectations.get("stage_count", {})
        minimum = bounds.get("min")
        maximum = bounds.get("max")
        if minimum is not None and stage_count < minimum:
            errors.append(f"Plan has {stage_count} stages, below expected minimum {minimum}")
        if maximum is not None and stage_count > maximum:
            errors.append(f"Plan has {stage_count} stages, above expected maximum {maximum}")

        errors.extend(check_plan_graph(plan_md))

        steps = extract_plan_steps(plan_md)
        for step_id in step_expectations.get("must_include", []):
            if step_id not in steps:
                errors.append(f"Plan is missing required step {step_id!r}")
        for step_id in step_expectations.get("must_exclude", []):
            if step_id in steps:
                errors.append(f"Plan includes excluded step {step_id!r}")
    else:
        bounds = plan_expectations.get("stage_count", {})
        if bounds.get("min") not in (None, 0) or bounds.get("max") not in (None, 0):
            errors.append("Stage-count assertions require a Mermaid plan graph")

    normalized_plan = plan_md.lower()
    for phrase in plan_expectations.get("must_mention", []):
        if phrase.lower() not in normalized_plan:
            errors.append(f"Plan is missing required phrase {phrase!r}")

    return errors


def _validate_project(project_dir: Path | None, expected: dict) -> list[str]:
    project_expectations = expected.get("project", {})
    if not project_expectations:
        return []

    if project_dir is None:
        if project_expectations.get("allow_missing"):
            return []
        return ["Missing baseline project/ directory for structural output validation"]

    failures: list[str] = []
    for result in check_project(project_dir, project_expectations):
        if result.ok:
            continue
        location = f" [{result.path}]" if result.path else ""
        failures.append(f"{result.check}{location}: {result.detail}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
