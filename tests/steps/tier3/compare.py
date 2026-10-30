"""A/B comparison helpers for Tier 3 judge runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json

from .judge import JudgeResult


@dataclass(frozen=True)
class CaseComparison:
    """Per-case comparison between baseline and candidate judge scores."""

    case_name: str
    baseline_overall: float | None
    candidate_overall: float | None
    delta: float | None
    status: str


@dataclass(frozen=True)
class ComparisonReport:
    """Aggregate A/B report for a set of judge results."""

    cases: tuple[CaseComparison, ...]
    improved: int
    regressed: int
    stable: int
    net_score_change: float

    def to_markdown(self) -> str:
        """Render the report as markdown."""
        lines = [
            "## Tier 3 A/B Report",
            "",
            "| Case | Baseline | Candidate | Delta |",
            "|------|----------|-----------|-------|",
        ]

        for case in self.cases:
            baseline = _format_score(case.baseline_overall)
            candidate = _format_score(case.candidate_overall)
            delta = _format_delta(case.delta, case.status)
            lines.append(f"| {case.case_name} | {baseline} | {candidate} | {delta} |")

        lines.extend(
            [
                "",
                "### Summary",
                f"- {self.improved} case(s) improved",
                f"- {self.regressed} case(s) regressed",
                f"- {self.stable} case(s) stable",
                f"- Net score change: {_signed(self.net_score_change)} average",
            ]
        )

        return "\n".join(lines)


def compare_results(
    baseline: dict[str, JudgeResult],
    candidate: dict[str, JudgeResult],
) -> ComparisonReport:
    """Generate A/B diff report."""
    case_names = sorted(set(baseline) | set(candidate))
    rows: list[CaseComparison] = []
    improved = 0
    regressed = 0
    stable = 0
    comparable_deltas: list[float] = []

    for case_name in case_names:
        baseline_result = baseline.get(case_name)
        candidate_result = candidate.get(case_name)

        baseline_score = baseline_result.overall if baseline_result else None
        candidate_score = candidate_result.overall if candidate_result else None

        if baseline_score is None or candidate_score is None:
            status = "missing"
            delta = None
        else:
            delta = round(candidate_score - baseline_score, 2)
            comparable_deltas.append(delta)
            if delta > 0:
                status = "improved"
                improved += 1
            elif delta < 0:
                status = "regressed"
                regressed += 1
            else:
                status = "stable"
                stable += 1

        rows.append(
            CaseComparison(
                case_name=case_name,
                baseline_overall=baseline_score,
                candidate_overall=candidate_score,
                delta=delta,
                status=status,
            )
        )

    net_score_change = round(sum(comparable_deltas) / len(comparable_deltas), 2) if comparable_deltas else 0.0
    return ComparisonReport(
        cases=tuple(rows),
        improved=improved,
        regressed=regressed,
        stable=stable,
        net_score_change=net_score_change,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compare two Tier 3 judge result files")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline JSON results")
    parser.add_argument("--candidate", type=Path, required=True, help="Path to candidate JSON results")
    args = parser.parse_args(argv)

    baseline = _load_results(args.baseline)
    candidate = _load_results(args.candidate)
    report = compare_results(baseline, candidate)
    print(report.to_markdown())
    return 0


def _load_results(path: Path) -> dict[str, JudgeResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if _looks_like_judge_result(data):
        return {path.stem: JudgeResult.from_dict(data)}

    results: dict[str, JudgeResult] = {}
    for case_name, value in dict(data).items():
        if not isinstance(value, dict):
            raise ValueError(f"Expected result object for case {case_name!r} in {path}")
        results[str(case_name)] = JudgeResult.from_dict(value)
    return results


def _looks_like_judge_result(data: object) -> bool:
    return isinstance(data, dict) and "scores" in data and "reasoning" in data


def _format_score(score: float | None) -> str:
    return "—" if score is None else f"{score:.2f}"


def _format_delta(delta: float | None, status: str) -> str:
    if delta is None:
        return "n/a ⚪"
    if status == "improved":
        return f"{_signed(delta)} 🟢"
    if status == "regressed":
        return f"{_signed(delta)} 🔴"
    return "= ⚪"


def _signed(value: float) -> str:
    return f"{value:+.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
