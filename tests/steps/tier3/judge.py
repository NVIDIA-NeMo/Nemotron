"""Tier 3 LLM-as-judge helpers for nemotron-customize outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import re
import sys


DEFAULT_RUBRIC_PATH = "tests/steps/tier3/rubric.md"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
CRITERIA = (
    "Correctness",
    "Completeness",
    "Readability",
    "Runability",
    "Forkability",
)


@dataclass(frozen=True)
class JudgeResult:
    """Structured judge output for one generated project."""

    scores: dict[str, int]
    overall: float
    reasoning: dict[str, str]
    model: str
    timestamp: str

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeResult":
        """Build a JudgeResult from stored JSON data."""
        scores = {str(key): int(value) for key, value in dict(data.get("scores", {})).items()}
        reasoning = {str(key): str(value) for key, value in dict(data.get("reasoning", {})).items()}
        overall = float(data.get("overall", _compute_overall(scores)))
        model = str(data.get("model", DEFAULT_MODEL))
        timestamp = str(data.get("timestamp", ""))
        return cls(
            scores=scores,
            overall=overall,
            reasoning=reasoning,
            model=model,
            timestamp=timestamp,
        )


def judge_output(
    user_request: str,
    generated_plan: str,
    generated_project_files: dict[str, str],
    rubric_path: str = DEFAULT_RUBRIC_PATH,
    model: str = DEFAULT_MODEL,
) -> JudgeResult:
    """Score a generated project against the quality rubric."""
    try:
        import anthropic
    except ModuleNotFoundError as exc:
        raise RuntimeError("The 'anthropic' package is required to run Tier 3 judging.") from exc

    resolved_rubric = _resolve_repo_path(rubric_path)
    rubric_text = resolved_rubric.read_text(encoding="utf-8")
    prompt = _build_judge_prompt(
        rubric_text=rubric_text,
        user_request=user_request,
        generated_plan=generated_plan,
        generated_project_files=generated_project_files,
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = _message_text(response)
    parsed = _parse_judge_payload(response_text)

    scores = _normalize_scores(parsed.get("scores", {}))
    reasoning = _normalize_reasoning(parsed.get("reasoning", {}))
    overall = _compute_overall(scores)

    return JudgeResult(
        scores=scores,
        overall=overall,
        reasoning=reasoning,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def load_project_files(project_root: Path) -> dict[str, str]:
    """Load a generated project directory into a path-to-content mapping."""
    project_root = project_root.resolve()
    files: dict[str, str] = {}

    for path in sorted(project_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(project_root).as_posix()
        files[relative] = path.read_text(encoding="utf-8", errors="replace")

    return files


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run Tier 3 LLM-as-judge scoring")
    parser.add_argument("--plan", type=Path, required=True, help="Path to generated plan markdown")
    parser.add_argument("--project", type=Path, required=True, help="Path to generated project directory")
    parser.add_argument("--request", required=True, help="Original user request text")
    parser.add_argument("--rubric-path", default=DEFAULT_RUBRIC_PATH, help="Path to rubric.md")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge model name")
    args = parser.parse_args(argv)

    try:
        plan_md = args.plan.read_text(encoding="utf-8")
        project_files = load_project_files(args.project)
        result = judge_output(
            user_request=args.request,
            generated_plan=plan_md,
            generated_project_files=project_files,
            rubric_path=args.rubric_path,
            model=args.model,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate


def _build_judge_prompt(
    *,
    rubric_text: str,
    user_request: str,
    generated_plan: str,
    generated_project_files: dict[str, str],
) -> str:
    project_sections: list[str] = []
    for relative_path, content in sorted(generated_project_files.items()):
        project_sections.append(f"--- FILE: {relative_path} ---\n{content.rstrip()}\n")

    project_blob = "\n".join(project_sections) if project_sections else "(no project files provided)"

    return (
        "You are grading a generated ML pipeline project.\n\n"
        "Apply the rubric below and return JSON only using the required schema.\n\n"
        "=== RUBRIC ===\n"
        f"{rubric_text.strip()}\n\n"
        "=== USER REQUEST ===\n"
        f"{user_request.strip()}\n\n"
        "=== GENERATED PLAN ===\n"
        f"{generated_plan.strip()}\n\n"
        "=== GENERATED PROJECT FILES ===\n"
        f"{project_blob.strip()}\n"
    )


def _message_text(response: object) -> str:
    parts = getattr(response, "content", None)
    if not parts:
        return str(response)

    text_chunks: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str):
            text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def _parse_judge_payload(text: str) -> dict:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Judge model returned an empty response.")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = _extract_json_object(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Judge response did not contain a JSON object.")

    return parsed


def _extract_json_object(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()

    for candidate in candidates:
        snippet = candidate.strip()
        if not snippet:
            continue
        for index, char in enumerate(snippet):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(snippet[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise ValueError("Could not parse JSON from judge response.")


def _normalize_scores(raw_scores: dict) -> dict[str, int]:
    normalized: dict[str, int] = {}

    for criterion in CRITERIA:
        if criterion not in raw_scores:
            raise ValueError(f"Judge response is missing a score for {criterion}.")
        value = int(raw_scores[criterion])
        if value < 1 or value > 5:
            raise ValueError(f"Judge score for {criterion} must be between 1 and 5, got {value}.")
        normalized[criterion] = value

    return normalized


def _normalize_reasoning(raw_reasoning: dict) -> dict[str, str]:
    normalized: dict[str, str] = {}

    for criterion in CRITERIA:
        if criterion not in raw_reasoning:
            raise ValueError(f"Judge response is missing reasoning for {criterion}.")
        normalized[criterion] = str(raw_reasoning[criterion]).strip()

    return normalized


def _compute_overall(scores: dict[str, int]) -> float:
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 2)


if __name__ == "__main__":
    raise SystemExit(main())
