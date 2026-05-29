"""Historical score ratchet helpers for Tier 3 judge runs."""

from __future__ import annotations

from pathlib import Path
import json


DEFAULT_RATCHET_PATH = "tests/steps/tier3/ratchet.json"


def check_ratchet(
    case_name: str,
    current_score: float,
    ratchet_path: str = DEFAULT_RATCHET_PATH,
    tolerance: float = 0.3,
) -> bool:
    """Returns True if score is acceptable (not regressed beyond tolerance)."""
    path = _resolve_repo_path(ratchet_path)
    scores = load_ratchet(path)
    historical_best = scores.get(case_name)

    if historical_best is None:
        scores[case_name] = float(current_score)
        save_ratchet(path, scores)
        return True

    if float(current_score) < float(historical_best) - float(tolerance):
        return False

    if float(current_score) > float(historical_best):
        scores[case_name] = float(current_score)
        save_ratchet(path, scores)

    return True


def load_ratchet(path: str | Path = DEFAULT_RATCHET_PATH) -> dict[str, float]:
    """Load the ratchet score store from disk."""
    resolved = _resolve_repo_path(path)
    if not resolved.exists():
        return {}

    data = json.loads(resolved.read_text(encoding="utf-8") or "{}")
    return {str(key): float(value) for key, value in dict(data).items()}


def save_ratchet(path: str | Path, scores: dict[str, float]) -> None:
    """Persist ratchet scores to disk."""
    resolved = _resolve_repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[3] / candidate
