from __future__ import annotations

import shutil
from pathlib import Path

from nemotron.steps.byob.scripts.validate import _discover_families, validate_skill_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
BYOB_DIR = REPO_ROOT / "src" / "nemotron" / "steps" / "byob"


def test_checked_in_byob_assets_validate() -> None:
    assert validate_skill_dir(BYOB_DIR) == []


def test_validator_reports_missing_family_asset(tmp_path: Path) -> None:
    skill_dir = tmp_path / "byob"
    shutil.copytree(BYOB_DIR, skill_dir)
    (skill_dir / "mcq" / "config" / "tiny.yaml").unlink()

    errors = validate_skill_dir(skill_dir)

    assert "missing required file: mcq/config/tiny.yaml" in errors


def test_validator_discovers_additional_family(tmp_path: Path) -> None:
    skill_dir = tmp_path / "byob"
    shutil.copytree(BYOB_DIR, skill_dir)
    # Use a synthetic third-family name because BFCL is already registered.
    shutil.copytree(skill_dir / "mcq", skill_dir / "gsm8k")
    shutil.copytree(
        skill_dir / "runtime" / "benchmark_families" / "mcq",
        skill_dir / "runtime" / "benchmark_families" / "gsm8k",
    )

    assert validate_skill_dir(skill_dir) == []
    assert _discover_families(skill_dir) == ["bfcl", "gsm8k", "mcq"]
