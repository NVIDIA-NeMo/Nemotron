from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nemotron.steps.byob.scripts.scaffold_oracle_pack import (
    scaffold_oracle_pack,
)

REPOSITORY = Path(__file__).resolve().parents[3]
MODULE = "nemotron.steps.byob.scripts.validate_oracle_pack"


def _run(config: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            MODULE,
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_standalone_validator_accepts_a_scaffolded_python_pack(
    tmp_path: Path,
) -> None:
    pack = scaffold_oracle_pack(
        tmp_path / "pack",
        domain="Records",
        include_held_out=True,
    )

    output = tmp_path / "validation"
    completed = _run(pack / "validate.yaml", output)

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["gold_eligible"] is True
    assert report["pack_id"] == "records"
    assert report["pack_version"] == "0.1.0"
    assert (output / "resolved-config.yaml").is_file()
    report_paths = list(output.rglob("oracle_validation_report.json"))
    assert len(report_paths) == 1
    assert json.loads(report_paths[0].read_text(encoding="utf-8")) == report


def test_standalone_validator_reports_invalid_config_as_an_error_envelope(
    tmp_path: Path,
) -> None:
    """An unusable config exits 1 with a machine-readable envelope, not a traceback.

    Exit 1 is reserved for "the validator could not reach a verdict". A pack that
    was validated and refused Gold exits 2, so a caller can tell the two apart.
    The envelope goes to stderr so that stdout carries only report documents and
    stays parseable by a caller that pipes it straight into a JSON reader.
    """
    config = tmp_path / "invalid.yaml"
    config.write_text("[]\n", encoding="utf-8")

    completed = _run(config, tmp_path / "validation")

    assert completed.returncode == 1, completed.stdout
    assert completed.stdout == ""
    envelope = json.loads(completed.stderr)
    assert envelope["status"] == "fail"
    assert envelope["error_type"] == "ValueError"
    assert "Config must be a YAML mapping" in envelope["reason"]
