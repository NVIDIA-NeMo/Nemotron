from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.endpoint import (
    load_endpoint_config,
)
from nemotron.steps.byob.scripts.scaffold_oracle_pack import (
    OPERATOR_REFERENCE_SLUGS,
    scaffold_oracle_pack,
)

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"


def test_python_scaffold_contains_a_complete_runnable_pack(tmp_path: Path) -> None:
    target = scaffold_oracle_pack(
        tmp_path / "inventory-pack",
        domain="Inventory Service",
        include_held_out=True,
    )

    assert {path.name for path in target.iterdir()} == {
        "README.md",
        "assertions.py",
        "backend.py",
        "fixtures.json",
        "held_out.yaml",
        "manifest.yaml",
        "task_templates.yaml",
        "tools.json",
        "validate.yaml",
        "validation_cases.yaml",
    }
    manifest = yaml.safe_load((target / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["pack_id"] == "inventory_service"
    assert manifest["paths"]["backend"] == "backend.py"
    assert manifest["held_out"] == "held_out.yaml"
    assert "held_out" not in manifest["paths"]
    assert "endpoint" not in manifest["paths"]
    fixtures = json.loads((target / "fixtures.json").read_text(encoding="utf-8"))
    assert {row["record_id"] for row in fixtures["records"]} == {
        "REC-001",
        "REC-HELD-OUT-1",
    }
    backend = (target / "backend.py").read_text(encoding="utf-8")
    assert "Return public names that must match tools.json exactly." in backend
    assert "Start one deterministic episode" in backend
    assert "Return a copy of the complete state" in backend
    assert "return expected business failures as data" in backend
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "## Fill order and contracts" in readme
    assert "def list_tools() -> list[str]" in readme
    for reference in OPERATOR_REFERENCE_SLUGS:
        assert f"reference/{reference}.md" in readme
    assert (
        "/tmp/bfcl-inventory_service-validation/"
        "bfcl_inventory_service_starter/stage_cache/oracle_validation_report.json" in readme
    )
    assert "not bundled in the Python wheel" in readme


def test_endpoint_scaffold_is_transport_specific_and_parseable(
    tmp_path: Path,
) -> None:
    target = scaffold_oracle_pack(
        tmp_path / "endpoint-pack",
        domain="Claims",
        transport="endpoint",
    )

    manifest = yaml.safe_load((target / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["paths"]["endpoint"] == "endpoint_config.yaml"
    assert "backend" not in manifest["paths"]
    assert not (target / "backend.py").exists()
    endpoint = load_endpoint_config(
        target / "endpoint_config.yaml",
        allowed_roots=(target,),
    )
    assert endpoint.expected.oracle_id == "claims"
    assert endpoint.base_url == "https://oracle.example.invalid"
    readme = (target / "README.md").read_text(encoding="utf-8")
    assert "GET /v1/metadata" in readme
    assert "GET /v1/conformance" in readme
    assert "placeholder endpoint cannot reach Gold" in readme


def test_scaffold_command_creates_the_requested_pack(tmp_path: Path) -> None:
    target = tmp_path / "command-pack"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemotron.steps.byob.scripts.scaffold_oracle_pack",
            "--domain",
            "Customer Records",
            "--target",
            str(target),
            "--transport",
            "python",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == target
    assert (target / "backend.py").is_file()


def test_scaffold_never_overwrites_an_existing_target(tmp_path: Path) -> None:
    target = scaffold_oracle_pack(tmp_path / "pack", domain="Records")
    marker = target / "owner-note.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="target already exists"):
        scaffold_oracle_pack(target, domain="Replacement")

    assert marker.read_text(encoding="utf-8") == "keep"


def test_manual_recipe_and_reference_pack_readmes_cover_the_runnable_workflow() -> None:
    recipe = (BYOB_ROOT / "patterns" / "create-bfcl-from-oracle-pack.md").read_text(encoding="utf-8")
    assert "eval runner is unwired" not in recipe
    assert "## Python-backend quick start" in recipe
    assert "## Endpoint-backed quick start" in recipe
    assert "scaffold_oracle_pack" in recipe
    assert "validate_oracle_pack" in recipe
    family_readme = (BYOB_ROOT / "bfcl" / "README.md").read_text(encoding="utf-8")
    assert "Until the eval runner lands" not in family_readme

    # What is under test is that an operator can find the file set and the commands
    # for each reference pack, not which file holds them. A frozen pack keeps its
    # operator notes outside the pack directory so that editing the notes is not an
    # edit to the pack, so the notes are read from wherever that pack keeps them.
    for docs in (
        (BYOB_ROOT / "data" / "tiny_oracle_pack" / "README.md",),
        (
            BYOB_ROOT / "data" / "banking_vn_oracle_pack" / "README.md",
            BYOB_ROOT / "references" / "bfcl-banking-vn-pack-operations.md",
        ),
    ):
        notes = "\n".join(doc.read_text(encoding="utf-8") for doc in docs)
        for name in (
            "manifest.yaml",
            "tools.json",
            "fixtures.json",
            "task_templates.yaml",
            "assertions.py",
            "validation_cases.yaml",
        ):
            assert name in notes
        assert "validate_oracle_pack" in notes
        assert "--stage prepare" in notes
        assert "--stage generate" in notes
