from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

REFERENCE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "nemotron"
    / "steps"
    / "byob"
    / "references"
)
AUTHORING_DOCS = (
    REFERENCE_ROOT / "bfcl-authoring-user-guide.md",
    REFERENCE_ROOT / "bfcl-authoring-cache-retention.md",
    REFERENCE_ROOT / "bfcl-authoring-revocation.md",
    REFERENCE_ROOT / "bfcl-mcp-user-guide.md",
)
SUPPORT_MATRICES = (
    REFERENCE_ROOT / "bfcl-authoring-support-matrix.md",
    REFERENCE_ROOT / "bfcl-mcp-support-matrix.md",
)
SMOKE_BLOCK = re.compile(
    r"<!-- doc-smoke: (?P<name>[a-z0-9-]+) -->\s*"
    r"```(?:bash|shell)\n(?P<command>.*?)\n```",
    re.DOTALL,
)
SHELL_BLOCK = re.compile(r"```(?:bash|shell)\n.*?\n```", re.DOTALL)
TEST_LINK = re.compile(r"\((?P<path>\.\./[^)\s]*test_[^)\s]*\.py)\)")

ACCEPTANCE_MATRIX = REFERENCE_ROOT / "bfcl-workflow-acceptance-matrix.md"
ACCEPTANCE_ROW = re.compile(r"^\| (?P<id>AC-\d+) \| (?P<criterion>.+?) \| (?P<tests>.+?) \|$")
OWNING_TEST = re.compile(r"`(?P<file>test_bfcl_[a-z0-9_]+\.py)::(?P<name>test_[a-z0-9_]+)`")
TEST_ROOT = Path(__file__).resolve().parent


def test_every_authoring_cli_example_is_an_executable_smoke_case() -> None:
    cases: list[tuple[str, str]] = []
    for path in AUTHORING_DOCS:
        text = path.read_text(encoding="utf-8")
        matches = list(SMOKE_BLOCK.finditer(text))
        assert len(matches) == len(SHELL_BLOCK.findall(text)), (
            f"{path.name} has an unregistered shell example"
        )
        cases.extend((match["name"], match["command"]) for match in matches)

    names = [name for name, _command in cases]
    assert len(names) == len(set(names)), "documentation smoke IDs must be unique"

    for name, command in cases:
        argv = shlex.split(command)
        assert argv[:2] == ["python", "-m"], name
        completed = subprocess.run(
            [sys.executable, *argv[1:]],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, (
            f"{name} failed:\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
        assert "usage:" in completed.stdout.lower(), name


def test_support_claims_name_a_test_or_are_unimplemented() -> None:
    for path in SUPPORT_MATRICES:
        rows = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("| ") and not line.startswith("| ---")
        ]
        assert rows, path.name
        for row in rows[1:]:
            columns = [column.strip() for column in row.strip("|").split("|")]
            status = columns[1].lower()
            evidence = columns[-1].lower()
            if "unimplemented" in status or "unimplemented" in evidence:
                continue
            if any(
                word in status
                for word in ("supported", "implemented", "experimental", "refused")
            ):
                assert "test_" in evidence, f"unsupported claim in {path.name}: {row}"
            else:
                raise AssertionError(
                    f"status must be test-linked or unimplemented in {path.name}: {row}"
                )


def test_documentation_test_links_resolve() -> None:
    docs = (*AUTHORING_DOCS, *SUPPORT_MATRICES)
    for path in docs:
        for match in TEST_LINK.finditer(path.read_text(encoding="utf-8")):
            target = (path.parent / match["path"]).resolve()
            assert target.is_file(), f"broken test link in {path.name}: {match['path']}"


def test_workflow_acceptance_criteria_are_backed_by_named_tests() -> None:
    """Every workflow acceptance criterion must name a test function that exists."""
    rows = [
        match
        for line in ACCEPTANCE_MATRIX.read_text(encoding="utf-8").splitlines()
        if (match := ACCEPTANCE_ROW.match(line)) is not None
    ]
    identifiers = [row["id"] for row in rows]
    assert identifiers, "the acceptance matrix must declare at least one criterion"
    assert len(identifiers) == len(set(identifiers)), "criterion IDs must be unique"
    assert identifiers == [f"AC-{number}" for number in range(1, len(rows) + 1)], (
        "criterion IDs must be a gapless AC-1..AC-N sequence"
    )

    sources: dict[str, str] = {}
    missing: list[str] = []
    for row in rows:
        owners = OWNING_TEST.findall(row["tests"])
        assert owners, f"{row['id']} names no owning test"
        for file_name, test_name in owners:
            path = TEST_ROOT / file_name
            if not path.is_file():
                missing.append(f"{row['id']}: {file_name} does not exist")
                continue
            source = sources.setdefault(file_name, path.read_text(encoding="utf-8"))
            if f"def {test_name}(" not in source:
                missing.append(f"{row['id']}: {file_name}::{test_name} is not defined")
    assert not missing, "acceptance matrix names tests that do not exist: " + "; ".join(
        missing
    )


def test_user_guide_contract_links_resolve() -> None:
    guide = AUTHORING_DOCS[0]
    linked_markdown = re.findall(r"\]\((bfcl-[^)]+\.md)\)", guide.read_text("utf-8"))
    assert linked_markdown
    for relative in linked_markdown:
        assert (guide.parent / relative).is_file(), relative
