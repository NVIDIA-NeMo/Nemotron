from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from nemotron.steps.byob.scripts.archive_bfcl_release import (
    ReleaseArchiveError,
    build_release_archive,
)


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def test_release_archive_is_deterministic_and_inventories_every_input(
    tmp_path: Path,
) -> None:
    release = _tree(tmp_path / "release", {"run_manifest.json": b"{}\n", "benchmark/data.parquet": b"rows"})
    evaluation = _tree(tmp_path / "eval-1", {"eval_manifest.json": b"{}\n", "cache.jsonl": b"one\n"})
    evidence = _tree(tmp_path / "audit", {"bias_audit_report.json": b"{}\n"})

    first, inventory_path = build_release_archive(
        release_dir=release,
        evaluation_dirs=(evaluation,),
        evidence_dirs=(evidence,),
        output_dir=tmp_path / "out",
        bundle_name="candidate",
    )
    before = first.read_bytes()
    second, _ = build_release_archive(
        release_dir=release,
        evaluation_dirs=(evaluation,),
        evidence_dirs=(evidence,),
        output_dir=tmp_path / "out",
        bundle_name="candidate",
    )

    assert second.read_bytes() == before
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert [item["path"] for item in inventory["files"]] == [
        "release/benchmark/data.parquet",
        "release/run_manifest.json",
        "evaluations/eval-1/cache.jsonl",
        "evaluations/eval-1/eval_manifest.json",
        "evidence/audit/bias_audit_report.json",
    ]
    with tarfile.open(first, "r:gz") as archive:
        assert archive.getnames() == [
            *[item["path"] for item in inventory["files"]],
            "artifact_inventory.json",
        ]
        assert all(member.mtime == 0 for member in archive.getmembers())


def test_release_archive_refuses_symlinks(tmp_path: Path) -> None:
    release = _tree(tmp_path / "release", {"run_manifest.json": b"{}\n"})
    (release / "linked.json").symlink_to(release / "run_manifest.json")

    with pytest.raises(ReleaseArchiveError, match="symlink"):
        build_release_archive(
            release_dir=release,
            evaluation_dirs=(),
            evidence_dirs=(),
            output_dir=tmp_path / "out",
            bundle_name="candidate",
        )
