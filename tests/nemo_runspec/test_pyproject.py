"""Tests for temporary pyproject generation."""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

from nemo_runspec._pyproject import _write_temp_pyproject


def test_write_temp_pyproject_preserves_extras_sources_and_indexes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    stage_dir = repo_root / "src" / "nemotron" / "recipes" / "embed" / "stage4_export"

    with open(stage_dir / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    temp_dir = _write_temp_pyproject(data, stage_dir, ["torch", "transformer-engine"])
    try:
        with open(temp_dir / "pyproject.toml", "rb") as f:
            temp = tomllib.load(f)

        optional_deps = temp["project"]["optional-dependencies"]
        assert "tensorrt" in optional_deps

        uv = temp["tool"]["uv"]
        assert uv["sources"]["torch"] == [
            {"index": "pytorch-cu129", "marker": "sys_platform == 'linux'"}
        ]
        assert {index["name"] for index in uv["index"]} >= {"pytorch-cu129"}
        assert "torch" in uv["exclude-dependencies"]
        assert "nvidia-pytriton" in uv["exclude-dependencies"]
        assert "transformer-engine" in uv["exclude-dependencies"]
    finally:
        shutil.rmtree(temp_dir)
