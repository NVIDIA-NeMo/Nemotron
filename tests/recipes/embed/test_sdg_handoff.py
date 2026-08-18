"""Allium-derived integration tests for the Stage 0 to Stage 1 handoff."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from data_designer.engine.storage.artifact_storage import ArtifactStorage

pytest.importorskip("data_designer_retrieval_sdg")

from nemotron.recipes.embed.sdg_manifest import (
    GENERATION_MANIFEST_FILENAME,
    resolve_generation_input,
    write_generation_manifest,
)
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig, run_sdg
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
from nemotron.recipes.embed.stage1_data_prep.plugin_adapter import build_conversion_config


def _touch_jsonl(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_default_preparation_selects_first_generation_result(tmp_path: Path) -> None:
    stage0 = SDGConfig(artifact_root=tmp_path, dataset_name="nv_pp_random")
    output_path = _touch_jsonl(stage0.output_dir / "nv_pp_random.jsonl")
    manifest_path = write_generation_manifest(
        output_dir=stage0.output_dir,
        output_path=output_path,
        dataset_name="nv_pp_random",
    )

    stage1 = DataPrepConfig(artifact_root=tmp_path)
    package_config = build_conversion_config(stage1)

    assert stage1.sdg_input_path == manifest_path
    assert package_config.input_path == output_path.resolve()


def test_repeated_fresh_generation_replaces_latest_handoff(tmp_path: Path) -> None:
    stage0 = SDGConfig(artifact_root=tmp_path, dataset_name="nv_pp_random", resume="never")
    first_output = _touch_jsonl(stage0.output_dir / "nv_pp_random.jsonl")
    manifest_path = write_generation_manifest(
        output_dir=stage0.output_dir,
        output_path=first_output,
        dataset_name="nv_pp_random",
    )

    prior_artifacts = stage0.artifact_path / stage0.dataset_name
    prior_artifacts.mkdir(parents=True)
    (prior_artifacts / "metadata.json").write_text("{}", encoding="utf-8")
    storage = ArtifactStorage(
        artifact_path=stage0.artifact_path,
        dataset_name=stage0.dataset_name,
        resume=stage0.resume,
    )
    repeated_output = _touch_jsonl(stage0.output_dir / f"{storage.resolved_dataset_name}.jsonl")

    write_generation_manifest(
        output_dir=stage0.output_dir,
        output_path=repeated_output,
        dataset_name=storage.resolved_dataset_name,
    )

    assert resolve_generation_input(manifest_path) == repeated_output.resolve()


@pytest.mark.parametrize("resume", ["always", "if_possible"])
def test_resumed_generation_publishes_exact_result(tmp_path: Path, resume: str) -> None:
    stage0 = SDGConfig(artifact_root=tmp_path, dataset_name="nv_pp_random", resume=resume)
    output_path = _touch_jsonl(stage0.output_dir / "nv_pp_random.jsonl")

    manifest_path = write_generation_manifest(
        output_dir=stage0.output_dir,
        output_path=output_path,
        dataset_name="nv_pp_random",
    )

    assert resolve_generation_input(manifest_path) == output_path.resolve()


def test_explicit_preparation_input_remains_authoritative(tmp_path: Path) -> None:
    generated_output = _touch_jsonl(tmp_path / "stage0_sdg" / "generated.jsonl")
    write_generation_manifest(
        output_dir=generated_output.parent,
        output_path=generated_output,
        dataset_name="generated",
    )
    explicit_input = _touch_jsonl(tmp_path / "external" / "explicit.jsonl")

    package_config = build_conversion_config(
        DataPrepConfig(
            artifact_root=tmp_path,
            sdg_input_path=explicit_input,
        )
    )

    assert package_config.input_path == explicit_input.resolve()


def test_generation_manifest_rejects_missing_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage0_sdg"
    with pytest.raises(FileNotFoundError, match="Generation output"):
        write_generation_manifest(
            output_dir=output_dir,
            output_path=output_dir / "missing.jsonl",
            dataset_name="missing",
        )

    assert not (output_dir / GENERATION_MANIFEST_FILENAME).exists()


def test_generation_manifest_rejects_empty_dataset_name(tmp_path: Path) -> None:
    output_dir = tmp_path / "stage0_sdg"
    output_path = _touch_jsonl(output_dir / "generated.jsonl")

    with pytest.raises(ValueError, match="dataset_name"):
        write_generation_manifest(
            output_dir=output_dir,
            output_path=output_path,
            dataset_name="",
        )

    assert not (output_dir / GENERATION_MANIFEST_FILENAME).exists()


def test_default_manifest_filename_is_stable() -> None:
    assert GENERATION_MANIFEST_FILENAME == "generation_result.json"


def test_successful_generation_publishes_exact_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from nemotron.recipes.embed.stage0_sdg import plugin_adapter

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "document.txt").write_text("retrieval content " * 10, encoding="utf-8")
    output_path = _touch_jsonl(tmp_path / "stage0_sdg" / "resolved-name.jsonl")
    result = SimpleNamespace(
        output_path=output_path,
        dataset_name="resolved-name",
        num_records=1,
        resolved_config_path=None,
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setattr(plugin_adapter, "execute_generation", lambda *_args: result)

    returned_path = run_sdg(
        SDGConfig(
            artifact_root=tmp_path,
            corpus_dir=str(corpus_dir),
            dataset_name="requested-name",
        )
    )

    assert returned_path == output_path
    assert resolve_generation_input(tmp_path / "stage0_sdg" / GENERATION_MANIFEST_FILENAME) == output_path.resolve()
