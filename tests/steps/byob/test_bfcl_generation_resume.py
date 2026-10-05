"""Verified checkpoint/resume contract for BFCL generation."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemotron.steps.byob.runtime.benchmark_families.bfcl.checkpoint import (
    CANONICAL_STAGES,
    CheckpointError,
)
from nemotron.steps.byob.runtime.benchmark_families.bfcl.pipeline import generate_bfcl
from nemotron.steps.byob.scripts.runtime import run_byob

BYOB_ROOT = Path(__file__).resolve().parents[3] / "src" / "nemotron" / "steps" / "byob"


def _config(tmp_path: Path) -> Path:
    value = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    value["output_dir"] = str(tmp_path / "output")
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def _mutable_pack_config(tmp_path: Path) -> tuple[Path, Path]:
    pack = tmp_path / "tiny_oracle_pack"
    shutil.copytree(BYOB_ROOT / "data" / "tiny_oracle_pack", pack)
    value = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
    value["output_dir"] = str(tmp_path / "output")
    value["oracle_pack"] = {"manifest_path": str(pack / "manifest.yaml")}
    value["oracle_runtime"]["allowed_roots"] = [str(tmp_path)]
    path = tmp_path / "tiny-mutable.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path, pack


def test_resume_from_expected_trace_reconstructs_equivalent_output(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    config = _config(tmp_path)
    first = generate_bfcl(config)
    first_rows = pq.read_table(first).to_pylist()
    checkpoints = first.parent / "stage_cache" / "checkpoints"
    enabled = tuple(name for name in CANONICAL_STAGES if name not in {"surface_quality", "dedup_balancing"})
    assert [name for name in enabled if (checkpoints / name / "manifest.json").is_file()] == list(enabled)

    resumed = generate_bfcl(config, skip_until="expected_trace")
    assert pq.read_table(resumed).to_pylist() == first_rows


def _io_cache_entry(request_hash: str) -> str:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.row_schema import canonical_json

    response = {"text": request_hash}
    digest = hashlib.sha256(canonical_json(response).encode()).hexdigest()
    entry = {
        "request_hash": request_hash,
        "response": response,
        "response_hash": f"sha256:{digest}",
    }
    return json.dumps(entry, sort_keys=True) + "\n"


def test_resume_preserves_append_only_caches_but_drops_stale_stage_output(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    cache = generate_bfcl(config).parent / "stage_cache"
    caches = {
        cache / "reference_profile_io_cache.jsonl": _io_cache_entry("profile"),
        cache / "surface_judge_io_cache.jsonl": _io_cache_entry("judge"),
        cache / "paraphrase_io_cache.jsonl": _io_cache_entry("paraphrase"),
    }
    for path, text in caches.items():
        path.write_text(text, encoding="utf-8")
    # Stage 12 rewrites its scan only when a held-out policy declares one, so a
    # leftover copy proves stale output is still removed rather than inherited.
    stale = cache / "held_out_scan.json"
    stale.write_text("{}\n", encoding="utf-8")

    # Resuming before render is the case that used to re-spend paraphrase tokens.
    generate_bfcl(config, skip_until="expand")

    assert {path: path.read_text(encoding="utf-8") for path in caches} == caches
    assert not stale.exists()


def test_tampered_checkpoint_fails_closed_and_removes_old_publication(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = generate_bfcl(config)
    checkpoint = benchmark.parent / "stage_cache" / "checkpoints" / "executable_replay"
    state = checkpoint / "state.json"
    state.write_text(state.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(CheckpointError, match="metadata mismatch"):
        generate_bfcl(config, skip_until="final_output")

    assert not benchmark.exists()
    assert not (benchmark.parent / "run_manifest.json").exists()


def test_resume_rejects_config_drift_and_disabled_or_unknown_stages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    benchmark = generate_bfcl(config)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["random_seed"] = int(value.get("random_seed") or 0) + 1
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(CheckpointError, match="incompatible"):
        generate_bfcl(config, skip_until="final_output")
    assert not benchmark.exists()

    with pytest.raises(CheckpointError, match="disabled"):
        generate_bfcl(config, skip_until="surface_quality")
    with pytest.raises(CheckpointError, match="unknown"):
        generate_bfcl(config, skip_until="not-a-stage")


def test_config_identity_ignores_symlinks_in_the_spelling_of_a_root(tmp_path: Path) -> None:
    """A host that reaches output_dir through a symlink must not fork the identity.

    macOS states /tmp for a directory it resolves to /private/tmp, so a run generated
    there and resumed on Linux would otherwise be rejected as a different config.
    """
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.config import BfclConfig
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages.final_output import (
        generation_config_hash,
    )

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    hashes = set()
    for name, root in (("direct", real), ("through-link", link)):
        value = yaml.safe_load((BYOB_ROOT / "bfcl" / "config" / "tiny.yaml").read_text(encoding="utf-8"))
        value["output_dir"] = str(root / "output")
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        hashes.add(generation_config_hash(BfclConfig.from_yaml(path)))

    assert len(hashes) == 1


def test_resume_rejects_pack_and_pipeline_identity_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, pack = _mutable_pack_config(tmp_path)
    generate_bfcl(config)
    prompt = pack / "README.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "\nDrift.\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="incompatible"):
        generate_bfcl(config, skip_until="final_output")

    # Rebuild a clean chain, then make the running code identity differ from it.
    prompt.write_text(prompt.read_text(encoding="utf-8").replace("\nDrift.\n", ""), encoding="utf-8")
    generate_bfcl(config)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import checkpoint

    real_metadata = checkpoint.runtime_metadata

    def drifted_metadata():  # type: ignore[no-untyped-def]
        metadata = real_metadata()
        return {**metadata, "pipeline_source_hash": "sha256:" + "0" * 64}

    monkeypatch.setattr(checkpoint, "runtime_metadata", drifted_metadata)
    with pytest.raises(CheckpointError, match="incompatible"):
        generate_bfcl(config, skip_until="final_output")


def test_runtime_metadata_does_not_stamp_an_unrelated_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import runtime_metadata

    unrelated_repo = tmp_path / "application"
    installed_package = unrelated_repo / ".venv" / "site-packages" / "byob"
    installed_package.mkdir(parents=True)
    (installed_package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(
        runtime_metadata,
        "_step_package_root",
        lambda: installed_package,
    )

    def unrelated_git(*arguments: str) -> str | None:
        responses = {
            ("rev-parse", "--is-inside-work-tree"): "true",
            ("rev-parse", "--show-toplevel"): str(unrelated_repo),
            ("rev-parse", "HEAD"): "unrelated-head",
        }
        return responses.get(arguments)

    monkeypatch.setattr(runtime_metadata, "_git", unrelated_git)
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("CI_COMMIT_SHA", raising=False)

    metadata = runtime_metadata.runtime_metadata()
    assert metadata["pipeline_git_sha"] is None
    assert metadata["pipeline_git_dirty"] is None


def test_stage_all_resume_does_not_run_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    generate_bfcl(config)

    from nemotron.steps.byob.scripts import runtime

    def fail_prepare(_path: Path) -> Path:
        raise AssertionError("prepare must not run before a verified resume")

    monkeypatch.setattr(
        runtime,
        "get_family",
        lambda _name: SimpleNamespace(
            prepare_data=fail_prepare,
            generate=generate_bfcl,
            translate=None,
            evaluate=None,
        ),
    )
    result = run_byob(
        config=config,
        stage="all",
        family="bfcl",
        skip_until="final_output",
    )
    assert result is not None
    manifest = json.loads((Path(result).parent / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_config_hash"].startswith("sha256:")


def test_stage_twelve_checkpoint_is_durable_before_publication_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    real_commit = final_output._commit_staged_publication

    def checked_commit(staging_dir: Path, output_dir: Path) -> Path:
        assert not (output_dir / "run_manifest.json").exists()
        assert (output_dir / "stage_cache" / "checkpoints" / "final_output" / "manifest.json").is_file()
        return real_commit(staging_dir, output_dir)

    monkeypatch.setattr(final_output, "_commit_staged_publication", checked_commit)
    benchmark = generate_bfcl(config)
    assert benchmark.is_file()
    assert (benchmark.parent / "run_manifest.json").is_file()


def test_downstream_checkpoint_rejects_modified_inherited_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import checkpoint

    real_write = checkpoint.write_checkpoint

    def write_then_tamper(config_value, stage, state, **kwargs):  # type: ignore[no-untyped-def]
        result = real_write(config_value, stage, state, **kwargs)
        if stage == "render":
            artifact = (
                Path(config_value.output_dir) / config_value.expt_name / "stage_cache" / "paraphrase_rejections.json"
            )
            artifact.write_text(
                artifact.read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
        return result

    monkeypatch.setattr(checkpoint, "write_checkpoint", write_then_tamper)
    with pytest.raises(CheckpointError, match="changed inherited artifacts"):
        generate_bfcl(config)


def test_missing_staged_publication_cannot_receive_a_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    from nemotron.steps.byob.runtime.benchmark_families.bfcl import checkpoint

    real_write = checkpoint.write_checkpoint

    def remove_publication_file(config_value, stage, state, **kwargs):  # type: ignore[no-untyped-def]
        if stage == "final_output":
            publication_paths = kwargs["publication_paths"]
            next(path for path in publication_paths if path.name == "benchmark_raw.parquet").unlink()
        return real_write(config_value, stage, state, **kwargs)

    monkeypatch.setattr(checkpoint, "write_checkpoint", remove_publication_file)
    with pytest.raises(CheckpointError, match="publication artifact is missing"):
        generate_bfcl(config)
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    output_dir = Path(config_value["output_dir"]) / config_value["expt_name"]
    assert not (output_dir / "run_manifest.json").exists()
    assert not (output_dir / "benchmark.parquet").exists()


def test_declared_export_removed_before_checkpoint_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config_value = yaml.safe_load(config.read_text(encoding="utf-8"))
    config_value["exports"]["bfcl_json"] = True
    config.write_text(yaml.safe_dump(config_value), encoding="utf-8")
    from nemotron.steps.byob.runtime.benchmark_families.bfcl.stages import final_output

    real_write = final_output._write_endpoint_manifest_atomic

    def write_then_remove_export(text: str, path: Path, **kwargs) -> None:
        real_write(text, path, **kwargs)
        manifest = json.loads(text)
        export = manifest["exports"]["formats"]["bfcl_json"]
        export_root = path.parent / export["path"]
        next(child for child in export_root.rglob("*") if child.is_file()).unlink()

    monkeypatch.setattr(
        final_output,
        "_write_endpoint_manifest_atomic",
        write_then_remove_export,
    )
    with pytest.raises(CheckpointError, match="export is missing or changed"):
        generate_bfcl(config)
    output_dir = Path(config_value["output_dir"]) / config_value["expt_name"]
    assert not (output_dir / "run_manifest.json").exists()
