"""Source resolution and immutable comparison inputs."""

import json
import os
import sys
from types import SimpleNamespace

import pytest

from nemotron.recipes.parse.dental.comparison import uplift
from nemotron.recipes.parse.training.lineage import model_reference, verify_model
from nemotron.recipes.parse.training.model import prepare_model, resolve_model


def test_custom_hub_source_records_the_resolved_revision(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    config = dict(is_encoder_decoder=False, decoder=dict(causal_vision_prefix=True))
    (snapshot / "config.json").write_text(json.dumps(config))
    (snapshot / "model.safetensors").write_bytes(b"customer model")
    calls = []

    def download(repo_id, revision):
        calls.append((repo_id, revision))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    output = prepare_model("customer/custom-parse", tmp_path / "view", revision="release")
    record = json.loads((output / "model-source.json").read_text())
    assert calls == [("customer/custom-parse", "release")]
    assert record["source"] == "customer/custom-parse"
    assert record["revision"] == "a" * 40
    assert record["checkpoint"] == str(snapshot)
    assert (output / "model.safetensors").resolve() == snapshot / "model.safetensors"


def test_local_source_and_content_checks_do_not_depend_on_default_model_or_mtime(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"original")
    assert resolve_model(tmp_path) == (tmp_path, None)
    reference = model_reference(tmp_path)
    verify_model(reference)
    stat = weights.stat()
    weights.write_bytes(b"modified")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="changed"):
        verify_model(reference)
    with pytest.raises(ValueError, match="local checkpoints"):
        resolve_model(tmp_path, revision="ignored")
    with pytest.raises(FileNotFoundError):
        resolve_model(tmp_path / "missing")


def test_uplift_handles_zero_cer_and_regression_without_dividing_by_zero():
    before = dict(pass_at_k={"1": 0.5}, metrics=dict(text_cer=0.0))
    after = dict(pass_at_k={"1": 0.25}, metrics=dict(text_cer=0.1))
    assert uplift(before, after) == dict(pass_at_1_gain_pp=-25, text_cer_reduction_pp=-10,
                                        text_cer_relative_reduction=None)


@pytest.mark.parametrize("cross_filesystem", [False, True])
def test_persistent_base_survives_removed_hub_cache(tmp_path, monkeypatch, cross_filesystem):
    import errno
    import shutil

    from nemotron.recipes.parse.training import model

    cache = tmp_path / "cache"
    snapshot = cache / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    originals = {
        "config.json": json.dumps(dict(is_encoder_decoder=False, decoder=dict(causal_vision_prefix=True))).encode(),
        "model.safetensors": b"unchanged upstream weights",
        "tokenizer_config.json": b'{"eos_token": "</s>"}',
        "preprocessor_config.json": b'{"size": 2048}',
        "hf_native.py": b"# original remote model code\n",
    }
    for name, content in originals.items():
        (blobs / name).write_bytes(content)
        (snapshot / name).symlink_to(blobs / name)
    calls = []

    def download(repo_id, revision):
        calls.append((repo_id, revision))
        return str(snapshot)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    if cross_filesystem:
        def cannot_link(*args):
            raise OSError(errno.EXDEV, "Cross-device link")
        monkeypatch.setattr(model.os, "link", cannot_link)
    persistent, revision = model.materialize_base_model("customer/parse", tmp_path / "sft/base-model", "release")
    assert calls == [("customer/parse", "release")]
    assert revision == "b" * 40
    assert {p.name: p.read_bytes() for p in persistent.iterdir()} == originals
    assert not any(p.is_symlink() for p in persistent.iterdir())
    assert {p.name: p.read_bytes() for p in blobs.iterdir()} == originals
    reference = model_reference(persistent, source="customer/parse", revision=revision)
    adapted = prepare_model(persistent, tmp_path / "sft/initial-model")
    assert json.loads((persistent / "config.json").read_text()).get("parse_train_scope") is None
    assert json.loads((adapted / "config.json").read_text())["parse_train_scope"] == "full"
    shutil.rmtree(cache)
    verify_model(reference)
    assert (adapted / "model.safetensors").read_bytes() == originals["model.safetensors"]
    assert (adapted / "preprocessor_config.json").read_bytes() == originals["preprocessor_config.json"]
    assert reference["source"] == "customer/parse" and reference["revision"] == revision


@pytest.mark.parametrize("via_symlink", [False, True])
def test_persistent_base_rejects_output_inside_source_before_copying(tmp_path, via_symlink):
    from nemotron.recipes.parse.training.model import materialize_base_model

    source = tmp_path / "model"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"untouched source")
    parent = source / "sft"
    parent.mkdir()
    if via_symlink:
        alias = tmp_path / "model-alias"
        alias.symlink_to(source, target_is_directory=True)
        parent = alias / "sft"
    output = parent / "base-model"
    with pytest.raises(ValueError, match="outside the source model"):
        materialize_base_model(source, output)
    assert not output.exists()
    assert (source / "model.safetensors").read_bytes() == b"untouched source"
