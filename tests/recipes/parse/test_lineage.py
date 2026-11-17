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
