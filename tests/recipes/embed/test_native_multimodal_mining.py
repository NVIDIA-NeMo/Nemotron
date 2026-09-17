# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native mining command contracts; no model or subprocess execution."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.embed.stage1_data_prep import data_prep
from nemotron.recipes.embed.stage1_data_prep.scripts.unroll_pos_docs import unroll_training_data
from nemotron.recipes.retrieval_vl import validate_vl_training_data


@pytest.mark.parametrize("score", [0.2, None, float("nan"), True])
def test_native_mining_output_satisfies_training_contract(tmp_path, monkeypatch, score):
    """Only successful, scored mining can produce the marker required by training."""
    monkeypatch.setattr(data_prep.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=0)))
    output = tmp_path / "train_mined.automodel.json"
    row = {
        "question_id": "q",
        "question": "unchanged",
        "pos_doc": [{"id": "p"}],
        "neg_doc": [{"id": "n", "score": score}],
        "relevance_judgements_complete": False,
    }
    output.write_text(json.dumps({"corpus": {"path": "/corpus"}, "data": [row]}))
    cfg = data_prep.DataPrepConfig(output_dir=tmp_path, model_family="mistral3_vl", hard_negatives_to_mine=1)
    if score is None or isinstance(score, bool) or score != score:
        with pytest.raises(ValueError, match="finite negative score"):
            data_prep.run_mining(cfg, tmp_path / "train.json")
        assert "negative_mining_performed" not in json.loads(output.read_text())["data"][0]
        return
    data_prep.run_mining(cfg, tmp_path / "train.json")
    payload = json.loads(output.read_text())
    payload["data"] = unroll_training_data(payload["data"])
    unrolled = tmp_path / "unrolled.json"
    unrolled.write_text(json.dumps(payload))
    assert validate_vl_training_data(unrolled, required_negatives=1, require_mined_negatives=True) == 1
    assert payload["data"][0] == dict(row, negative_mining_performed=True)
    assert payload["mining"]["relevance_semantics"] == "retrieval-mined candidates remain unjudged"


@pytest.mark.parametrize("use_images,use_text", [(True, True), (True, False), (False, True)])
def test_native_vl_mining_forwards_training_processor_policy(tmp_path, monkeypatch, use_images, use_text):
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(data_prep.subprocess, "run", run)
    output = tmp_path / "train_mined.automodel.json"
    output.write_text('{"corpus": {"path": "/corpus"}, "data": []}')
    cfg = data_prep.DataPrepConfig(
        train_input_file=tmp_path / "train.json",
        sdg_input_path=None,
        output_dir=tmp_path,
        model_family="mistral3_vl",
        base_model="/pinned-vl-model",
        query_max_length=256,
        passage_max_length=8192,
        query_prefix="query: ",
        passage_prefix="passage: ",
        image_longest_edge=1120,
        mining_use_images=use_images,
        use_text_in_document=use_text,
    )
    assert data_prep.run_mining(cfg, cfg.train_input_file) == output
    cmd = run.call_args.args[0]
    expected = {
        "_target_": "nemo_automodel._transformers.mining.CheckpointMiningEncoderConfig",
        "q_max_length": "256",
        "p_max_length": "8192",
        "query_prefix": "query:",
        "passage_prefix": "passage:",
        "image_longest_edge": "1120",
        "use_images": str(use_images).lower(),
        "use_text_in_document": str(use_text).lower(),
    }
    for field, value in expected.items():
        assert cmd[cmd.index(f"--mining.multimodal_encoder.{field}") + 1] == value
    assert cmd[cmd.index("--mining.query_prefix") + 1] == "query: "
    assert cmd[cmd.index("--mining.passage_prefix") + 1] == "passage: "
    assert Path(cmd[cmd.index("--config") - 1]).name == "mine_hard_negatives.py"


def test_text_default_does_not_select_multimodal_processor(tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(data_prep.subprocess, "run", run)
    (tmp_path / "train_mined.automodel.json").write_text('{"data": []}')
    cfg = data_prep.DataPrepConfig(output_dir=tmp_path)
    data_prep.run_mining(cfg, tmp_path / "train.json")
    assert not any(arg.startswith("--mining.multimodal_encoder.") for arg in run.call_args.args[0])


@pytest.mark.parametrize("returncode", [0, 1])
def test_native_mining_failure_does_not_add_success_marker(tmp_path, monkeypatch, returncode):
    """Failed subprocesses and insufficient negatives cannot be labelled successful."""
    monkeypatch.setattr(
        data_prep.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=returncode, stderr="failed"))
    )
    output = tmp_path / "train_mined.automodel.json"
    original = json.dumps({"corpus": {"path": "/corpus"}, "data": [{"neg_doc": []}]})
    output.write_text(original)
    cfg = data_prep.DataPrepConfig(output_dir=tmp_path, model_family="mistral3_vl", hard_negatives_to_mine=1)
    with pytest.raises(SystemExit if returncode else ValueError):
        data_prep.run_mining(cfg, tmp_path / "train.json")
    assert output.read_text() == original


def test_invalid_image_budget_rejected():
    with pytest.raises(ValidationError):
        data_prep.DataPrepConfig(image_longest_edge=0)


def test_vl_profile_native_override_preserves_image_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching the public VL profile to native mining must not select text encoding."""
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(data_prep.subprocess, "run", run)
    monkeypatch.setenv("MISTRAL3_VL_EMBED_MODEL", "nvidia/test-mistral3-vl-embed")
    output = tmp_path / "train_mined.automodel.json"
    output.write_text('{"corpus": {"path": "/corpus"}, "data": []}')
    cfg = load_pydantic_config(
        data_prep.STAGE_PATH / "config/mistral3-vl.yaml",
        ["mining_backend=automodel", f"output_dir={tmp_path}"],
        data_prep.DataPrepConfig,
    )

    assert cfg.model_family == "mistral3_vl"
    assert data_prep.run_mining(cfg, tmp_path / "train.json") == output
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--mining.multimodal_encoder._target_") + 1] == (
        "nemo_automodel._transformers.mining.CheckpointMiningEncoderConfig"
    )
    assert cmd[cmd.index("--mining.multimodal_encoder.use_images") + 1] == "true"
    assert "--mining.multimodal_encoder.processor_name_or_path" not in cmd
    assert "--mining.multimodal_encoder.query_prefix" not in cmd
    assert "--mining.multimodal_encoder.passage_prefix" not in cmd
    assert "--mining.multimodal_encoder.image_longest_edge" not in cmd
    assert "--mining.query_prefix" not in cmd
    assert "--mining.passage_prefix" not in cmd
    assert cmd[cmd.index("--mining.multimodal_encoder.q_max_length") + 1] == "512"
    assert cmd[cmd.index("--mining.multimodal_encoder.p_max_length") + 1] == "8192"
    assert cmd[cmd.index("--mining.tokenizer_force_default") + 1] == "true"


def test_native_vl_mining_preserves_explicit_empty_prefixes(tmp_path, monkeypatch):
    """An empty prompt is an explicit processor override, not an omitted value."""
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(data_prep.subprocess, "run", run)
    output = tmp_path / "train_mined.automodel.json"
    output.write_text('{"corpus": {"path": "/corpus"}, "data": []}')
    cfg = data_prep.DataPrepConfig(
        output_dir=tmp_path,
        model_family="mistral3_vl",
        query_prefix="",
        passage_prefix="",
    )

    data_prep.run_mining(cfg, tmp_path / "train.json")

    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--mining.multimodal_encoder.query_prefix") + 1] == ""
    assert cmd[cmd.index("--mining.multimodal_encoder.passage_prefix") + 1] == ""
