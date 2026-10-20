# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The public smoke fixture and commands must satisfy production contracts."""

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from nemo_runspec.config import load_pydantic_config
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig
from nemotron.recipes.embed.stage2_finetune.train import FinetuneConfig as EmbedFinetuneConfig
from nemotron.recipes.embed.stage3_eval.eval import EvalConfig
from nemotron.recipes.embed.stage5_deploy.deploy import DeployConfig
from nemotron.recipes.rerank.stage2_finetune.train import FinetuneConfig as RerankFinetuneConfig

ROOT = Path(__file__).resolve().parents[3]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/retrieval/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_has_all_modalities_and_valid_images_qrels_and_training_paths(tmp_path):
    from PIL import Image

    root = tmp_path / "fixture"
    _module("smoke_fixture").write_fixture(root)
    payload = json.loads((root / "train.json").read_text())
    corpus = root / payload["corpus"]["path"]
    rows = pq.read_table(corpus / "part-00000.parquet").to_pylist()
    assert len(rows) == len(payload["data"]) == 12
    assert {(bool(row["text"]), bool(row["image"])) for row in rows} == {(True, False), (False, True), (True, True)}
    ids = {row["docid"] for row in rows}
    assert all(record["pos_doc"][0]["id"] in ids for record in payload["data"])
    assert all(record["negative_mining_performed"] is False for record in payload["data"])
    documents = [json.loads(line) for line in (root / "beir/corpus.jsonl").read_text().splitlines()]
    for document in documents:
        if "image_path" in document:
            with Image.open(root / document["image_path"]) as image:
                image.verify()
    qrels = (root / "beir/qrels/test.tsv").read_text().splitlines()
    assert len(qrels) == 13
    assert {line.split("\t")[1] for line in qrels[1:]} == ids
    with pytest.raises(FileExistsError):
        _module("smoke_fixture").write_fixture(root)


def test_smoke_commands_load_real_profiles_and_keep_frozen_dependencies(tmp_path, monkeypatch):
    smoke = _module("smoke")
    monkeypatch.setattr("sys.argv", ["smoke.py", "--output", str(tmp_path / "run"), "--rerank", "--serve"])
    commands = []

    def capture(command, log, gpu):
        commands.append(command)
        if "embedding-download.log" == log.name:
            (tmp_path / "run/embedding-model.txt").write_text("/public/model")
        if "train-" in log.name:
            log.write_text("Loading checkpoint from saved-state\n")

    monkeypatch.setattr(smoke, "run", capture)
    monkeypatch.setattr(smoke, "check_metrics", lambda *args: None)
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: None)
    smoke.main()
    stages = {
        "embed/stage1_data_prep": DataPrepConfig,
        "embed/stage2_finetune": EmbedFinetuneConfig,
        "embed/stage3_eval": EvalConfig,
        "rerank/stage2_finetune": RerankFinetuneConfig,
        "embed/stage5_deploy": DeployConfig,
    }
    trained, evaluated = [], []
    for command in commands:
        assert "--frozen" in command
        if "--extra" in command:
            assert command[command.index("--extra") + 1] == "vl"
        runtime = Path(command[command.index("--project") + 1])
        assert (runtime / "uv.lock").is_file()
        if "--config" not in command:
            continue
        config_index = command.index("--config") + 1
        config = Path(command[config_index])
        stage = str(config.parents[1].relative_to(smoke.RECIPES))
        if stage in stages:
            resolved = load_pydantic_config(config, command[config_index + 1 :], stages[stage])
            if "finetune" in stage:
                trained.append(resolved)
            if stage == "embed/stage3_eval":
                evaluated.append(resolved)
    assert [item.max_steps for item in trained[:2]] == [2, 4]
    assert [item.num_epochs for item in trained[2:]] == [1, 2]
    assert all(item.train_n_passages == 2 for item in trained)
    assert all(item.local_batch_size == 1 for item in trained)
    assert len(evaluated) == 2
    assert evaluated[0].eval_base and evaluated[0].eval_finetuned and not evaluated[0].eval_nim
    assert evaluated[1].eval_nim and not evaluated[1].eval_base and not evaluated[1].eval_finetuned
    assert not any("writer" in argument for command in commands for argument in command)


@pytest.mark.parametrize("value", [None, math.nan, math.inf])
def test_metrics_check_rejects_missing_or_nonfinite_results(tmp_path, value):
    result = {
        name: {"NDCG": {"NDCG@10": value}, "MAP": {}, "Recall": {}, "Precision": {}} for name in ("base", "finetuned")
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        _module("smoke").check_metrics(path, ("base", "finetuned"))


def test_metric_check_requires_each_requested_backend(tmp_path):
    path = tmp_path / "results.json"
    path.write_text("{}")
    with pytest.raises(KeyError):
        _module("smoke").check_metrics(path, ("finetuned", "vllm"))


@pytest.mark.parametrize("hook", ["pytest_collectreport", "pytest_runtest_logreport"])
def test_ci_rejects_dependency_and_runtime_skips(hook):
    spec = importlib.util.spec_from_file_location("retrieval_ci_run", Path(__file__).with_name("run.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    plugin = runner.RequireCoverage()
    getattr(plugin, hook)(SimpleNamespace(skipped=True, nodeid="missing-dependency"))
    session = SimpleNamespace(exitstatus=pytest.ExitCode.OK)
    plugin.pytest_sessionfinish(session, pytest.ExitCode.OK)
    assert session.exitstatus == pytest.ExitCode.TESTS_FAILED
