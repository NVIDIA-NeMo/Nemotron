"""Exercise stage boundaries without loading a model or rendering a dataset."""

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron.recipes.parse import common
from nemotron.recipes.parse.config import DataConfig, EvalConfig, RLVRConfig, SFTConfig
from nemotron.recipes.parse.dental import config as dental_config
from nemotron.recipes.parse.dental import data, evaluate, model, rlvr
from nemotron.recipes.parse.stage1_sft import train as sft_stage
from nemotron.recipes.parse.stage2_rlvr import train as rlvr_stage
from nemotron.recipes.parse.stage3_eval import evaluate as eval_stage


@pytest.fixture
def dataset(tmp_path):
    """Tiny file-state fixture; bytes are deliberately not renderable images."""
    directory = tmp_path / "data"
    (directory / "images").mkdir(parents=True)
    for split in ("train", "monitor", "test"):
        image = f"images/{split}.png"
        (directory / image).write_bytes(f"unit fixture: {split}".encode())
        row = dict(id=split, image=image, context_id=split, prompt="prompt", target="target", record={})
        (directory / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    common.artifact(directory, "parse/dataset", directory, identity=common.dataset_identity(directory))
    return directory


def test_dataset_artifact_detects_changed_image_bytes(dataset):
    assert common.validate_dataset(dataset)["pages"] == {"train": 1, "monitor": 1, "test": 1}
    (dataset / "images/test.png").write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="contents changed"):
        common.validate_dataset(dataset)


def test_dataset_artifact_rejects_cross_split_context_overlap(dataset):
    path = dataset / "monitor.jsonl"
    row = json.loads(path.read_text())
    row["context_id"] = "train"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="context overlap"):
        common.dataset_identity(dataset)


def test_stage_failure_is_recorded_and_cannot_silently_replay(tmp_path):
    cfg = DataConfig(data_dir=tmp_path / "output")
    with pytest.raises(RuntimeError, match="test failure"):
        with common.stage_run("data", cfg.data_dir, cfg):
            raise RuntimeError("test failure")
    receipt = json.loads((cfg.data_dir / "stage.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["error"] == "RuntimeError"
    with pytest.raises(FileExistsError, match="empty output"):
        with common.stage_run("data", cfg.data_dir, cfg):
            pytest.fail("Failed outputs must not be replayed")


def test_sft_runs_greedy_baseline_and_collate_before_training(dataset, tmp_path, monkeypatch):
    calls = []
    cfg = SFTConfig(data_dir=dataset, output_dir=tmp_path / "sft", model="local-base")

    def prepare(checkpoint, output, scope):
        calls.append(("prepare", checkpoint, scope))
        return output

    class Trainer:
        baseline = True
        val_dataloader = object()
        processor = object()

        def _run_validation_epoch(self, loader):
            assert loader is self.val_dataloader and self.baseline
            calls.append("baseline")
            return "baseline-metrics"

        def log_val_metrics(self, metrics):
            assert metrics == "baseline-metrics"
            calls.append("log-baseline")

        def run_train_validation_loop(self):
            assert self.baseline is False and self.started > 0
            calls.append("train")
            (cfg.output_dir / "checkpoints/epoch_11_step_2851/model/consolidated").mkdir(parents=True)

    class Labels:
        def __ne__(self, other):
            assert other == -100
            return self

        def sum(self):
            return 3

    trainer = Trainer()

    def make_trainer(config, observer):
        assert config["step_scheduler"]["max_steps"] == 2852
        calls.append("setup")
        return trainer

    def collate(rows, processor):
        assert [row["id"] for row in rows] == ["train"] and processor is trainer.processor
        calls.append("collate")
        return {"labels": [Labels()]}

    monkeypatch.setattr(model, "prepare_model", prepare)
    monkeypatch.setitem(sys.modules, "nemotron.recipes.parse.dental.sft", SimpleNamespace(make_trainer=make_trainer))
    monkeypatch.setitem(sys.modules, "nemotron.recipes.parse.dental.sft_data", SimpleNamespace(collate=collate))
    result = sft_stage.main(cfg)
    assert calls == [("prepare", "local-base", "full"), "setup", "baseline", "log-baseline", "collate", "train"]
    artifact = json.loads(result.read_text())
    assert artifact["path"].endswith("epoch_11_step_2851/model/consolidated")
    assert artifact["processor_source"] == artifact["path"]
    assert common.validate_dataset(dataset)["pages"]["test"] == 1


def test_rlvr_preserves_order_and_processor_while_isolating_curriculum(dataset, tmp_path, monkeypatch):
    from nemotron.recipes.parse.dental import curriculum

    calls = []
    fixed_before = common.dataset_identity(dataset)
    checkpoint = tmp_path / "sft-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"unit fixture")
    sft_output = tmp_path / "sft"
    sft_output.mkdir()
    common.artifact(sft_output, "parse/model", checkpoint, processor_source=str(checkpoint))
    cfg = RLVRConfig(data_dir=dataset, sft_output=sft_output, output_dir=tmp_path / "rlvr")

    def make_records(seed, contexts):
        assert (seed, contexts) == (31415, 512)
        return ["candidate fixture"]

    def save_pages(records, output, split):
        assert records == ["candidate fixture"] and split == "rl_candidates"
        row = dict(id="candidate", image="images/candidate.png", record={})
        manifest = output / f"{split}.jsonl"
        manifest.write_text(json.dumps(row) + "\n")
        calls.append("generate-candidates")
        return manifest

    def sample(model_path, rows, output, samples, processor_source):
        assert processor_source == checkpoint
        calls.append((output.name, samples))
        if output.name in ("mining", "before"):
            assert model_path == checkpoint
        else:
            assert model_path == cfg.output_dir / "checkpoint-final"
        answer = dict(correct=True, ended=True, text_cer=0.0, chart_exact=1.0, patient_fields=1.0, history_rows=1.0)
        return [dict(id=row["id"], answers=[dict(answer) for _ in range(samples)]) for row in rows]

    def select(candidates, answers, pages, seed, excluded):
        assert pages == 256 and seed == 42
        assert [row["id"] for row in excluded] == ["train", "monitor", "test"]
        assert answers[0]["id"] == candidates[0]["id"]
        calls.append("select")
        return copy.deepcopy(candidates)

    def prepare(model_path, output, scope, processor_source):
        assert model_path == checkpoint and processor_source == checkpoint and scope == "decoder"
        calls.append("prepare-decoder")
        return output

    def load_config(path, model_path, data_dir, output):
        assert data_dir == cfg.output_dir / "curriculum"
        monitor = json.loads((data_dir / "monitor.jsonl").read_text())
        selected = json.loads((data_dir / "rl_train.jsonl").read_text())
        assert Path(monitor["image"]).is_absolute()
        assert monitor["image"] == str(dataset / "images/monitor.png")
        assert selected["image"] == "images/candidate.png"
        return {"unit_test": "configuration"}

    def train(config, observer):
        assert config == {"unit_test": "configuration"}
        calls.append("train")
        return cfg.output_dir / "checkpoint-final"

    monkeypatch.setattr(data, "make_records", make_records)
    monkeypatch.setattr(data, "save_pages", save_pages)
    monkeypatch.setattr(evaluate, "evaluate_isolated", sample)
    monkeypatch.setattr(curriculum, "select_training_pages", select)
    monkeypatch.setattr(model, "prepare_model", prepare)
    monkeypatch.setattr(dental_config, "rlvr_config", load_config)
    monkeypatch.setattr(rlvr, "train", train)
    result = rlvr_stage.main(cfg)
    assert calls == [
        "generate-candidates", ("mining", 4), "select", ("before", 8), "prepare-decoder", "train", ("after", 8)
    ]
    assert common.dataset_identity(dataset) == fixed_before
    artifact = json.loads(result.read_text())
    assert artifact["processor_source"] == str(checkpoint)
    summary = json.loads(Path(artifact["evaluation"]).read_text())
    assert summary["before"]["samples"] == summary["after"]["samples"] == 8


def test_isolated_eval_worker_receives_recipe_path_and_native_sampler(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(("/existing/site-packages", "/another/path")))

    class Worker:
        def options(self, **kwargs):
            calls["options"] = kwargs
            return self

        def remote(self, *args):
            return "sampled answers"

    ray = SimpleNamespace(
        is_initialized=lambda: True,
        remote=lambda **kwargs: lambda function: Worker(),
        get=lambda result: result,
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    assert evaluate.evaluate_isolated(tmp_path / "model", [], tmp_path / "new-output") == "sampled answers"
    runtime = calls["options"]["runtime_env"]
    assert runtime["py_executable"] == sys.executable
    assert runtime["env_vars"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    paths = runtime["env_vars"]["PYTHONPATH"].split(os.pathsep)
    assert paths[0] == str(Path(dental_config.__file__).resolve().parents[4])
    assert paths[1:] == ["/existing/site-packages", "/another/path"]


def test_standalone_eval_publishes_directory_artifact_with_model_processor(dataset, tmp_path, monkeypatch):
    checkpoint = tmp_path / "rlvr-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"unit fixture")
    processor = tmp_path / "sft-processor"
    processor.mkdir()
    model_output = tmp_path / "rlvr"
    model_output.mkdir()
    model_artifact = common.artifact(
        model_output, "parse/model", checkpoint, processor_source=str(processor)
    )
    cfg = EvalConfig(data_dir=dataset, model_artifact=model_artifact, output_dir=tmp_path / "evaluation")

    def sample(model_path, rows, output, samples, processor_source):
        assert model_path == checkpoint
        assert processor_source == str(processor)
        assert [row["id"] for row in rows] == ["test"]
        assert samples == 8
        assert output == cfg.output_dir / "predictions"
        answer = dict(correct=True, ended=True, text_cer=0.0, chart_exact=1.0, patient_fields=1.0, history_rows=1.0)
        answers = [dict(id="test", answers=[dict(answer) for _ in range(samples)])]
        output.mkdir()
        (output / "answers.json").write_text(json.dumps(answers))
        return answers

    monkeypatch.setattr(evaluate, "evaluate_isolated", sample)
    # Exercise the real common.artifact / Artifact.save serialization boundary.
    result = eval_stage.main(cfg)
    handoff = json.loads(result.read_text())
    metadata = json.loads((cfg.output_dir / "metadata.json").read_text())
    summary = json.loads((cfg.output_dir / "evaluation.json").read_text())
    receipt = json.loads((cfg.output_dir / "stage.json").read_text())
    assert handoff["kind"] == metadata["type"] == "parse/evaluation"
    assert handoff["path"] == metadata["path"] == str(cfg.output_dir)
    assert Path(handoff["path"]).is_dir()
    assert handoff["evaluation"] == metadata["metadata"]["evaluation"] == str(cfg.output_dir / "evaluation.json")
    assert handoff["model"] == metadata["metadata"]["model"] == str(checkpoint)
    assert summary["pages"] == 1 and summary["answers"] == 8
    assert summary["metrics"]["ended"] == 1.0
    assert receipt["status"] == "completed"
