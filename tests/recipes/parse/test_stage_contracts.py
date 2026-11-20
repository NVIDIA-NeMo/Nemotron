"""Exercise stage boundaries without loading a model or rendering a dataset."""

import copy
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemotron.recipes.parse import RECIPE_VERSION, common
from nemotron.recipes.parse.config import DataConfig, EvalConfig, RLVRConfig, SFTConfig
from nemotron.recipes.parse.dental import data, evaluate
from nemotron.recipes.parse.stage1_sft import train as sft_stage
from nemotron.recipes.parse.stage2_rlvr import train as rlvr_stage
from nemotron.recipes.parse.stage3_eval import evaluate as eval_stage
from nemotron.recipes.parse.training import config as training_config
from nemotron.recipes.parse.training import model, rlvr
from nemotron.recipes.parse.training.data import load_pages
from nemotron.recipes.parse.training.lineage import model_reference, read_lineage


@pytest.fixture
def dataset(tmp_path):
    """Tiny file-state fixture; bytes are deliberately not renderable images."""
    directory = tmp_path / "data"
    (directory / "images").mkdir(parents=True)
    for split in ("train", "monitor", "test"):
        image = f"images/{split}.png"
        (directory / image).write_bytes(f"unit fixture: {split}".encode())
        row = dict(
            id=split, image=image, context_id=split, prompt="prompt", target="target", reference={"invoice_total": 17}
        )
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
    assert receipt["recipe_version"] == RECIPE_VERSION
    assert "source_revision" not in receipt
    with pytest.raises(FileExistsError, match="empty output"):
        with common.stage_run("data", cfg.data_dir, cfg):
            pytest.fail("Failed outputs must not be replayed")


@pytest.mark.parametrize("task_kind", ["dental", "custom", "none"])
def test_sft_runs_baseline_and_collate_before_training(dataset, tmp_path, monkeypatch, task_kind):
    calls = []
    cfg = SFTConfig(data_dir=dataset, output_dir=tmp_path / "sft", model="local-base")
    if task_kind != "dental":
        for name in list(sys.modules):
            if name.startswith("nemotron.recipes.parse.dental"):
                monkeypatch.setitem(sys.modules, name, None)

        def monitor(model, processor, rows):
            return rows, {"invoice_exact": 1.0}

        monkeypatch.setitem(
            sys.modules,
            "invoice_task",
            SimpleNamespace(
                Task=lambda: SimpleNamespace(monitor=monitor),
            ),
        )
        cfg.task = "invoice_task.Task" if task_kind == "custom" else None

    base = tmp_path / "configured-base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"base fixture")

    def prepare(checkpoint, output, scope, revision=None):
        calls.append(("prepare", checkpoint, scope))
        output.mkdir()
        (output / "model-source.json").write_text(json.dumps(dict(
            checkpoint=str(base), processor=str(base), revision=revision)))
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
            checkpoint = cfg.output_dir / "checkpoints/epoch_11_step_2851/model/consolidated"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}")
            (checkpoint / "model.safetensors").write_bytes(b"trained fixture")

    class Labels:
        def __ne__(self, other):
            assert other == -100
            return self

        def sum(self):
            return 3

    trainer = Trainer()

    def make_trainer(config, observer, *, monitor):
        if task_kind == "none":
            assert monitor is None
        elif task_kind == "custom":
            assert monitor(None, None, [])[1] == {"invoice_exact": 1.0}
        else:
            assert callable(monitor)
        assert config["step_scheduler"]["max_steps"] == 2852
        calls.append("setup")
        return trainer

    def collate(rows, processor):
        assert [row["id"] for row in rows] == ["train"] and processor is trainer.processor
        calls.append("collate")
        return {"labels": [Labels()]}

    monkeypatch.setattr(model, "resolve_model", lambda checkpoint, revision: (base, "resolved-revision"))
    monkeypatch.setattr(model, "prepare_model", prepare)
    monkeypatch.setitem(sys.modules, "nemotron.recipes.parse.training.sft", SimpleNamespace(make_trainer=make_trainer))
    monkeypatch.setitem(sys.modules, "nemotron.recipes.parse.training.sft_data", SimpleNamespace(collate=collate))
    result = sft_stage.main(cfg)
    persistent_base = cfg.output_dir / "base-model"
    assert calls == [("prepare", persistent_base, "full"), "setup", "baseline", "log-baseline", "collate", "train"]
    artifact = json.loads(result.read_text())
    assert artifact["path"].endswith("epoch_11_step_2851/model/consolidated")
    assert artifact["processor_source"] == artifact["path"]
    assert artifact["lineage"]["models"]["base"]["path"] == str(persistent_base)
    assert artifact["lineage"]["models"]["base"]["revision"] == "resolved-revision"
    # Consuming the artifact must not require the original container/cache.
    shutil.rmtree(base)
    read_lineage(result, "sft", common.dataset_identity(dataset))
    assert artifact["lineage"]["models"]["base"]["source"] == "local-base"
    assert common.validate_dataset(dataset)["pages"]["test"] == 1


@pytest.mark.parametrize("task_kind", ["dental", "custom"])
def test_rlvr_preserves_order_and_processor_while_isolating_curriculum(dataset, tmp_path, monkeypatch, task_kind):
    from nemotron.recipes.parse.dental import curriculum

    calls = []
    fixed_before = common.dataset_identity(dataset)
    checkpoint = tmp_path / "sft-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"unit fixture")
    sft_output = tmp_path / "sft"
    sft_output.mkdir()
    lineage = dict(schema_version=1, dataset_identity=fixed_before,
                   models=dict(base=model_reference(checkpoint), sft=model_reference(checkpoint)))
    common.artifact(sft_output, "parse/model", checkpoint, stage="sft",
                    processor_source=str(checkpoint), lineage=lineage)
    cfg = RLVRConfig(data_dir=dataset, sft_output=sft_output, output_dir=tmp_path / "rlvr")
    expected_environment = "nemotron.recipes.parse.dental.environment.DentalEnvironment"
    expected_reward_metric = "page_exact"

    def make_records(seed, contexts):
        assert (seed, contexts) == (31415, 512)
        return ["candidate fixture"]

    def save_pages(records, output, split):
        assert records == ["candidate fixture"] and split == "rl_candidates"
        row = dict(
            id="candidate", image="images/candidate.png", context_id="candidate", reference={"invoice_total": 17}
        )
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
        if task_kind == "custom":
            answer = dict(invoice_exact=1.0)
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

    def load_config(path, model_path, data_dir, output, *, prompt_file, overrides=None):
        assert overrides in (None, {})
        assert Path(prompt_file).name == "prompt.txt"
        assert data_dir == cfg.output_dir / "curriculum"
        monitor = json.loads((data_dir / "monitor.jsonl").read_text())
        selected = json.loads((data_dir / "rl_train.jsonl").read_text())
        assert Path(monitor["image"]).is_absolute()
        assert monitor["image"] == str(dataset / "images/monitor.png")
        assert selected["image"] == "images/candidate.png"
        return {"unit_test": "configuration"}

    def train(config, observer, *, environment_class, reward_metric):
        assert environment_class == expected_environment
        assert reward_metric == expected_reward_metric
        assert config == {"unit_test": "configuration"}
        calls.append("train")
        final = cfg.output_dir / "checkpoint-final"
        final.mkdir()
        (final / "config.json").write_text("{}")
        (final / "model.safetensors").write_bytes(b"RL fixture")
        return final

    monkeypatch.setattr(data, "make_records", make_records)
    monkeypatch.setattr(data, "save_pages", save_pages)
    monkeypatch.setattr(evaluate, "evaluate_isolated", sample)
    monkeypatch.setattr(curriculum, "select_training_pages", select)
    monkeypatch.setattr(model, "prepare_model", prepare)
    monkeypatch.setattr(training_config, "rlvr_config", load_config)
    monkeypatch.setattr(rlvr, "train", train)
    if task_kind == "custom":
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("prompt")
        expected_environment = "invoice_task.InvoiceEnvironment"
        expected_reward_metric = "invoice_exact"
        task = SimpleNamespace(
            prompt_file=prompt_file,
            environment_class=expected_environment,
            reward_metric=expected_reward_metric,
            create_candidates=lambda output, seed, contexts: load_pages(
                save_pages(make_records(seed, contexts), output, "rl_candidates")
            ),
            select_candidates=select,
            evaluate=sample,
            summarize=lambda answers: dict(samples=len(answers[0]["answers"]), invoice_accuracy=1.0),
        )
        monkeypatch.setitem(sys.modules, "invoice_task", SimpleNamespace(Task=lambda: task))
        cfg.task = "invoice_task.Task"
        for name in list(sys.modules):
            if name.startswith("nemotron.recipes.parse.dental"):
                monkeypatch.setitem(sys.modules, name, None)
    result = rlvr_stage.main(cfg)
    assert calls == [
        "generate-candidates",
        ("mining", 4),
        "select",
        ("before", 8),
        "prepare-decoder",
        "train",
        ("after", 8),
    ]
    assert common.dataset_identity(dataset) == fixed_before
    artifact = json.loads(result.read_text())
    assert artifact["processor_source"] == str(checkpoint)
    assert artifact["sft_artifact"] == str(sft_output / "artifact.json")
    assert artifact["lineage"]["models"]["base"] == lineage["models"]["base"]
    assert set(artifact["lineage"]["models"]) == {"base", "sft", "rlvr"}
    summary = json.loads(Path(artifact["evaluation"]).read_text())
    assert summary["before"]["samples"] == summary["after"]["samples"] == 8
    if task_kind == "custom":
        assert summary["before"]["invoice_accuracy"] == summary["after"]["invoice_accuracy"] == 1.0


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
    assert paths[0] == str(Path(training_config.__file__).resolve().parents[4])
    assert paths[1:] == ["/existing/site-packages", "/another/path"]


def test_explicit_single_model_eval_publishes_directory_artifact(dataset, tmp_path, monkeypatch):
    checkpoint = tmp_path / "rlvr-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"unit fixture")
    processor = tmp_path / "sft-processor"
    processor.mkdir()
    model_output = tmp_path / "rlvr"
    model_output.mkdir()
    common.artifact(model_output, "parse/model", checkpoint, processor_source=str(processor))
    cfg = EvalConfig(data_dir=dataset, model=str(checkpoint), processor_source=str(processor),
                     output_dir=tmp_path / "evaluation")

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
    assert receipt["recipe_version"] == handoff["recipe_version"] == RECIPE_VERSION


@pytest.fixture
def comparison_artifact(dataset, tmp_path):
    models = {}
    for name in ("base", "sft", "rlvr"):
        path = tmp_path / f"custom-{name}"
        path.mkdir()
        (path / "config.json").write_text("{}")
        (path / "tokenizer.json").write_text("{}")
        (path / "model.safetensors").write_bytes(f"{name} weight fixture".encode())
        models[name] = model_reference(path, source="customer/parse-base" if name == "base" else None)
    output = tmp_path / "rlvr-artifact"
    output.mkdir()
    lineage = dict(schema_version=1, dataset_identity=common.dataset_identity(dataset), models=models)
    return common.artifact(output, "parse/model", Path(models["rlvr"]["path"]), stage="rlvr",
                           processor_source=models["rlvr"]["processor_source"], lineage=lineage)


def test_default_eval_compares_actual_base_sft_rlvr(dataset, comparison_artifact, tmp_path, monkeypatch):
    from nemotron.recipes.parse.dental import comparison

    calls = []
    source = json.loads(comparison_artifact.read_text())
    before = comparison_artifact.read_bytes()

    def sample(path, rows, output, samples, processor_source):
        name = output.parent.name
        calls.append(name)
        reference = source["lineage"]["models"][name]
        assert str(path) == reference["path"]
        assert processor_source == reference["processor_source"]
        assert [r["id"] for r in rows] == ["test"]
        assert samples == 4
        correct = {"base": 1, "sft": 2, "rlvr": 3}[name]
        cer = {"base": 0.3, "sft": 0.1, "rlvr": 0.05}[name]
        answers = [dict(id="test", answers=[dict(correct=i < correct, text_cer=cer, ended=True,
                                                chart_exact=1.0, patient_fields=1.0, history_rows=1.0)
                                            for i in range(samples)])]
        output.mkdir()
        (output / "answers.json").write_text(json.dumps(answers))
        return answers

    monkeypatch.setattr(comparison, "evaluate_isolated", sample)
    cfg = EvalConfig(data_dir=dataset, model_artifact=comparison_artifact,
                     output_dir=tmp_path / "comparison", samples=4)
    result = eval_stage.main(cfg)
    assert calls == ["base", "sft", "rlvr"]
    assert comparison_artifact.read_bytes() == before
    handoff = json.loads(result.read_text())
    summary = json.loads((cfg.output_dir / "evaluation.json").read_text())
    assert handoff["kind"] == "parse/evaluation" and handoff["lineage"] == source["lineage"]
    assert summary["protocol"] == evaluate.evaluation_protocol(4)
    assert summary["models"]["base"]["pass_at_k"]["1"] == 0.25
    assert summary["models"]["rlvr"]["pass_at_k"]["1"] == 0.75
    assert summary["uplift"]["base_to_rlvr"]["pass_at_1_gain_pp"] == 50
    assert summary["uplift"]["base_to_sft"]["text_cer_reduction_pp"] == pytest.approx(20)
    report = (cfg.output_dir / "comparison.md").read_text()
    assert "| Base | 25.00% | 30.0000% |" in report
    assert "| RLVR | 75.00% | 5.0000% |" in report
    for name in calls:
        assert (cfg.output_dir / name / "predictions/answers.json").is_file()
        assert json.loads((cfg.output_dir / name / "evaluation.json").read_text()) == summary["models"][name]


@pytest.mark.parametrize("problem", ["missing_lineage", "changed_base", "changed_processor", "different_dataset"])
def test_comparison_rejects_incompatible_lineage_before_generation(
    dataset, comparison_artifact, tmp_path, monkeypatch, problem
):
    from nemotron.recipes.parse.dental import comparison

    record = json.loads(comparison_artifact.read_text())
    if problem == "missing_lineage":
        record.pop("lineage")
    elif problem in ("changed_base", "changed_processor"):
        reference = record["lineage"]["models"]["base"]
        filename = "model.safetensors" if problem == "changed_base" else "tokenizer.json"
        (Path(reference["path"]) / filename).write_bytes(b"changed")
    else:
        record["lineage"]["dataset_identity"]["manifests"]["test"] = "different-test-set"
    comparison_artifact.write_text(json.dumps(record))
    monkeypatch.setattr(comparison, "evaluate_isolated", lambda *a, **kw: pytest.fail("must reject before generation"))
    cfg = EvalConfig(data_dir=dataset, model_artifact=comparison_artifact, output_dir=tmp_path / "comparison")
    with pytest.raises(ValueError, match="lineage|changed|dataset"):
        eval_stage.main(cfg)
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("problem", ["page_order", "sample_count"])
def test_comparison_rejects_incomplete_answers(dataset, comparison_artifact, tmp_path, monkeypatch, problem):
    from nemotron.recipes.parse.dental import comparison

    def sample(path, rows, output, samples, processor_source):
        answer = dict(correct=True, text_cer=0.0, ended=True, chart_exact=1.0, patient_fields=1.0, history_rows=1.0)
        return [dict(id="wrong-page" if problem == "page_order" else "test",
                     answers=[answer] * (samples - 1 if problem == "sample_count" else samples))]

    monkeypatch.setattr(comparison, "evaluate_isolated", sample)
    cfg = EvalConfig(data_dir=dataset, model_artifact=comparison_artifact, output_dir=tmp_path / "comparison")
    with pytest.raises(ValueError, match="ordered|sample count"):
        eval_stage.main(cfg)
    assert not (cfg.output_dir / "artifact.json").exists()
    assert json.loads((cfg.output_dir / "stage.json").read_text())["status"] == "failed"
