"""Task boundaries must preserve native training while accepting non-dental labels."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from nemotron.recipes.parse.training import config, rlvr, sft
from nemotron.recipes.parse.training.task import load_task


def test_task_loader_checks_only_the_requested_stage_contract(tmp_path, monkeypatch):
    task = SimpleNamespace(monitor=lambda *args: ([], {}))
    monkeypatch.setitem(sys.modules, "invoice_task", SimpleNamespace(Task=lambda: task))
    assert load_task("invoice_task.Task", stage="sft") is task
    with pytest.raises(TypeError, match="create_candidates"):
        load_task("invoice_task.Task", stage="rlvr")
    with pytest.raises(ValueError, match="fully qualified"):
        load_task("Task", stage="sft")

    for name in ("create_candidates", "select_candidates", "evaluate", "summarize"):
        setattr(task, name, lambda *args: [])
    task.environment_class = "invoice_task.InvoiceEnvironment"
    task.reward_metric = None
    task.prompt_file = tmp_path / "prompt.txt"
    with pytest.raises(ValueError, match="existing prompt_file"):
        load_task("invoice_task.Task", stage="rlvr")
    task.prompt_file.write_text("Extract invoice fields")
    assert load_task("invoice_task.Task", stage="rlvr") is task


@pytest.mark.parametrize("with_monitor", [True, False])
def test_sft_monitor_preserves_native_loss_and_accepts_custom_metrics(tmp_path, monkeypatch, with_monitor):
    calls = []
    manifest = tmp_path / "monitor.jsonl"
    manifest.write_text(json.dumps(dict(id="invoice", image="invoice.png", reference={"total": 17})) + "\n")

    class NativeTrainer:
        def __init__(self, cfg):
            self.processor = object()
            self.model_parts = [object()]
            self.step_scheduler = SimpleNamespace(epoch_len=5)

        def setup(self):
            pass

        def _run_validation_epoch(self, loader):
            return SimpleNamespace(step=2, metrics={"loss": 0.25})

    monkeypatch.setitem(sys.modules, "nemo_rl", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.components.config.loader", SimpleNamespace(ConfigNode=lambda c: c)
    )
    monkeypatch.setitem(
        sys.modules, "nemo_automodel.recipes.vlm.finetune", SimpleNamespace(FinetuneRecipeForVLM=NativeTrainer)
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda count: None))

    def monitor(model, processor, rows):
        assert rows[0]["reference"] == {"total": 17} and "record" not in rows[0]
        assert rows[0]["image"] == str(tmp_path / "invoice.png")
        calls.append("monitor")
        return [dict(id="invoice", invoice_exact=1.0)], {"invoice_exact": 1.0}

    cfg = {
        "validation_dataset": {"path": str(manifest)},
        "checkpoint": {"checkpoint_dir": str(tmp_path / "checkpoints")},
        "step_scheduler": {"max_steps": 10},
    }
    observer = SimpleNamespace(configure=lambda total, per_epoch: calls.append((total, per_epoch)))
    trainer = sft.make_trainer(cfg, observer, monitor=monitor if with_monitor else None)
    result = trainer._run_validation_epoch(None)
    assert result.metrics["loss"] == 0.25
    predictions = tmp_path / "predictions-0.json"
    if with_monitor:
        assert calls == [(10, 5), "monitor"]
        assert result.metrics == {"loss": 0.25, "invoice_exact": 1.0}
        assert json.loads(predictions.read_text()) == [dict(id="invoice", invoice_exact=1.0)]
    else:
        assert calls == [(10, 5)] and not predictions.exists()
        assert result.metrics == {"loss": 0.25}


def test_rlvr_config_resolves_task_prompt_and_generic_targets(tmp_path, monkeypatch):
    framework = tmp_path / "framework"
    native = framework / "examples/configs/vlm_grpo_3B.yaml"
    native.parent.mkdir(parents=True)
    native.write_text("env:\n  unrelated_task: {}\n")
    monkeypatch.setitem(sys.modules, "nemo_rl", SimpleNamespace(__file__=str(framework / "nemo_rl/__init__.py")))
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.utils.config",
        SimpleNamespace(
            load_config=OmegaConf.load,
            register_omegaconf_resolvers=lambda: None,
        ),
    )
    prompt = tmp_path / "invoice-prompt.txt"
    prompt.write_text("Extract invoice fields")
    cfg = config.rlvr_config(
        config.recipe_root() / "stage2_rlvr/config/grpo.yaml",
        tmp_path / "model",
        tmp_path / "data",
        tmp_path / "output",
        prompt_file=prompt,
    )
    assert cfg["data"]["default"]["prompt_file"] == str(prompt)
    assert cfg["data"]["default"]["processor"] == "parse_page"
    assert cfg["data"]["train"]["dataset_name"] == "nemotron.recipes.parse.training.rl_data.ParsePages"
    assert set(cfg["env"]) == {"parse-page", "parse-page-validation"}
    assert cfg["policy"]["hf_config_overrides"] == {"parse_train_scope": "decoder"}


def test_rlvr_driver_registers_selected_reward_actor_and_reports_custom_components(tmp_path, monkeypatch):
    calls, updates = [], []
    processor_registry, environment_registry, actor_registry = {}, {}, {}
    checkpoint = tmp_path / "checkpoints/step_1/policy/weights/model/consolidated"
    checkpoint.mkdir(parents=True)
    metrics_file = tmp_path / "components.jsonl"
    metrics_file.write_text(json.dumps({"samples": [{"invoice_exact": 0.75}]}) + "\n")
    environment_class = "invoice_task.InvoiceEnvironment"
    cfg = {
        "policy": {"tokenizer": {}, "generation": {}},
        "data": {},
        "data_plane": {},
        "env": {
            "parse-page": {"phase": "train", "metrics_file": str(metrics_file)},
            "parse-page-validation": {"phase": "validation"},
        },
        "grpo": {"max_num_steps": 1},
        "checkpointing": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    processor = SimpleNamespace(tokenizer=object())
    dataset = SimpleNamespace(task_data_processors={"parse-page": object()})
    logger = SimpleNamespace(loggers=[], finish=lambda: calls.append("logger-finished"))
    policy = SimpleNamespace(shutdown=lambda: calls.append("policy-stopped"))
    generation = SimpleNamespace(shutdown=lambda: calls.append("generation-stopped"))

    def master_config(**values):
        return SimpleNamespace(**dict(values, grpo=SimpleNamespace(**values["grpo"])))

    def setup(master, tokenizer, data, validation, **kwargs):
        return (policy, generation, None, None, [object()], [], None, logger, nullcontext(), {}, master, None, None)

    def create_env(name, settings):
        assert environment_registry[name] == {"actor_class_fqn": environment_class}
        calls.append(name)
        return name

    def run(*args, **kwargs):
        assert args[6] == {"parse-page": "parse-page"}
        assert args[7] == {"parse-page": "parse-page-validation"}
        local = logger.loggers[0]
        local.log_metrics({"loss": 0.2}, 1, prefix="train")
        local.log_metrics({}, 1, prefix="timing/train", step_finished=True)

    def select(master, label):
        assert label == "Parse RLVR"
        return run

    def process_page(*args):
        raise AssertionError("Data processing is outside this driver test")

    modules = {
        "ray": SimpleNamespace(
            init=lambda **kwargs: calls.append("ray-started"), shutdown=lambda: calls.append("ray-stopped")
        ),
        "nemo_rl.algorithms.grpo": SimpleNamespace(MasterConfig=master_config, setup=setup),
        "nemo_rl.algorithms.utils": SimpleNamespace(get_tokenizer=lambda *a, **kw: processor),
        "nemo_rl.data.utils": SimpleNamespace(setup_response_data=lambda *a, **kw: (dataset, dataset)),
        "nemo_rl.data.processors": SimpleNamespace(
            PROCESSOR_REGISTRY=processor_registry,
            register_processor=lambda name, fn: processor_registry.update({name: fn}),
        ),
        "nemo_rl.data_plane.factory": SimpleNamespace(
            make_policy_factory=lambda cfg: None, select_sync_trainer=select
        ),
        "nemo_rl.distributed.ray_actor_environment_registry": SimpleNamespace(
            ACTOR_ENVIRONMENT_REGISTRY=actor_registry
        ),
        "nemo_rl.distributed.virtual_cluster": SimpleNamespace(PY_EXECUTABLES=SimpleNamespace(SYSTEM="system")),
        "nemo_rl.environments.utils": SimpleNamespace(
            ENV_REGISTRY=environment_registry,
            create_env=create_env,
            register_env=lambda name, actor: environment_registry.update({name: {"actor_class_fqn": actor}}),
        ),
        "nemo_rl.models.generation": SimpleNamespace(configure_generation_config=lambda cfg, tokenizer: cfg),
        "nemo_rl.utils.logger": SimpleNamespace(LoggerInterface=object),
        "nemo_rl.utils.venvs": SimpleNamespace(add_hf_modules_cache_to_pythonpath=lambda value: value),
        "nemotron.recipes.parse.training.rl_data": SimpleNamespace(process_page=process_page),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    observer = SimpleNamespace(configure=lambda *a: None, update=lambda *args: updates.append(args))
    result = rlvr.train(cfg, observer, environment_class=environment_class, reward_metric="invoice_exact")
    assert result == checkpoint
    assert processor_registry == {"parse_page": process_page}
    assert actor_registry[environment_class] == "system"
    assert updates == [(1, "train", {"loss": 0.2, "invoice_exact": 0.75, "reward": 0.75})]
    assert calls == [
        "ray-started",
        "parse-page",
        "parse-page-validation",
        "logger-finished",
        "generation-stopped",
        "policy-stopped",
        "ray-stopped",
    ]
    # A second task must not silently reuse the first task's registered reward.
    with pytest.raises(ValueError, match="different reward environment"):
        rlvr.train(cfg, observer, environment_class="other_task.Environment")


def test_dental_monitor_keeps_reference_scores_after_training_extraction(monkeypatch):
    from nemotron.recipes.parse.dental import metrics
    from nemotron.recipes.parse.dental.task import DentalTask

    row = json.loads((Path(__file__).parent / "fixtures/example.json").read_text())[0]
    row["id"] = "dental-example"
    monkeypatch.setattr(sft, "predict", lambda *args: [dict(id=row["id"], prediction=row["target"], ended=True)])
    predictions, summary = DentalTask().monitor(None, None, [row])
    expected = metrics.components(row["target"], row)
    assert {key: predictions[0][key] for key in expected} == expected
    assert summary == {"chart_exact": expected["chart_exact"], "text_cer": expected["text_cer"]}
