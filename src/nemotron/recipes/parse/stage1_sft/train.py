# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/sft"
# image = "nvcr.io/nvidia/nemo-automodel:26.10.00@sha256:0e4279f2740b4de9c3884f5d434d1a1520057c825f8a28d1e66dd16515451e9c"
# setup = "Stage dependencies resolved with uv; see docs/nemotron/parse/environments.md."
# [tool.runspec.run]
# launch = "direct"
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 1
# ///

"""Run SFT with terminal progress and explicit artifacts."""

import time
from pathlib import Path

from omegaconf import OmegaConf

from nemotron.recipes.parse.common import Progress, artifact, stage_run, validate_dataset, write_json
from nemotron.recipes.parse.config import SFTConfig, worker_config


def main(cfg: SFTConfig | None = None):
    from nemotron.recipes.parse.training.config import sft_config
    from nemotron.recipes.parse.training.data import load_pages
    from nemotron.recipes.parse.training.lineage import model_reference, verify_model
    from nemotron.recipes.parse.training.model import materialize_base_model, prepare_model, sft_checkpoint
    from nemotron.recipes.parse.training.sft import make_trainer
    from nemotron.recipes.parse.training.sft_data import collate
    from nemotron.recipes.parse.training.task import load_task

    cfg = cfg or worker_config("sft")
    task = load_task(cfg.task, stage="sft") if cfg.task is not None else None
    identity = validate_dataset(cfg.data_dir)
    with stage_run("sft", cfg.output_dir, cfg) as output:
        # Cache paths inside a training container disappear with that container.
        # Persist the unchanged ancestor separately from the adapted model view.
        base_model, revision = materialize_base_model(cfg.model, output / "base-model", revision=cfg.model_revision)
        base = model_reference(base_model, source=cfg.model, revision=revision)
        model = prepare_model(base_model, output / "initial-model", scope="full")
        raw = sft_config(cfg.training_config, model, cfg.data_dir, output)
        config = OmegaConf.to_container(OmegaConf.merge(raw, cfg.training), resolve=True)
        for section in ("model", "processor"):
            if Path(config[section]["pretrained_model_name_or_path"]).resolve() != model.resolve():
                raise ValueError("Set the SFT model through model/model_revision so lineage matches training")
        for section, split in (("dataset", "train"), ("validation_dataset", "monitor")):
            path = config.get(section, {}).get("path")
            if path is None or Path(path).resolve() != (cfg.data_dir / f"{split}.jsonl").resolve():
                raise ValueError(f"Set SFT data through data_dir; {section}.path must match the dataset artifact")
        write_json(output / "training-config.json", config)
        observer = Progress(output, "sft")
        trainer = make_trainer(config, observer=observer, monitor=task.monitor if task is not None else None)
        # Evaluate the initial model and validate a training batch before optimization.
        trainer.log_val_metrics(trainer._run_validation_epoch(trainer.val_dataloader))
        batch = collate(load_pages(cfg.data_dir / "train.jsonl")[:1], trainer.processor)
        print("Supervised tokens:", int((batch["labels"][0] != -100).sum()), flush=True)
        trainer.baseline = False
        trainer.started = time.monotonic()
        trainer.run_train_validation_loop()
        checkpoint = sft_checkpoint(output)
        verify_model(base)
        lineage = dict(
            schema_version=1, dataset_identity=identity, models=dict(base=base, sft=model_reference(checkpoint))
        )
        result = artifact(
            output,
            "parse/model",
            checkpoint,
            stage="sft",
            base_model=cfg.model,
            lineage=lineage,
            dataset_manifests=identity["manifests"],
            processor_source=str(checkpoint.resolve()),
        )
        print(f"SFT model: {result}", flush=True)
        return result


if __name__ == "__main__":
    main()
