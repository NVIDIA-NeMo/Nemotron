# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/sft"
# setup = "Use the isolated Parse runtime; see recipes/parse/README.md."
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

"""Run the reference SFT schedule with terminal progress and explicit artifacts."""

import time

from omegaconf import OmegaConf

from nemotron.recipes.parse.common import Progress, artifact, stage_run, validate_dataset, write_json
from nemotron.recipes.parse.config import SFTConfig, worker_config


def main(cfg: SFTConfig | None = None):
    from nemotron.recipes.parse.dental.config import sft_config
    from nemotron.recipes.parse.dental.data import load_pages
    from nemotron.recipes.parse.dental.model import prepare_model, sft_checkpoint
    from nemotron.recipes.parse.dental.sft import make_trainer
    from nemotron.recipes.parse.dental.sft_data import collate

    cfg = cfg or worker_config("sft")
    identity = validate_dataset(cfg.data_dir)
    with stage_run("sft", cfg.output_dir, cfg) as output:
        model = prepare_model(cfg.model, output / "initial-model", scope="full")
        raw = sft_config(cfg.training_config, model, cfg.data_dir, output)
        config = OmegaConf.to_container(OmegaConf.merge(raw, cfg.training), resolve=True)
        write_json(output / "training-config.json", config)
        observer = Progress(output, "sft")
        trainer = make_trainer(config, observer=observer)
        # Preserve the notebook's baseline validation and pre-training collate call.
        trainer.log_val_metrics(trainer._run_validation_epoch(trainer.val_dataloader))
        batch = collate(load_pages(cfg.data_dir / "train.jsonl")[:1], trainer.processor)
        print("Supervised tokens:", int((batch["labels"][0] != -100).sum()), flush=True)
        trainer.baseline = False
        trainer.started = time.monotonic()
        trainer.run_train_validation_loop()
        checkpoint = sft_checkpoint(output)
        result = artifact(output, "parse/model", checkpoint, stage="sft", base_model=cfg.model,
                          dataset_manifests=identity["manifests"], processor_source=str(checkpoint))
        print(f"SFT model: {result}", flush=True)
        return result


if __name__ == "__main__":
    main()
