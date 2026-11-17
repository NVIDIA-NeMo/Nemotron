# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/rlvr"
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

"""Mine candidates, train with RLVR and evaluate the resulting checkpoint."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemotron.recipes.parse.common import (
    Progress,
    artifact,
    model_from_artifact,
    stage_run,
    validate_dataset,
    write_json,
)
from nemotron.recipes.parse.config import RLVRConfig, worker_config


def main(cfg: RLVRConfig | None = None):
    from nemotron.recipes.parse.training.config import rlvr_config
    from nemotron.recipes.parse.training.data import load_pages
    from nemotron.recipes.parse.training.lineage import model_reference, read_lineage, verify_model
    from nemotron.recipes.parse.training.model import prepare_model
    from nemotron.recipes.parse.training.rlvr import train
    from nemotron.recipes.parse.training.task import load_task

    cfg = cfg or worker_config("rlvr")
    task = load_task(cfg.task, stage="rlvr")
    identity = validate_dataset(cfg.data_dir)
    sft_artifact = cfg.sft_output / "artifact.json"
    lineage = read_lineage(sft_artifact, "sft", identity)
    sft_model = Path(cfg.model) if cfg.model else model_from_artifact(cfg.sft_output / "artifact.json")
    parent = lineage["models"]["sft"]
    processor = Path(cfg.processor_source or parent["processor_source"])
    if sft_model.resolve() != Path(parent["path"]) or processor.resolve() != Path(parent["processor_source"]):
        raise ValueError("RLVR model and processor must match the SFT artifact's lineage")
    data = cfg.data_dir.absolute()
    monitoring, test = load_pages(data / "monitor.jsonl"), load_pages(data / "test.jsonl")
    with stage_run("rlvr", cfg.output_dir, cfg) as output:
        # Mining manifests belong to this run, keeping the fixed dataset immutable.
        curriculum = output / "curriculum"
        curriculum.mkdir()
        candidates = task.create_candidates(curriculum, seed=cfg.candidate_seed, contexts=cfg.candidate_contexts)
        sampled = task.evaluate(
            sft_model, candidates, output / "mining", samples=cfg.mining_samples, processor_source=processor
        )
        selected = task.select_candidates(
            candidates,
            sampled,
            pages=cfg.selection_pages,
            seed=cfg.selection_seed,
            excluded=load_pages(data / "train.jsonl") + monitoring + test,
        )
        for row in selected:
            row["image"] = str(Path(row["image"]).relative_to(curriculum))
        (curriculum / "rl_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        # The native configuration expects monitor.jsonl next to rl_train.jsonl.
        # Keep image paths absolute when relocating this manifest.
        (curriculum / "monitor.jsonl").write_text("".join(json.dumps(row) + "\n" for row in monitoring))
        before = task.evaluate(
            sft_model, test, output / "before", samples=cfg.evaluation_samples, processor_source=processor
        )
        model = prepare_model(sft_model, output / "initial-model", scope="decoder", processor_source=processor)
        raw = rlvr_config(cfg.training_config, model, curriculum, output, prompt_file=task.prompt_file)
        config = OmegaConf.to_container(OmegaConf.merge(raw, cfg.training), resolve=True)
        write_json(output / "training-config.json", config)
        checkpoint = train(
            config,
            observer=Progress(output, "rlvr"),
            environment_class=task.environment_class,
            reward_metric=task.reward_metric,
        )
        after = task.evaluate(
            checkpoint, test, output / "after", samples=cfg.evaluation_samples, processor_source=processor
        )
        if [row["id"] for row in before] != [row["id"] for row in after]:
            raise ValueError("Before/after evaluation must use the same ordered pages")
        summary = dict(before=task.summarize(before), after=task.summarize(after))
        write_json(output / "evaluation.json", summary)
        for reference in lineage["models"].values():
            verify_model(reference)
        lineage["models"]["rlvr"] = model_reference(checkpoint, processor)
        result = artifact(
            output,
            "parse/model",
            checkpoint,
            stage="rlvr",
            sft_model=str(sft_model.absolute()),
            sft_artifact=str(sft_artifact.resolve()),
            lineage=lineage,
            dataset_manifests=identity["manifests"],
            processor_source=str(processor.resolve()),
            evaluation=str(output / "evaluation.json"),
        )
        print(json.dumps(summary, indent=2), flush=True)
        print(f"RLVR model: {result}", flush=True)
        return result


if __name__ == "__main__":
    main()
