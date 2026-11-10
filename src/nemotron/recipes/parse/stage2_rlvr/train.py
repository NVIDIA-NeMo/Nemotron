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

"""Run the reference RLVR notebook as an isolated, configurable recipe stage."""

import json
from pathlib import Path

from omegaconf import OmegaConf

from nemotron.recipes.parse.common import (
    Progress,
    artifact,
    model_from_artifact,
    stage_run,
    summarize,
    validate_dataset,
    write_json,
)
from nemotron.recipes.parse.config import RLVRConfig, worker_config


def main(cfg: RLVRConfig | None = None):
    from nemotron.recipes.parse.dental.config import rlvr_config
    from nemotron.recipes.parse.dental.curriculum import select_training_pages
    from nemotron.recipes.parse.dental.data import load_pages, make_records, save_pages
    from nemotron.recipes.parse.dental.evaluate import evaluate_isolated
    from nemotron.recipes.parse.dental.model import prepare_model
    from nemotron.recipes.parse.dental.rlvr import train

    cfg = cfg or worker_config("rlvr")
    identity = validate_dataset(cfg.data_dir)
    sft_model = Path(cfg.model) if cfg.model else model_from_artifact(cfg.sft_output / "artifact.json")
    processor = Path(cfg.processor_source) if cfg.processor_source else sft_model
    data = cfg.data_dir.absolute()
    monitoring, test = load_pages(data / "monitor.jsonl"), load_pages(data / "test.jsonl")
    with stage_run("rlvr", cfg.output_dir, cfg) as output:
        # Mining manifests belong to this run, keeping the fixed dataset immutable.
        curriculum = output / "curriculum"
        curriculum.mkdir()
        manifest = save_pages(make_records(seed=cfg.candidate_seed, contexts=cfg.candidate_contexts),
                              curriculum, "rl_candidates")
        candidates = load_pages(manifest)
        sampled = evaluate_isolated(sft_model, candidates, output / "mining",
                                    samples=cfg.mining_samples, processor_source=processor)
        selected = select_training_pages(candidates, sampled, pages=cfg.selection_pages,
                                         seed=cfg.selection_seed,
                                         excluded=load_pages(data / "train.jsonl") + monitoring + test)
        for row in selected:
            row["image"] = str(Path(row["image"]).relative_to(curriculum))
        (curriculum / "rl_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        # The native configuration expects monitor.jsonl next to rl_train.jsonl.
        # Keep image paths absolute when relocating this manifest.
        (curriculum / "monitor.jsonl").write_text("".join(json.dumps(row) + "\n" for row in monitoring))
        before = evaluate_isolated(sft_model, test, output / "before",
                                   samples=cfg.evaluation_samples, processor_source=processor)
        model = prepare_model(sft_model, output / "initial-model", scope="decoder", processor_source=processor)
        raw = rlvr_config(cfg.training_config, model, curriculum, output)
        config = OmegaConf.to_container(OmegaConf.merge(raw, cfg.training), resolve=True)
        write_json(output / "training-config.json", config)
        checkpoint = train(config, observer=Progress(output, "rlvr"))
        after = evaluate_isolated(checkpoint, test, output / "after",
                                  samples=cfg.evaluation_samples, processor_source=processor)
        if [row["id"] for row in before] != [row["id"] for row in after]:
            raise ValueError("Before/after evaluation must use the same ordered pages")
        summary = dict(before=summarize(before), after=summarize(after))
        write_json(output / "evaluation.json", summary)
        result = artifact(output, "parse/model", checkpoint, stage="rlvr", sft_model=str(sft_model.absolute()),
                          dataset_manifests=identity["manifests"], processor_source=str(processor.absolute()),
                          evaluation=str(output / "evaluation.json"))
        print(json.dumps(summary, indent=2), flush=True)
        print(f"RLVR model: {result}", flush=True)
        return result


if __name__ == "__main__":
    main()
