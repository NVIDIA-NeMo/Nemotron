# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/eval"
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

"""Reference dental evaluation of a saved Parse checkpoint on held-out pages."""

import json
from pathlib import Path

from nemotron.recipes.parse.common import (
    artifact,
    model_from_artifact,
    stage_run,
    validate_dataset,
    write_json,
)
from nemotron.recipes.parse.config import EvalConfig, worker_config


def main(cfg: EvalConfig | None = None):
    from nemotron.recipes.parse.dental.evaluate import evaluate_isolated, summarize
    from nemotron.recipes.parse.training.data import load_pages

    cfg = cfg or worker_config("eval")
    identity = validate_dataset(cfg.data_dir)
    model = Path(cfg.model) if cfg.model else model_from_artifact(cfg.model_artifact)
    processor = cfg.processor_source
    if processor is None and cfg.model is None:
        processor = json.loads(cfg.model_artifact.read_text())["processor_source"]
    with stage_run("eval", cfg.output_dir, cfg) as output:
        answers = evaluate_isolated(
            model,
            load_pages(cfg.data_dir / "test.jsonl"),
            output / "predictions",
            samples=cfg.samples,
            processor_source=processor,
        )
        summary = summarize(answers)
        write_json(output / "evaluation.json", summary)
        result = artifact(
            output,
            "parse/evaluation",
            output,
            model=str(model.absolute()),
            evaluation=str(output / "evaluation.json"),
            dataset_manifests=identity["manifests"],
        )
        print(json.dumps(summary, indent=2), flush=True)
        return result


if __name__ == "__main__":
    main()
