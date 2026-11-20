# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/eval"
# image = "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
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

"""Compare the configured base, SFT and RLVR models on held-out dental pages."""

import json
from pathlib import Path

from nemotron.recipes.parse.common import (
    artifact,
    stage_run,
    validate_dataset,
    write_json,
)
from nemotron.recipes.parse.config import EvalConfig, worker_config


def main(cfg: EvalConfig | None = None):
    from nemotron.recipes.parse.dental.comparison import evaluate_comparison
    from nemotron.recipes.parse.dental.evaluate import evaluate_isolated, summarize
    from nemotron.recipes.parse.training.data import load_pages
    from nemotron.recipes.parse.training.lineage import read_lineage

    cfg = cfg or worker_config("eval")
    identity = validate_dataset(cfg.data_dir)
    if cfg.model is None:
        if cfg.processor_source is not None:
            raise ValueError("Comparison uses processors recorded in lineage; do not override processor_source")
        lineage = read_lineage(cfg.model_artifact, "rlvr", identity)
        with stage_run("eval", cfg.output_dir, cfg) as output:
            summary = evaluate_comparison(lineage, load_pages(cfg.data_dir / "test.jsonl"), output, cfg.samples)
            result = artifact(output, "parse/evaluation", output,
                              evaluation=str(output / "evaluation.json"),
                              comparison=str(output / "comparison.md"),
                              dataset_manifests=identity["manifests"],
                              lineage=lineage)
            print((output / "comparison.md").read_text(), flush=True)
            return result
    # An explicit local model remains useful for single-checkpoint diagnostics.
    model = Path(cfg.model)
    processor = cfg.processor_source
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
