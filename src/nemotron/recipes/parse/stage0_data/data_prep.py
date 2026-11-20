# /// script
# [tool.runspec]
# schema = "1"
# name = "parse/data"
# setup = "Use the isolated Parse runtime; see recipes/parse/README.md."
# [tool.runspec.run]
# launch = "direct"
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "yaml"
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

"""Generate dental pages and publish a verified dataset artifact."""

from nemotron.recipes.parse.common import artifact, dataset_identity, stage_run
from nemotron.recipes.parse.config import DataConfig, worker_config


def main(cfg: DataConfig | None = None):
    from nemotron.recipes.parse.dental.data import generate_dataset

    cfg = cfg or worker_config("data")
    with stage_run("data", cfg.data_dir, cfg) as output:
        generate_dataset(output, seed=cfg.seed)
        identity = dataset_identity(output)
        result = artifact(output, "parse/dataset", output, seed=cfg.seed, identity=identity)
        print(f"Dataset: {result}; pages={identity['pages']}", flush=True)
        return result


if __name__ == "__main__":
    main()
