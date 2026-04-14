#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/eval/model_eval"
#
# [tool.runspec.run]
# launch = "python"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///
"""Thin NeMo Evaluator wrapper; see `Evaluator/docs/deployment/nemo-fw/_snippets/`."""
from __future__ import annotations
from pathlib import Path; from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, ConfigParams, EvaluationConfig, EvaluationTarget; from omegaconf import OmegaConf
from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides
DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG); cfg = OmegaConf.to_container(apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides), resolve=True)
    endpoint = ApiEndpoint(url=cfg["deployment"]["url"], type=cfg["deployment"].get("endpoint_type", "completions"), model_id=cfg["deployment"]["model_id"], api_key=cfg["deployment"].get("api_key_name")); target = EvaluationTarget(api_endpoint=endpoint); params = ConfigParams(**cfg.get("params", {})); check_endpoint(endpoint_url=endpoint.url, endpoint_type=endpoint.type, model_name=endpoint.model_id)
    for benchmark in cfg["benchmarks"]: evaluate(target_cfg=target, eval_cfg=EvaluationConfig(type=benchmark, params=params, output_dir=str(Path(cfg["output_dir"]) / benchmark)))
if __name__ == "__main__":
    main()
