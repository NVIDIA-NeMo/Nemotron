#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/prep/sft_packing"
# image = "anyscale/ray:2.49.2-py312"
#
# [tool.runspec.run]
# launch = "ray"
# cmd = "uv run --extra xenna python {script} --config {config}"
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
"""Thin SFT packing wrapper; full recipe: `src/nemotron/recipes/nano3/stage1_sft/data_prep.py`."""
from __future__ import annotations
from pathlib import Path; from omegaconf import OmegaConf
from nemotron.data_prep import DataBlend, ObservabilityConfig, TokenizerConfig, run_sft_pipeline; from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides
DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG); cfg = OmegaConf.to_container(apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides), resolve=True)
    run_sft_pipeline(blend=DataBlend.load(cfg["blend_path"]), output_dir=cfg["output_dir"], tokenizer=TokenizerConfig(**cfg["tokenizer"]), num_shards=cfg.get("num_shards", 128), pack_size=cfg.get("pack_size", 4096), algorithm=cfg.get("algorithm", "first_fit_shuffle"), chat_template=cfg.get("chat_template", "nano3"), messages_field_default=cfg.get("messages_field", "messages"), tools_field_default=cfg.get("tools_field", "tools"), sample=cfg.get("sample"), sample_seed=cfg.get("sample_seed", 42), force=cfg.get("force", False), observability=ObservabilityConfig(**cfg.get("observability", {})))
if __name__ == "__main__":
    main()
