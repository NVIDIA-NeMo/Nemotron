#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/sft/megatron_bridge"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "nano3"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
# ///
"""Thin Megatron-Bridge SFT wrapper; full recipe: `src/nemotron/recipes/nano3/stage1_sft/train.py`."""
from __future__ import annotations
from pathlib import Path
from megatron.bridge.training.finetune import finetune; from megatron.bridge.training.gpt_step import forward_step
from nemotron.kit.recipe_loader import extract_recipe_config, import_recipe_function; from nemotron.kit.train_script import apply_hydra_overrides, load_omegaconf_yaml, parse_config_and_overrides
DEFAULT_CONFIG = Path(__file__).parent / "config" / "nano3.yaml"; DEFAULT_RECIPE = "megatron.bridge.recipes.nemotronh.nemotron_3_nano.nemotron_3_nano_finetune_config"
def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG); config = apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides)
    target, kwargs = extract_recipe_config(config, default_target=DEFAULT_RECIPE); finetune(config=import_recipe_function(target)(**kwargs), forward_step_func=forward_step)
if __name__ == "__main__":
    main()
