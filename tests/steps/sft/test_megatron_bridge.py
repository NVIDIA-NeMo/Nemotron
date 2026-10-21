# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/sft/megatron_bridge``."""

from omegaconf import OmegaConf

from nemo_runspec.config.loader import load_config

from .._step_helpers import assert_step_static, step_dir

STEP_DIR = step_dir(__file__, "sft", "megatron_bridge")


def test_sft_megatron_bridge_static() -> None:
    assert_step_static(
        STEP_DIR,
        expected_name="steps/sft/megatron_bridge",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_lightning35_overlay_inherits_default_yaml() -> None:
    cfg = load_config(STEP_DIR / "config" / "lightning35.yaml")
    assert "defaults" not in cfg
    assert cfg.recipe._target_ == (
        "megatron.bridge.recipes.nemotronh.nemotron_3_5_lightning_sft_config"
    )
    assert cfg.recipe.packed_sequence is None
    assert cfg.hf_model_path is None
    assert cfg.load_hf_weights is False
    assert cfg.model.tensor_model_parallel_size == 2
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.dataset.seq_length == 4096
    assert cfg.train.micro_batch_size == 1
    assert cfg.train.global_batch_size == 128
    assert cfg.checkpoint.finetune is True
    assert cfg.run.env.container_image == "nvcr.io/nvidia/nemo:26.08"
    assert OmegaConf.select(cfg, "logger.wandb_exp_name") is not None

