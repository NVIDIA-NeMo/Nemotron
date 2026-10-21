# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/data_prep/pretrain_prep``."""

from omegaconf import OmegaConf

from .._step_helpers import assert_step_static, step_dir


def test_pretrain_prep_static() -> None:
    assert_step_static(
        step_dir(__file__, "data_prep", "pretrain_prep"),
        expected_name="steps/data_prep/pretrain_prep",
        expected_launch="python",
        expected_default_config="default",
    )


def test_tiny_tokenization_fits_shipped_lepton_cpu_profile() -> None:
    cfg = OmegaConf.load(step_dir(__file__, "data_prep", "pretrain_prep") / "config" / "tiny.yaml")

    # The pipeline reserves one CPU in addition to the tokenization worker;
    # cpu.large provides three CPUs on the shipped NVIDIA Lepton profile.
    assert 1 + cfg.tokenization.cpus_per_worker <= 3
