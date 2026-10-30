# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/rl/nemo_rl/rlvr``."""

import pytest
from omegaconf import OmegaConf

from nemotron.steps._runners.nemo_rl import load_nemo_rl_step_config
from tests.steps._step_helpers import assert_step_static, step_dir

RLVR_STEP_DIR = step_dir(__file__, "rl", "nemo_rl", "rlvr")


def test_rl_rlvr_static() -> None:
    assert_step_static(
        RLVR_STEP_DIR,
        expected_name="steps/rl/nemo_rl/rlvr",
        expected_launch="ray",
        expected_default_config="default",
        require_workdir=True,
    )


@pytest.mark.parametrize("config_name", ["default", "tiny"])
def test_rlvr_sets_force_hf_for_automodel_weight_sync(config_name: str) -> None:
    cfg = load_nemo_rl_step_config(RLVR_STEP_DIR / "config" / f"{config_name}.yaml")

    assert OmegaConf.select(cfg, "policy.dtensor_cfg.automodel_kwargs.force_hf") is True
