# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/sft/automodel``."""

import tomllib
import yaml

from .._step_helpers import assert_step_static, step_dir


def test_sft_automodel_static() -> None:
    assert_step_static(
        step_dir(__file__, "sft", "automodel"),
        expected_name="steps/sft/automodel",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_sft_automodel_default_is_full_sft_not_peft() -> None:
    directory = step_dir(__file__, "sft", "automodel")
    manifest = tomllib.loads((directory / "step.toml").read_text(encoding="utf-8"))
    config = yaml.safe_load((directory / "config" / "default.yaml").read_text(encoding="utf-8"))

    peft_param = next(param for param in manifest["parameters"] if param["name"] == "peft")
    assert peft_param["default"] == "null"
    assert "peft" not in config
