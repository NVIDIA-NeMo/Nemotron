# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/pretrain/automodel``."""

import tomllib
import yaml

from .._step_helpers import assert_step_static, step_dir


def test_pretrain_automodel_static() -> None:
    assert_step_static(
        step_dir(__file__, "pretrain", "automodel"),
        expected_name="steps/pretrain/automodel",
        expected_launch="torchrun",
        expected_default_config="default",
    )


def test_pretrain_automodel_model_contract_matches_default_config() -> None:
    directory = step_dir(__file__, "pretrain", "automodel")
    manifest = tomllib.loads((directory / "step.toml").read_text(encoding="utf-8"))
    config = yaml.safe_load((directory / "config" / "default.yaml").read_text(encoding="utf-8"))

    default_models = [model["name"] for model in manifest["models"] if model.get("default")]
    assert default_models == [config["model"]["pretrained_model_name_or_path"]]
    assert manifest["parameters"][0]["default"] == default_models[0]
    assert config["dataset"]["tokenizer"]["pretrained_model_name_or_path"] == default_models[0]
    assert config["validation_dataset"]["tokenizer"]["pretrained_model_name_or_path"] == default_models[0]
