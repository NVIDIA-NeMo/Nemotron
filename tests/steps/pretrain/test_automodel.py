# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/pretrain/automodel``."""

from .._step_helpers import assert_step_static, step_dir


def test_pretrain_automodel_static() -> None:
    assert_step_static(
        step_dir(__file__, "pretrain", "automodel"),
        expected_name="steps/pretrain/automodel",
        expected_launch="torchrun",
        expected_default_config="default",
    )
