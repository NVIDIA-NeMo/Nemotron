# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/prep/rl_prep``."""

from .._step_helpers import assert_step_static, step_dir


def test_rl_prep_static() -> None:
    assert_step_static(
        step_dir(__file__, "prep", "rl_prep"),
        expected_name="steps/prep/rl_prep",
        expected_launch="python",
        expected_default_config="default",
    )
