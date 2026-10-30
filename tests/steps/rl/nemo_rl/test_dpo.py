# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/rl/nemo_rl/dpo``."""

from tests.steps._step_helpers import assert_step_static, step_dir


def test_rl_dpo_static() -> None:
    assert_step_static(
        step_dir(__file__, "rl", "nemo_rl", "dpo"),
        expected_name="steps/rl/nemo_rl/dpo",
        expected_launch="ray",
        expected_default_config="default",
        require_workdir=True,
    )
