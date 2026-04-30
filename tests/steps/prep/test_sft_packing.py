# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Static checks for ``steps/prep/sft_packing``.

This step pre-existed the generic CLI work but is important for the agentic
pipeline (it produces ``packed_parquet`` consumed by sft/megatron_bridge).
"""

from .._step_helpers import assert_step_static, step_dir


def test_sft_packing_static() -> None:
    assert_step_static(
        step_dir(__file__, "prep", "sft_packing"),
        expected_name="steps/prep/sft_packing",
        expected_launch="python",
        expected_default_config="default",
    )
