# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static checks for ``steps/optimize/modelopt/prune``."""

from tests.steps._step_helpers import assert_step_static, step_dir


def test_optimize_modelopt_prune_static() -> None:
    assert_step_static(
        step_dir(__file__, "optimize", "modelopt", "prune"),
        expected_name="steps/optimize/modelopt/prune",
        expected_launch="python",
        expected_default_config="default",
        require_workdir=True,
    )
