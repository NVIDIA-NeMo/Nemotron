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

"""Omni3 RL command implementation.

Omni3 reuses the same Ray/nemo-run execution path as super3 RL while
parameterizing the sub-stage script path and runspec.
"""

from __future__ import annotations

from nemo_runspec.recipe_config import RecipeConfig
from nemotron.cli.commands.super3.rl._base import _execute_rl as _execute_super3_rl


def _execute_rl(cfg: RecipeConfig, script_path: str | None = None, spec=None):
    """Execute an Omni3 RL sub-stage via the shared super3 RL backend."""
    return _execute_super3_rl(cfg, script_path=script_path, spec=spec)
