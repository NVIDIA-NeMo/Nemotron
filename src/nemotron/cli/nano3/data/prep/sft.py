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

"""SFT data preparation command."""

from __future__ import annotations

import typer

from nemotron.kit.cli.recipe import recipe


@recipe(
    name="nano3/data/prep/sft",
    script_path="src/nemotron/recipes/nano3/stage1_sft/data_prep.py",
    config_dir="src/nemotron/recipes/nano3/stage1_sft/config/data_prep",
    default_config="default",
    torchrun=False,
    ray=True,
    packager="code",
)
def sft(ctx: typer.Context) -> None:
    """Prepare data for SFT (packed .npy format).

    Applies chat templates to OpenAI-format messages, tokenizes with role-based
    loss masking, and outputs packed .npy files compatible with GPTSFTPackedDataset.

    Config sources merged in order:
    1. Default config (default.yaml)
    2. Named config via -c/--config
    3. env.toml profile via --run/--batch (merged into run.env)
    4. CLI dotlist overrides (e.g., sample=1000)

    Examples:
        nemotron nano3 data prep sft                   # local execution
        nemotron nano3 data prep sft sample=1000       # with sampling
        nemotron nano3 data prep sft --config tiny     # use tiny config
        nemotron nano3 data prep sft --run prep        # nemo-run attached
        nemotron nano3 data prep sft --dry-run         # preview config
    """
    ...
