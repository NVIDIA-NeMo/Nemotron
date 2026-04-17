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

"""SFT (Supervised Fine-Tuning) command implementation.

Runs stage2 SFT training using the production training script for the
user's chosen model family (nano3, super3) with customization-specific
YAML configs.

Design: Same scripts, different configs.
- Training script: resolved dynamically based on --model-family
- Config dir: src/nemotron/customization_recipes/nemotron/stage2_sft/config/
"""

from __future__ import annotations

from pathlib import Path

import typer

from nemo_runspec import parse as parse_runspec
from nemo_runspec._models import Runspec, RunspecConfig
from nemo_runspec.recipe_config import parse_recipe_config
from nemo_runspec.recipe_typer import RecipeMeta

from nemotron.cli.commands.customize._execute import (
    DEFAULT_MODEL_FAMILY,
    execute_recipe,
    resolve_training_script,
)

_CUSTOMIZE_CONFIG_DIR = str(
    (Path.cwd() / "src/nemotron/customization_recipes/nemotron/stage2_sft/config").resolve()
)

_DEFAULT_SCRIPT = resolve_training_script("sft", DEFAULT_MODEL_FAMILY)
_DEFAULT_SPEC = parse_runspec(_DEFAULT_SCRIPT)

SPEC = _DEFAULT_SPEC
META = RecipeMeta(
    name="customize/sft",
    script_path=_DEFAULT_SCRIPT,
    config_dir=_CUSTOMIZE_CONFIG_DIR,
    default_config=_DEFAULT_SPEC.config.default,
    input_artifacts={
        "model": "Base model checkpoint (from pretrain or CPT)",
        "data": "SFT data (packed chat-format .npy)",
    },
    output_artifacts={"model": "Fine-tuned model checkpoint"},
)


def sft(
    ctx: typer.Context,
    model_family: str = typer.Option(
        DEFAULT_MODEL_FAMILY,
        "--model-family",
        "-m",
        help="Base model family (nano3, super3). Determines which training script to use.",
    ),
) -> None:
    """Run supervised fine-tuning (stage2).

    Fine-tunes a Nemotron model on chat-format data using Megatron-Bridge.
    The training script is selected based on --model-family.
    """
    script_path = resolve_training_script("sft", model_family)
    base_spec = parse_runspec(script_path)

    spec = Runspec(
        schema=base_spec.schema,
        docs=base_spec.docs,
        name="customize/sft",
        image=base_spec.image,
        setup=base_spec.setup,
        run=base_spec.run,
        config=RunspecConfig(
            dir=_CUSTOMIZE_CONFIG_DIR,
            default=base_spec.config.default,
            format=base_spec.config.format,
        ),
        resources=base_spec.resources,
        env=base_spec.env,
        script_path=base_spec.script_path,
    )

    cfg = parse_recipe_config(ctx)
    execute_recipe(cfg, spec, script_path)
