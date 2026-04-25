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

"""CPT (Continued Pre-Training) command implementation.

Runs stage1 CPT training using the production training script for the
user's chosen model family (nano3, super3) with customization-specific
YAML configs.

Design: Same scripts, different configs.
- Training script: resolved dynamically based on --model-family
- Config dir: src/nemotron/customization_recipes/nemotron/stage1_cpt/config/
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

# =============================================================================
# Config directory (constant — customization configs are model-family-agnostic)
# =============================================================================

_CUSTOMIZE_CONFIG_DIR = str(
    (Path.cwd() / "src/nemotron/customization_recipes/nemotron/stage1_cpt/config").resolve()
)

# Default SPEC/META using default model family (for --help and registration).
# At execution time, these are rebuilt with the user's actual model family choice.
_DEFAULT_SCRIPT = resolve_training_script("pretrain", DEFAULT_MODEL_FAMILY)
_DEFAULT_SPEC = parse_runspec(_DEFAULT_SCRIPT)

SPEC = _DEFAULT_SPEC
META = RecipeMeta(
    name="customize/cpt",
    script_path=_DEFAULT_SCRIPT,
    config_dir=_CUSTOMIZE_CONFIG_DIR,
    default_config=_DEFAULT_SPEC.config.default,
    input_artifacts={"data": "CPT data (raw text or pre-tokenized)"},
    output_artifacts={"model": "Continued pre-trained model checkpoint"},
)


# =============================================================================
# CLI Entry Point
# =============================================================================


def cpt(
    ctx: typer.Context,
    model_family: str = typer.Option(
        DEFAULT_MODEL_FAMILY,
        "--model-family",
        "-m",
        help="Base model family (nano3, super3). Determines which training script to use.",
    ),
) -> None:
    """Run continued pre-training (stage1).

    Tokenizes data and runs distributed pre-training using Megatron-Bridge.
    The training script is selected based on --model-family.
    """
    # Resolve the correct training script for this model family
    script_path = resolve_training_script("pretrain", model_family)
    base_spec = parse_runspec(script_path)

    # Build a SPEC that uses the model family's script but customization's config dir
    spec = Runspec(
        schema=base_spec.schema,
        docs=base_spec.docs,
        name="customize/cpt",
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
