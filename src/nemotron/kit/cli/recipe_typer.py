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

"""RecipeTyper - standardizes recipe command registration.

This module provides RecipeTyper, a Typer subclass that handles the
boilerplate for registering recipe commands:
- context_settings for allowing extra args (dotlist overrides)
- Rich help panels via make_recipe_command

RecipeTyper is ONLY for registration boilerplate. It does NOT handle
execution - that logic remains visible in each command file.

Design principle: keep registration DRY while keeping execution explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import typer

from nemotron.kit.cli.help import make_recipe_command


@dataclass(frozen=True)
class RecipeMeta:
    """All recipe metadata in one place.

    This bundles everything about a recipe:
    - name: Recipe identifier (e.g., "nano3/pretrain")
    - script_path: Path to the training script
    - config_dir: Where to find config files
    - default_config: Default config name
    - input_artifacts: What data the recipe consumes
    - output_artifacts: What the recipe produces

    Example:
        META = RecipeMeta(
            name="nano3/pretrain",
            script_path="src/nemotron/recipes/nano3/stage0_pretrain/train.py",
            config_dir="src/nemotron/recipes/nano3/stage0_pretrain/config",
            default_config="default",
            input_artifacts={"data": "Pretrain data artifact (bin/idx blends)"},
            output_artifacts={"model": "Pretrained model checkpoint"},
        )
    """

    name: str
    script_path: str
    config_dir: str
    default_config: str = "default"
    input_artifacts: dict[str, str] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(default_factory=dict)


class RecipeTyper(typer.Typer):
    """Typer subclass that standardizes recipe command registration.

    This class handles the common registration pattern for recipe commands:
    - Setting context_settings to allow extra args
    - Adding rich help panels via make_recipe_command

    It does NOT handle execution logic - that remains visible in each
    command file, enabling easy forking and modification.

    Example:
        app = RecipeTyper(name="nano3", help="Nano3 training recipes")

        app.add_recipe_command(
            pretrain,
            config_dir=CONFIG_DIR,
            input_artifacts={"data": "Pretrain data artifact"},
        )
    """

    def recipe_command(
        self,
        *,
        config_dir: str | None = None,
        input_artifacts: dict[str, str] | None = None,
        output_artifacts: dict[str, str] | None = None,
        rich_help_panel: str | None = None,
        name: str | None = None,
    ) -> Callable[[Callable], Callable]:
        """Register a recipe command with proper context_settings and help.

        Args:
            config_dir: Path to config directory (for help panel listing configs).
            input_artifacts: Dict of input artifact names to descriptions.
                            Displayed in --help as artifact overrides.
            output_artifacts: Dict of output artifact names to descriptions.
                             For documentation (not currently displayed in help).
            rich_help_panel: Panel name for grouping in --help.
            name: Command name. If None, uses the function name.

        Returns:
            Decorator that registers the command.
        """
        # Input artifacts are shown in help as overridable
        effective_artifact_overrides = dict(input_artifacts or {})

        def decorator(fn: Callable) -> Callable:
            # Determine command name
            cmd_name = name if name is not None else fn.__name__

            # Build command class with rich help
            cmd_class = make_recipe_command(
                artifact_overrides=effective_artifact_overrides,
                config_dir=config_dir,
            )

            # Register with standard recipe context settings
            return self.command(
                name=cmd_name,
                context_settings={
                    "allow_extra_args": True,
                    "ignore_unknown_options": True,
                },
                rich_help_panel=rich_help_panel,
                cls=cmd_class,
            )(fn)

        return decorator

    def add_recipe_command(
        self,
        fn: Callable,
        *,
        meta: RecipeMeta | None = None,
        config_dir: str | None = None,
        input_artifacts: dict[str, str] | None = None,
        output_artifacts: dict[str, str] | None = None,
        rich_help_panel: str | None = None,
        name: str | None = None,
    ) -> None:
        """Non-decorator variant of recipe_command for explicit registration.

        Useful when you want to register an already-defined function:

            app.add_recipe_command(pretrain, meta=PRETRAIN_META)

        Or with individual arguments:

            app.add_recipe_command(
                pretrain,
                config_dir=CONFIG_DIR,
                input_artifacts=INPUT_ARTIFACTS,
            )
        """
        # Extract from meta if provided
        if meta is not None:
            config_dir = meta.config_dir
            input_artifacts = meta.input_artifacts
            output_artifacts = meta.output_artifacts

        self.recipe_command(
            config_dir=config_dir,
            input_artifacts=input_artifacts,
            output_artifacts=output_artifacts,
            rich_help_panel=rich_help_panel,
            name=name,
        )(fn)
