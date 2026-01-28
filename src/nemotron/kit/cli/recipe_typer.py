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

"""RecipeTyper class for recipe command registration.

Provides RecipeTyper, a typer.Typer subclass with recipe_command() decorator
method that handles context_settings and help panel configuration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import typer


class RecipeTyper(typer.Typer):
    """Typer subclass for recipe commands.

    Extends typer.Typer with recipe_command() method that automatically
    configures context_settings and help panels for recipe commands.
    """

    def recipe_command(
        self,
        config_dir: str,
        artifact_overrides: dict[str, str] | None = None,
        run_fn: Callable | None = None,
        config_model: type | None = None,
        **kwargs: Any,
    ) -> Callable:
        """Register recipe command with proper context_settings and help.

        Args:
            config_dir: Path to config directory for this command
            artifact_overrides: Optional artifact slot descriptions for help
            run_fn: Optional _*_nemo_run function for pipe composition
            config_model: Optional Pydantic BaseModel class for config options help
            **kwargs: Additional arguments passed to typer.command()

        Returns:
            Decorator function for registering the command

        Example:
            app = RecipeTyper(...)
            app.recipe_command(
                config_dir="src/nemotron/recipes/embed/stage2_finetune/config",
                artifact_overrides={"data": "Training data artifact"},
                run_fn=finetune._finetune_nemo_run,
                config_model=finetune.CONFIG_MODEL,
            )(finetune.finetune)
        """
        # Import here to avoid circular dependency
        from nemotron.cli.embed.help import make_recipe_command

        # Build context_settings with allow_extra_args
        context_settings = kwargs.pop("context_settings", {})
        context_settings.setdefault("allow_extra_args", True)
        context_settings.setdefault("ignore_unknown_options", True)

        # Build cls with custom help panels
        cls = kwargs.pop("cls", None) or make_recipe_command(
            artifact_overrides=artifact_overrides,
            config_dir=config_dir,
            config_model=config_model,
        )

        def decorator(func: Callable) -> Callable:
            # Attach run_fn for pipe composition if provided
            if run_fn:
                func._run_fn = run_fn

            # Register command with typer
            return self.command(
                func.__name__,
                context_settings=context_settings,
                cls=cls,
                **kwargs,
            )(func)

        return decorator
