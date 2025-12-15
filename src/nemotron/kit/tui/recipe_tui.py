from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from nemotron.kit.artifact import ArtifactInput


@dataclass(frozen=True)
class RecipeTuiMeta:
    recipe_name: str
    script_path: str
    config_dir: str
    default_config: str
    artifacts: dict[str, ArtifactInput]


@dataclass(frozen=True)
class RecipeTuiResult:
    train_config: DictConfig
    config_path: str
    profile: str | None
    detached: bool


def run_recipe_tui(meta: RecipeTuiMeta) -> RecipeTuiResult | None:
    from nemotron.kit.tui.recipe_tui_app import RecipeTUIApp

    return RecipeTUIApp(meta).run()
