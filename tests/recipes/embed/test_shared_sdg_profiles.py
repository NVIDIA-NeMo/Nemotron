"""Allium-derived contract tests for every shared retrieval SDG consumer."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytest.importorskip("data_designer_retrieval_sdg")

from nemo_runspec.config import load_config
from nemotron.recipes.embed.stage0_sdg.data_prep import SDGConfig
from nemotron.recipes.embed.stage0_sdg.plugin_adapter import build_generation_config
from nemotron.recipes.embed.stage1_data_prep.data_prep import DataPrepConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_PATHS = [
    REPO_ROOT / "src/nemotron/recipes/embed/stage0_sdg/config/default.yaml",
    REPO_ROOT / "src/nemotron/recipes/embed/stage0_sdg/config/llama.yaml",
    REPO_ROOT / "src/nemotron/recipes/rerank/stage0_sdg/config/default.yaml",
]


@pytest.mark.parametrize("profile_path", PROFILE_PATHS, ids=lambda path: str(path.parent.parent.parent))
def test_shared_generation_profile_builds_released_package_config(
    profile_path: Path,
    tmp_path: Path,
) -> None:
    raw_config = OmegaConf.to_container(load_config(profile_path), resolve=True)
    raw_config.pop("run", None)

    recipe_config = SDGConfig.model_validate(raw_config)
    package_config, _ = build_generation_config(recipe_config, tmp_path)

    assert package_config.pipeline.num_pairs == sum(package_config.pipeline.query_counts.values())
    assert package_config.pipeline.num_pairs == sum(package_config.pipeline.reasoning_counts.values())


def test_rerank_preparation_profile_uses_exact_generation_handoff() -> None:
    profile_path = REPO_ROOT / "src/nemotron/recipes/rerank/stage1_prep/config/default.yaml"
    raw_config = OmegaConf.to_container(load_config(profile_path), resolve=True)
    raw_config.pop("run", None)

    recipe_config = DataPrepConfig.model_validate(raw_config)

    assert recipe_config.sdg_input_path.name == "generation_result.json"
