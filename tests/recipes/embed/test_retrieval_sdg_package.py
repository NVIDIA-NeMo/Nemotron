"""Dependency and ownership checks for the released retrieval SDG package."""

from __future__ import annotations

import tomllib

from .conftest import REPO_ROOT

EMBED_DIR = REPO_ROOT / "src" / "nemotron" / "recipes" / "embed"
PACKAGE_NAME = "data-designer-retrieval-sdg"
PACKAGE_VERSION = "0.2.0"
PACKAGE_INDEX = "https://nvidia-nemo.github.io/DataDesignerPlugins/simple/"
PACKAGE_WHEEL_HASH = "sha256:806d8ecc1775f143f3e02f15662a3c019d9f0f8320f650639e3e0e6b6d07d710"
PACKAGE_SDIST_HASH = "sha256:9497afaa0f20a574dfa8bb97deaa4fed462ec3a75ff6d41f6f2c5c173d2ecdd7"
DATA_DESIGNER_VERSION = "0.9.1"
PACKAGE_STAGES = ("stage0_sdg", "stage1_data_prep")


def test_generation_and_conversion_pin_the_released_package() -> None:
    for stage_name in PACKAGE_STAGES:
        stage_dir = EMBED_DIR / stage_name
        with open(stage_dir / "pyproject.toml", "rb") as file:
            project = tomllib.load(file)
        with open(stage_dir / "uv.lock", "rb") as file:
            lock = tomllib.load(file)

        assert f"{PACKAGE_NAME}=={PACKAGE_VERSION}" in project["project"]["dependencies"]
        assert project["tool"]["uv"]["sources"][PACKAGE_NAME] == {"index": "data-designer-plugins"}
        indexes = {entry["name"]: entry for entry in project["tool"]["uv"]["index"]}
        assert indexes["data-designer-plugins"] == {
            "name": "data-designer-plugins",
            "url": PACKAGE_INDEX,
            "explicit": True,
        }

        package = next(item for item in lock["package"] if item["name"] == PACKAGE_NAME)
        assert package["version"] == PACKAGE_VERSION
        assert package["source"] == {"registry": PACKAGE_INDEX}
        assert package["wheels"][0]["hash"] == PACKAGE_WHEEL_HASH
        assert package["sdist"]["hash"] == PACKAGE_SDIST_HASH

        data_designer = next(item for item in lock["package"] if item["name"] == "data-designer")
        assert data_designer["version"] == DATA_DESIGNER_VERSION


def test_recipe_no_longer_owns_retrieval_sdg_implementations() -> None:
    assert not (EMBED_DIR / "stage0_sdg" / "vendor" / "retriever-sdg").exists()
    assert not (EMBED_DIR / "stage1_data_prep" / "scripts" / "convert_to_retriever_data.py").exists()
