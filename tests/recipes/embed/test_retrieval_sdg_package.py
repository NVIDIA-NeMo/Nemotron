"""Dependency and ownership checks for the source-pinned retrieval SDG package."""

from __future__ import annotations

import tomllib

from .conftest import REPO_ROOT

EMBED_DIR = REPO_ROOT / "src" / "nemotron" / "recipes" / "embed"
PACKAGE_NAME = "data-designer-retrieval-sdg"
PACKAGE_VERSION = "0.3.0a1"
DATA_DESIGNER_VERSION = "0.9.1"
PACKAGE_STAGES = ("stage0_sdg", "stage1_data_prep")


def test_generation_and_conversion_pin_the_same_source_package() -> None:
    sources = []
    for stage_name in PACKAGE_STAGES:
        stage_dir = EMBED_DIR / stage_name
        project = tomllib.loads((stage_dir / "pyproject.toml").read_text())
        lock = tomllib.loads((stage_dir / "uv.lock").read_text())
        requirement = PACKAGE_NAME + ("[multimodal]" if stage_name == "stage0_sdg" else "")
        assert requirement in project["project"]["dependencies"]
        assert f"data-designer=={DATA_DESIGNER_VERSION}" in project["project"]["dependencies"]
        source = project["tool"]["uv"]["sources"][PACKAGE_NAME]
        assert source["git"] == "https://github.com/NVIDIA-NeMo/DataDesignerPlugins.git"
        assert source["subdirectory"] == "plugins/data-designer-retrieval-sdg"
        assert len(source["rev"]) == 40 and all(c in "0123456789abcdef" for c in source["rev"])
        sources.append(source)
        package = next(item for item in lock["package"] if item["name"] == PACKAGE_NAME)
        assert package["version"] == PACKAGE_VERSION
        assert package["source"]["git"].endswith(f"#{source['rev']}")
        data_designer = next(item for item in lock["package"] if item["name"] == "data-designer")
        assert data_designer["version"] == DATA_DESIGNER_VERSION
    assert sources[0] == sources[1]


def test_recipe_no_longer_owns_retrieval_sdg_implementations() -> None:
    assert not (EMBED_DIR / "stage0_sdg" / "vendor" / "retriever-sdg").exists()
    assert not (EMBED_DIR / "stage1_data_prep" / "scripts" / "convert_to_retriever_data.py").exists()
