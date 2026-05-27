"""Validate embed stage CUDA torch source metadata."""

from __future__ import annotations

from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
EMBED_DIR = REPO_ROOT / "src" / "nemotron" / "recipes" / "embed"
TORCH_STAGE_NAMES = [
    "stage0_sdg",
    "stage1_data_prep",
    "stage2_finetune",
    "stage3_eval",
    "stage4_export",
]


def test_embed_torch_stages_pin_linux_torch_to_cu129() -> None:
    cu129_source = {"index": "pytorch-cu129", "marker": "sys_platform == 'linux'"}

    for stage_name in TORCH_STAGE_NAMES:
        with open(EMBED_DIR / stage_name / "pyproject.toml", "rb") as f:
            data = tomllib.load(f)

        torch_sources = data["tool"]["uv"]["sources"]["torch"]
        assert cu129_source in torch_sources

        if stage_name == "stage2_finetune":
            assert {
                "index": "pytorch-cpu",
                "marker": "sys_platform != 'darwin' and sys_platform != 'linux'",
            } in torch_sources

        if stage_name == "stage4_export":
            assert data["tool"]["uv"]["sources"]["torchvision"] == [cu129_source]

        indexes = {entry["name"]: entry["url"] for entry in data["tool"]["uv"]["index"]}
        assert indexes["pytorch-cu129"] == "https://download.pytorch.org/whl/cu129"
        if stage_name == "stage2_finetune":
            assert indexes["pytorch-cpu"] == "https://download.pytorch.org/whl/cpu"
        assert "pytorch-cu130" not in indexes


def test_embed_export_lock_pins_linux_torchvision_to_cu129() -> None:
    with open(EMBED_DIR / "stage4_export" / "uv.lock", "rb") as f:
        data = tomllib.load(f)

    torchvision_packages = [
        package
        for package in data["package"]
        if package["name"] == "torchvision"
    ]

    assert any(
        package["version"].endswith("+cu129")
        and package["source"]["registry"] == "https://download.pytorch.org/whl/cu129"
        for package in torchvision_packages
    )


def test_embed_finetune_and_export_stages_limit_python_to_312() -> None:
    for stage_name in ("stage2_finetune", "stage4_export"):
        with open(EMBED_DIR / stage_name / "pyproject.toml", "rb") as f:
            pyproject_data = tomllib.load(f)
        with open(EMBED_DIR / stage_name / "uv.lock", "rb") as f:
            lock_data = tomllib.load(f)

        assert pyproject_data["project"]["requires-python"] == ">=3.12,<3.13"
        assert lock_data["requires-python"] == "==3.12.*"


def test_embed_export_stage_matches_finetune_transformers_range() -> None:
    with open(EMBED_DIR / "stage4_export" / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    assert data["tool"]["uv"]["override-dependencies"] == ["transformers>=5.0,<5.2"]


def test_embed_export_lock_matches_finetune_transformers_range() -> None:
    with open(EMBED_DIR / "stage4_export" / "uv.lock", "rb") as f:
        data = tomllib.load(f)

    assert data["requires-python"] == "==3.12.*"

    transformer_packages = [
        package
        for package in data["package"]
        if package["name"] == "transformers"
    ]
    assert [package["version"] for package in transformer_packages] == ["5.1.0"]

    assert not any(
        package["name"] == "nvidia-resiliency-ext"
        and package["version"] == "0.5.0"
        for package in data["package"]
    )
