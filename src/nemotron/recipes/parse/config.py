"""Lightweight stage configuration, shared by the CLI and isolated workers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_dir: Path = Path(".nemotron/parse-runtime")
    data_dir: Path = Path(".nemotron/parse/data")


class DataConfig(StageConfig):
    seed: int = 42


class SFTConfig(StageConfig):
    task: str | None = "nemotron.recipes.parse.dental.task.DentalTask"
    output_dir: Path = Path(".nemotron/parse/sft")
    model: str = "nvidia/Nemotron-3.5-Parse"
    training_config: Path = ROOT / "stage1_sft/config/automodel.yaml"
    training: dict[str, Any] = Field(default_factory=dict)


class RLVRConfig(StageConfig):
    task: str = Field(default="nemotron.recipes.parse.dental.task.DentalTask", min_length=1)
    output_dir: Path = Path(".nemotron/parse/rlvr")
    sft_output: Path = Path(".nemotron/parse/sft")
    model: str | None = None
    processor_source: str | None = None
    training_config: Path = ROOT / "stage2_rlvr/config/grpo.yaml"
    training: dict[str, Any] = Field(default_factory=dict)
    candidate_seed: int = 31415
    candidate_contexts: int = Field(default=512, gt=0)
    selection_seed: int = 42
    selection_pages: int = Field(default=256, gt=0)
    mining_samples: int = Field(default=4, gt=0)
    evaluation_samples: int = Field(default=8, gt=0)


class EvalConfig(StageConfig):
    output_dir: Path = Path(".nemotron/parse/eval")
    model: str | None = None
    model_artifact: Path = Path(".nemotron/parse/rlvr/artifact.json")
    processor_source: str | None = None
    samples: int = Field(default=8, gt=0)


STAGES = {
    "data": (DataConfig, "stage0_data", "data_prep"),
    "sft": (SFTConfig, "stage1_sft", "train"),
    "rlvr": (RLVRConfig, "stage2_rlvr", "train"),
    "eval": (EvalConfig, "stage3_eval", "evaluate"),
}


def load_config(stage: str, config: str | Path | None = None, overrides: list[str] | None = None) -> StageConfig:
    model, directory, _ = STAGES[stage]
    config_path = Path(config or "default")
    if not config_path.is_file():
        config_path = ROOT / directory / "config" / f"{config_path}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Parse configuration not found: {config}")
    raw = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(overrides or []))
    return model.model_validate(OmegaConf.to_container(raw, resolve=True))


def worker_config(stage: str) -> StageConfig:
    parser = argparse.ArgumentParser(description=f"Parse {stage} stage")
    parser.add_argument("--config", "-c", default="default")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    return load_config(stage, args.config, args.overrides)
