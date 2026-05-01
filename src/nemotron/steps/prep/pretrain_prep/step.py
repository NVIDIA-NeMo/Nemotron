#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/prep/pretrain_prep"
# image = "anyscale/ray:2.49.2-py312"
#
# [tool.runspec.run]
# launch = "python"
# cmd = "uv run --extra xenna python {script} --config {config}"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 0
# ///

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

"""Thin pretrain bin/idx wrapper. Tokenises HF/local text into Megatron bin/idx + blend.json.

Mirrors the runtime safety pattern from prep/sft_packing:
  * Falls back to a self-contained ``data/blend_tiny.json`` when YAML omits
    ``blend_path``, so the same config works under any source layout
    (local / slurm /nemo_run/code / lepton /mnt/lustre-shared/_nemotron).
  * ``chdir`` to a scratch dir before xenna's ``ray.init()`` so Ray's
    working_dir auto-upload doesn't try to package the cwd (which on Lepton
    contains accumulated wandb logs that easily exceed Ray's 512 MiB cap).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from nemotron.data_prep import (
    DataBlend,
    ObservabilityConfig,
    TokenizerConfig,
    run_pretrain_pipeline,
)
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)
from nemotron.steps.prep._common import (
    chdir_to_scratch,
    config_dataclass,
    init_prep_wandb,
    resolve_blend_path,
    resolve_output_dir,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
STEP_DIR = Path(__file__).parent


def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = OmegaConf.to_container(
        apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides),
        resolve=True,
    )

    from nemotron.data_prep.stages import (
        BinIdxTokenizationStageConfig,
        DownloadStageConfig,
        PlanStageConfig,
    )

    blend_path = resolve_blend_path(cfg, step_dir=STEP_DIR)
    output_dir = resolve_output_dir(cfg["output_dir"])

    chdir_to_scratch("nemotron-pretrain-prep-")
    init_prep_wandb(["data-prep", "pretrain", cfg.get("config_name", "pretrain-prep")])

    run_pretrain_pipeline(
        blend=DataBlend.load(blend_path),
        output_dir=output_dir,
        tokenizer=TokenizerConfig(**cfg["tokenizer"]),
        num_shards=cfg.get("num_shards", 128),
        dtype=cfg.get("dtype", "int32"),
        text_field_default=cfg.get("text_field", "text"),
        min_doc_chars=cfg.get("min_doc_chars"),
        max_doc_tokens=cfg.get("max_doc_tokens"),
        max_rows=cfg.get("max_rows"),
        sample=cfg.get("sample"),
        sample_seed=cfg.get("sample_seed", 42),
        force=cfg.get("force", False),
        execution_mode=cfg.get("execution_mode", "auto"),
        plan_stage=config_dataclass(PlanStageConfig, cfg.get("plan")),
        download_stage=config_dataclass(DownloadStageConfig, cfg.get("download")),
        tokenization_stage=config_dataclass(BinIdxTokenizationStageConfig, cfg.get("tokenization")),
        observability=ObservabilityConfig(**cfg.get("observability", {})),
    )

if __name__ == "__main__":
    main()
