#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# name = "steps/prep/sft_packing"
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

"""Thin SFT packing wrapper; full recipe: `src/nemotron/recipes/nano3/stage1_sft/data_prep.py`."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from nemotron.data_prep import (
    DataBlend,
    ObservabilityConfig,
    TokenizerConfig,
    run_sft_pipeline,
)
from nemotron.kit.train_script import (
    apply_hydra_overrides,
    load_omegaconf_yaml,
    parse_config_and_overrides,
)
from nemotron.steps.prep._common import (
    chdir_to_scratch,
    config_dataclass,
    resolve_blend_path,
    resolve_output_dir,
)

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"
STEP_DIR = Path(__file__).parent


def _ratio_to_shards(ratio: float, total: int) -> int:
    if ratio <= 0.0:
        return 0
    return max(1, int(round(total * ratio)))


def _split_shard_counts(cfg: dict, total: int) -> tuple[int, int]:
    if "valid_shards" in cfg or "test_shards" in cfg:
        return int(cfg.get("valid_shards", 1)), int(cfg.get("test_shards", 1))
    train_ratio = float(cfg.get("train_ratio", 0.98))
    valid_ratio = float(cfg.get("valid_ratio", 0.01))
    test_ratio = float(cfg.get("test_ratio", 0.01))
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + valid_ratio + test_ratio must equal 1.0; got {ratio_sum}")
    return (
        _ratio_to_shards(valid_ratio, total),
        _ratio_to_shards(test_ratio, total),
    )


def _materialize_split_dirs(result, output_dir: Path, *, cfg: dict, seed: int) -> None:
    """Create the canonical ``output_dir/splits/{train,valid,test}/`` layout.

    ``run_sft_pipeline`` writes packed parquet shards under
    ``runs/<hash>/datasets/<name>/<hash>_*.parquet``; downstream training
    configs expect a stable ``splits/<split>/`` view. This mirrors the
    recipe-level orchestration in ``recipes/nano3/stage1_sft/data_prep.py``
    so the generic step's output is usable without forking.
    """
    from nemotron.data_prep.utils.splits import (
        distribute_shards_to_splits,
        realize_packed_shards_into_split_dirs,
    )

    if not result.data_paths:
        return  # cache hit / no shards produced this run

    total = result.num_shards
    valid_shards, test_shards = _split_shard_counts(cfg, total)
    # Mirror the recipe's ratio→shards math, capped so train always gets ≥1.
    if total <= 2:
        valid_shards = 0
        test_shards = 0
    else:
        max_non_train = total - 1
        if valid_shards + test_shards > max_non_train:
            scale = max_non_train / max(valid_shards + test_shards, 1)
            valid_shards = max(0, int(valid_shards * scale))
            test_shards = max(0, max_non_train - valid_shards)

    blend_data = distribute_shards_to_splits(
        data_paths=result.data_paths,
        num_shards=total,
        valid_shards=valid_shards,
        test_shards=test_shards,
        seed=seed,
    )
    realize_packed_shards_into_split_dirs(
        output_dir=output_dir,
        split_to_paths=blend_data,
    )


def main() -> None:
    config_path, overrides = parse_config_and_overrides(default_config=DEFAULT_CONFIG)
    cfg = OmegaConf.to_container(
        apply_hydra_overrides(load_omegaconf_yaml(config_path), overrides),
        resolve=True,
    )

    from nemotron.data_prep.stages.download import DownloadStageConfig
    from nemotron.data_prep.stages.packed_sft_parquet import PackedSftParquetStageConfig
    from nemotron.data_prep.stages.sft_plan import SftPlanStageConfig

    blend_path = resolve_blend_path(cfg, step_dir=STEP_DIR)
    output_dir = resolve_output_dir(cfg["output_dir"])

    # Switch cwd after resolving local paths. Xenna/Ray otherwise packages cwd as
    # its runtime working_dir, which can exceed Ray's upload cap on shared mounts.
    chdir_to_scratch("nemotron-sft-packing-")

    result = run_sft_pipeline(
        blend=DataBlend.load(blend_path),
        output_dir=output_dir,
        tokenizer=TokenizerConfig(**cfg["tokenizer"]),
        num_shards=cfg.get("num_shards", 128),
        dtype=cfg.get("dtype", "int32"),
        pack_size=cfg.get("pack_size", 4096),
        algorithm=cfg.get("algorithm", "first_fit_shuffle"),
        seed=cfg.get("seed"),
        chat_template=cfg.get("chat_template", "nano3"),
        messages_field_default=cfg.get("messages_field", "messages"),
        tools_field_default=cfg.get("tools_field", "tools"),
        used_in_filter=cfg.get("used_in_filter"),
        used_in_field=cfg.get("used_in_field", "used_in"),
        parquet_row_group_size=cfg.get("parquet_row_group_size", 1000),
        parquet_compression=cfg.get("parquet_compression", "zstd"),
        max_doc_tokens=cfg.get("max_doc_tokens"),
        max_rows=cfg.get("max_rows"),
        sample=cfg.get("sample"),
        sample_seed=cfg.get("sample_seed", 42),
        force=cfg.get("force", False),
        execution_mode=cfg.get("execution_mode", "auto"),
        plan_stage=config_dataclass(SftPlanStageConfig, cfg.get("plan")),
        download_stage=config_dataclass(DownloadStageConfig, cfg.get("download")),
        tokenization_stage=config_dataclass(PackedSftParquetStageConfig, cfg.get("tokenization")),
        observability=ObservabilityConfig(**cfg.get("observability", {})),
    )

    # Default 1 valid + 1 test shard so downstream training has the canonical
    # splits/{train,valid,test}/ layout to point at. Configurable via YAML.
    _materialize_split_dirs(
        result,
        output_dir=Path(result.output_dir),
        cfg=cfg,
        seed=cfg.get("sample_seed", 42),
    )


if __name__ == "__main__":
    main()
