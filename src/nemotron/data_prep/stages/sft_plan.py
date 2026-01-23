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

"""
SFT Plan Stage - Discovers files, creates shard plans, fans out to SFT shard work items.

This stage mirrors PlanStage but emits SftShardWorkItem and carries SFT-specific
parameters forward to the packed Parquet processing stage.

In STREAMING mode, downstream stages start processing shards immediately
as this stage emits them - no barrier needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import DatasetConfig, InternalOutputConfig, InternalTokenizerConfig
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.core.planning import (
    apply_shard_sampling,
    create_shard_plan,
    get_pending_shards,
    serialize_shard_plan,
)
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.core.work_items import SftShardWorkItem

if TYPE_CHECKING:
    from nemotron.data_prep.core.work_items import SftDatasetWorkItem


@dataclass(frozen=True)
class SftPlanStageConfig:
    """Configuration for SftPlanStage.

    SftPlanStage mirrors PlanStage but for SFT pipelines.

    Attributes:
        planner_cpus: CPU request for the planner worker. Default 0.5 since
            this stage is I/O-bound, not CPU-bound.
    """

    planner_cpus: float = 0.5

    def __post_init__(self) -> None:
        if self.planner_cpus <= 0:
            raise ValueError(f"planner_cpus must be positive, got {self.planner_cpus}")


class SftPlanStage(pipelines_v1.Stage["SftDatasetWorkItem", SftShardWorkItem]):
    """Planning stage for xenna-native SFT Parquet pipeline.

    Args:
        stage_config: Stage-specific configuration (SftPlanStageConfig)
        pipeline_context: Shared runtime context (PipelineContext)
    """

    def __init__(
        self,
        stage_config: SftPlanStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._fs = None

    @property
    def stage_batch_size(self) -> int:
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        return pipelines_v1.Resources(cpus=self._cfg.planner_cpus, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        self._fs, _ = get_filesystem(self._ctx.output_root)

    def process_data(self, items: list["SftDatasetWorkItem"]) -> list[SftShardWorkItem]:
        out: list[SftShardWorkItem] = []
        for item in items:
            out.extend(self._plan_dataset(item))
        return out

    def _plan_dataset(self, item: "SftDatasetWorkItem") -> list[SftShardWorkItem]:
        dataset_cfg = DatasetConfig(
            name=item.dataset_name,
            path=item.path,
            weight=item.weight,
            split=item.split,
            subset=item.subset,
            text_field=item.messages_field,  # used for discovery/metadata only
        )

        tokenizer_cfg = InternalTokenizerConfig(**item.tokenizer_config)

        output_cfg = InternalOutputConfig(
            num_shards=item.num_shards,
            dtype=item.dtype,
            min_doc_chars=None,
            max_doc_tokens=item.max_doc_tokens,
            max_rows=item.max_rows,
        )

        plan = create_shard_plan(
            dataset_config=dataset_cfg,
            output_config=output_cfg,
            tokenizer_config=tokenizer_cfg,
            config_hash=item.config_hash,
            fs=self._fs,
        )

        dataset_dir = f"{item.run_dir}/datasets/{item.dataset_name}/{plan.plan_hash}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(self._fs, dataset_dir)
        ensure_dir(self._fs, receipts_dir)

        write_json(self._fs, f"{dataset_dir}/plan.json", serialize_shard_plan(plan))

        pending = get_pending_shards(plan, receipts_dir, self._fs)
        if item.sample is not None:
            pending = apply_shard_sampling(pending, plan, item.sample, item.sample_seed)

        assignment_dicts: dict[int, dict[str, Any]] = {
            a.shard_index: {
                "shard_index": a.shard_index,
                "files": [asdict(f) for f in a.files],
                "total_bytes": a.total_bytes,
            }
            for a in plan.file_assignments
        }

        shard_items: list[SftShardWorkItem] = []
        for shard_idx in pending:
            shard_id = f"shard_{int(shard_idx):06d}"
            spool_dir = f"{dataset_dir.rstrip('/')}/spool/{shard_id}"

            shard_items.append(
                SftShardWorkItem(
                    dataset_name=item.dataset_name,
                    plan_hash=plan.plan_hash,
                    shard_index=int(shard_idx),
                    assignment=assignment_dicts[int(shard_idx)],
                    output_dir=dataset_dir,
                    receipts_dir=receipts_dir,
                    spool_dir=spool_dir,
                    dtype=item.dtype,
                    messages_field=item.messages_field,
                    tools_field=item.tools_field,
                    chat_template=item.chat_template,
                    max_doc_tokens=item.max_doc_tokens,
                    max_rows=item.max_rows,
                    used_in_filter=item.used_in_filter,
                    used_in_field=item.used_in_field,
                    pack_size=item.pack_size,
                    algorithm=item.algorithm,
                    seed=item.seed,
                    parquet_row_group_size=item.parquet_row_group_size,
                    parquet_compression=item.parquet_compression,
                )
            )

        return shard_items


__all__ = ["SftPlanStage"]