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
Plan Stage - Discovers files, creates shard plans, fans out to work items.

This stage takes dataset-level work items (DatasetWorkItem) and produces
shard-level work items (ShardWorkItem), implementing a fan-out pattern.

In STREAMING mode, downstream stages start processing shards immediately
as this stage emits them - no barrier needed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
from nemotron.data_prep.core.work_items import DatasetWorkItem, ShardWorkItem


@dataclass(frozen=True)
class PlanStageConfig:
    """Configuration for PlanStage.

    PlanStage is I/O-bound (file discovery, plan creation) and requires
    minimal resources.

    Attributes:
        planner_cpus: CPU request for the planner worker. Default 0.5 since
            this stage is I/O-bound, not CPU-bound.
    """

    planner_cpus: float = 0.5

    def __post_init__(self) -> None:
        if self.planner_cpus <= 0:
            raise ValueError(f"planner_cpus must be positive, got {self.planner_cpus}")


class PlanStage(pipelines_v1.Stage[DatasetWorkItem, ShardWorkItem]):
    """
    Planning stage: discovers files, creates shard plans, emits ShardWorkItems.

    This stage runs with a single worker and fans out each DatasetWorkItem
    into multiple ShardWorkItems (one per pending shard).

    Responsibilities:
    - Discover input files for each dataset
    - Create shard plan (file → shard assignments)
    - Write plan.json for reproducibility
    - Check existing receipts for idempotency
    - Apply sampling if configured
    - Emit ShardWorkItem for each pending shard

    In STREAMING mode, downstream stages start processing shards immediately
    as this stage emits them.

    Args:
        stage_config: Stage-specific configuration (PlanStageConfig)
        pipeline_context: Shared runtime context (PipelineContext)
    """

    def __init__(
        self,
        stage_config: PlanStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._fs = None

    @property
    def stage_batch_size(self) -> int:
        """Process one dataset at a time."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        """Minimal CPU, I/O bound stage."""
        return pipelines_v1.Resources(cpus=self._cfg.planner_cpus, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        """Runtime environment with HF credentials for file discovery."""
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Initialize filesystem on worker."""
        self._fs, _ = get_filesystem(self._ctx.output_root)

    def process_data(self, items: list[DatasetWorkItem]) -> list[ShardWorkItem]:
        """Plan datasets and emit ShardWorkItems for pending shards."""
        output: list[ShardWorkItem] = []

        for item in items:
            shard_items = self._plan_dataset(item)
            output.extend(shard_items)

        return output

    def _plan_dataset(self, item: DatasetWorkItem) -> list[ShardWorkItem]:
        """Plan a single dataset and return ShardWorkItems."""
        # Build internal configs from work item
        dataset_cfg = DatasetConfig(
            name=item.dataset_name,
            path=item.path,
            weight=item.weight,
            split=item.split,
            subset=item.subset,
            text_field=item.text_field,
        )

        tokenizer_cfg = InternalTokenizerConfig(**item.tokenizer_config)

        output_cfg = InternalOutputConfig(
            num_shards=item.num_shards,
            dtype=item.dtype,
            min_doc_chars=item.min_doc_chars,
            max_doc_tokens=item.max_doc_tokens,
            max_rows=item.max_rows,
        )

        # Create shard plan (discovers files, computes assignments)
        plan = create_shard_plan(
            dataset_config=dataset_cfg,
            output_config=output_cfg,
            tokenizer_config=tokenizer_cfg,
            config_hash=item.config_hash,
            fs=self._fs,
        )

        # Create dataset directories
        dataset_dir = f"{item.run_dir}/datasets/{item.dataset_name}/{plan.plan_hash}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(self._fs, dataset_dir)
        ensure_dir(self._fs, receipts_dir)

        # Write plan.json for reproducibility
        write_json(self._fs, f"{dataset_dir}/plan.json", serialize_shard_plan(plan))

        # Find pending shards (those without completed receipts)
        pending = get_pending_shards(plan, receipts_dir, self._fs)
        if item.sample is not None:
            pending = apply_shard_sampling(pending, plan, item.sample, item.sample_seed)

        # Build assignment dicts for work items
        assignment_dicts: dict[int, dict[str, Any]] = {
            a.shard_index: {
                "shard_index": a.shard_index,
                "files": [asdict(f) for f in a.files],
                "total_bytes": a.total_bytes,
            }
            for a in plan.file_assignments
        }

        # Emit ShardWorkItem for each pending shard
        shard_items: list[ShardWorkItem] = []
        for shard_idx in pending:
            shard_items.append(
                ShardWorkItem(
                    dataset_name=item.dataset_name,
                    plan_hash=plan.plan_hash,
                    shard_index=int(shard_idx),
                    assignment=assignment_dicts[int(shard_idx)],
                    output_dir=dataset_dir,
                    receipts_dir=receipts_dir,
                    text_field=item.text_field,
                    dtype=item.dtype,
                    min_doc_chars=item.min_doc_chars,
                    max_doc_tokens=item.max_doc_tokens,
                    max_rows=item.max_rows,
                )
            )

        return shard_items
