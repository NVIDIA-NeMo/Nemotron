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
JSONL Plan Stage - Discovers files, creates shard plans, fans out to JSONL work items.

This stage takes dataset-level JSONL work items (JsonlDatasetWorkItem) and produces
shard-level work items (JsonlShardWorkItem), implementing a fan-out pattern.

Unlike PlanStage, this does not resolve tokenizers - JSONL pipelines don't
need tokenization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.config import DatasetConfig
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json
from nemotron.data_prep.core.planning import (
    create_jsonl_shard_plan,
    get_pending_jsonl_shards,
    serialize_shard_plan,
)
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.core.work_items import JsonlDatasetWorkItem, JsonlShardWorkItem


@dataclass(frozen=True)
class JsonlPlanStageConfig:
    """Configuration for JsonlPlanStage.

    Attributes:
        planner_cpus: CPU request for the planner worker. Default 0.5 since
            this stage is I/O-bound, not CPU-bound.
    """

    planner_cpus: float = 0.5

    def __post_init__(self) -> None:
        if self.planner_cpus <= 0:
            raise ValueError(f"planner_cpus must be positive, got {self.planner_cpus}")


class JsonlPlanStage(pipelines_v1.Stage[JsonlDatasetWorkItem, JsonlShardWorkItem]):
    """
    JSONL planning stage: discovers files, creates shard plans, emits JsonlShardWorkItems.

    This stage runs with a single worker and fans out each JsonlDatasetWorkItem
    into multiple JsonlShardWorkItems (one per pending shard).

    Unlike PlanStage, this does not resolve tokenizers. The plan uses a
    transform_fingerprint to ensure that changes to transforms (e.g. toggling
    placeholder resolution) invalidate cached results.

    Args:
        stage_config: Stage-specific configuration (JsonlPlanStageConfig)
        pipeline_context: Shared runtime context (PipelineContext)
    """

    def __init__(
        self,
        stage_config: JsonlPlanStageConfig,
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

    def process_data(self, items: list[JsonlDatasetWorkItem]) -> list[JsonlShardWorkItem]:
        """Plan datasets and emit JsonlShardWorkItems for pending shards."""
        output: list[JsonlShardWorkItem] = []

        for item in items:
            shard_items = self._plan_dataset(item)
            output.extend(shard_items)

        return output

    def _plan_dataset(self, item: JsonlDatasetWorkItem) -> list[JsonlShardWorkItem]:
        """Plan a single dataset and return JsonlShardWorkItems."""
        # Build dataset config from work item
        dataset_cfg = DatasetConfig(
            name=item.dataset_name,
            path=item.path,
            weight=item.weight,
            split=item.split,
            subset=item.subset,
            text_field=item.text_field,
        )

        # Build transform fingerprint from resolve_hf_placeholders flag
        # This ensures toggling placeholder resolution invalidates the cache
        import hashlib
        import json

        fingerprint_content = json.dumps(
            {"resolve_hf_placeholders": item.resolve_hf_placeholders},
            sort_keys=True,
        )
        transform_fingerprint = hashlib.sha256(fingerprint_content.encode()).hexdigest()[:16]

        # Create shard plan (discovers files, computes assignments, no tokenizer)
        plan = create_jsonl_shard_plan(
            dataset_config=dataset_cfg,
            num_shards=item.num_shards,
            config_hash=item.config_hash,
            fs=self._fs,
            transform_fingerprint=transform_fingerprint,
        )

        # Create dataset directories
        # Mirror pretrain layout: {run_dir}/datasets/{dataset_name}/{plan_hash}/
        dataset_dir = f"{item.run_dir}/datasets/{item.dataset_name}/{plan.plan_hash}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(self._fs, dataset_dir)
        ensure_dir(self._fs, receipts_dir)

        # Write plan.json for reproducibility
        write_json(self._fs, f"{dataset_dir}/plan.json", serialize_shard_plan(plan))

        # Find pending shards (those without completed receipts)
        pending = get_pending_jsonl_shards(plan, receipts_dir, self._fs)

        # Build assignment dicts for work items
        assignment_dicts: dict[int, dict[str, Any]] = {
            a.shard_index: {
                "shard_index": a.shard_index,
                "files": [asdict(f) for f in a.files],
                "total_bytes": a.total_bytes,
            }
            for a in plan.file_assignments
        }

        # Emit JsonlShardWorkItem for each pending shard
        shard_items: list[JsonlShardWorkItem] = []
        for shard_idx in pending:
            shard_items.append(
                JsonlShardWorkItem(
                    dataset_name=item.dataset_name,
                    plan_hash=plan.plan_hash,
                    shard_index=int(shard_idx),
                    assignment=assignment_dicts[int(shard_idx)],
                    output_dir=dataset_dir,
                    receipts_dir=receipts_dir,
                    text_field=item.text_field,
                    compression=item.compression,
                    max_rows=item.max_rows,
                    resolve_hf_placeholders=item.resolve_hf_placeholders,
                )
            )

        return shard_items
