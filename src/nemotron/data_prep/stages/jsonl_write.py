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
JSONL Shard Stage - Reads input files, applies transforms, writes JSONL + receipts.

This stage processes JsonlShardWorkItems by reading assigned input files,
optionally applying an HF placeholder resolver transform, and writing
JSONL output with receipt-based idempotency.

The resolver is instantiated once per worker and reused across tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cosmos_xenna.pipelines.v1 as pipelines_v1

from nemotron.data_prep.core.jsonl_shard_core import process_jsonl_shard_core
from nemotron.data_prep.formats.transforms import resolve_hf_placeholders
from nemotron.data_prep.utils.filesystem import get_filesystem
from nemotron.data_prep.utils.hf_placeholder import HFPlaceholderResolver
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.core.work_items import JsonlShardWorkItem


@dataclass(frozen=True)
class JsonlShardStageConfig:
    """Configuration for JsonlShardStage.

    Attributes:
        stage_cpus: CPU request per worker. Default 1.0 since this stage
            reads and transforms records.
        local_files_only: If True, only use locally cached HF files (faster
            but requires DownloadStage to have run first). Default True.
    """

    stage_cpus: float = 1.0
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if self.stage_cpus <= 0:
            raise ValueError(f"stage_cpus must be positive, got {self.stage_cpus}")


class JsonlShardStage(pipelines_v1.Stage[JsonlShardWorkItem, JsonlShardWorkItem]):
    """
    JSONL shard processing stage: reads files, transforms, writes JSONL.

    For each JsonlShardWorkItem:
    - If resolve_hf_placeholders is True, applies the HF placeholder resolver
    - Calls process_jsonl_shard_core() for atomic JSONL writing with receipts
    - Returns the work items for finalize to aggregate

    The HFPlaceholderResolver is instantiated once per worker in setup() and
    reused across all tasks to avoid repeated dataset loading.

    Args:
        stage_config: Stage-specific configuration (JsonlShardStageConfig)
        pipeline_context: Shared runtime context (PipelineContext)
    """

    def __init__(
        self,
        stage_config: JsonlShardStageConfig,
        pipeline_context: PipelineContext,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._output_fs = None
        self._resolver: HFPlaceholderResolver | None = None

    @property
    def stage_batch_size(self) -> int:
        """Process one shard at a time to limit memory."""
        return 1

    @property
    def required_resources(self) -> pipelines_v1.Resources:
        """CPU-bound stage for reading + transforming records."""
        return pipelines_v1.Resources(cpus=self._cfg.stage_cpus, gpus=0)

    @property
    def env_info(self) -> pipelines_v1.RuntimeEnv:
        """Runtime environment with HF credentials."""
        return self._ctx.hf_runtime_env()

    def setup(self, worker_metadata: pipelines_v1.WorkerMetadata) -> None:
        """Initialize filesystem and resolver on worker."""
        self._output_fs, _ = get_filesystem(self._ctx.output_root)
        # Resolver will be lazily created on first use if needed
        self._resolver = None

    def _get_resolver(self) -> HFPlaceholderResolver:
        """Get or create the HF placeholder resolver (lazy init per worker)."""
        if self._resolver is None:
            self._resolver = HFPlaceholderResolver.create()
        return self._resolver

    def process_data(self, tasks: list[JsonlShardWorkItem]) -> list[JsonlShardWorkItem]:
        """Process JSONL shards and return work items for finalize."""
        for t in tasks:
            # Build transform based on work item config
            transform = None
            if t.resolve_hf_placeholders:
                transform = resolve_hf_placeholders(resolver=self._get_resolver())

            process_jsonl_shard_core(
                shard_index=t.shard_index,
                files=t.assignment["files"],
                output_dir=t.output_dir,
                receipts_dir=t.receipts_dir,
                output_fs=self._output_fs,
                text_field=t.text_field,
                transform=transform,
                compression=t.compression,
                max_rows=t.max_rows,
                local_files_only=self._cfg.local_files_only,
                plan_hash=t.plan_hash,
                dataset_name=t.dataset_name,
            )

        return tasks
