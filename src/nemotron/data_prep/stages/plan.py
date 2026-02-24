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

"""Unified planning stage with pipeline-specific adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import cosmos_xenna.pipelines.v1 as pipelines_v1
from fsspec import AbstractFileSystem

from nemotron.data_prep.core.planning import (
    PlanRequest,
    apply_shard_sampling,
    create_plan,
    get_pending_shards,
    serialize_shard_plan,
)
from nemotron.data_prep.stages.context import PipelineContext
from nemotron.data_prep.utils.filesystem import ensure_dir, get_filesystem, write_json


class PlanAdapter(Protocol):
    """Pipeline-specific adapter for PlanStage.

    Each pipeline (pretrain, SFT, RL) provides an adapter that tells PlanStage
    how to create plan requests from dataset items, build shard work items from
    plan assignments, and verify output files for idempotent resume.

    See recipe implementations for concrete examples:
        - PretrainPlanAdapter in recipes/pretrain.py
        - SftPlanAdapter in recipes/sft.py
        - JsonlPlanAdapter in recipes/rl.py
    """

    def to_plan_request(self, item: Any) -> PlanRequest:
        """Convert a dataset work item into a PlanRequest for file discovery."""
        ...

    def to_shard_item(
        self,
        item: Any,
        *,
        plan_hash: str,
        shard_index: int,
        assignment: dict[str, Any],
        output_dir: str,
        receipts_dir: str,
    ) -> Any:
        """Create a pipeline-specific shard work item from plan assignment."""
        ...

    def get_output_verifier(
        self, fs: AbstractFileSystem
    ) -> Callable[[dict, str, AbstractFileSystem], bool] | None:
        """Return a function to verify outputs exist on resume, or None."""
        ...


@dataclass(frozen=True)
class PlanStageConfig:
    """Configuration for PlanStage."""

    planner_cpus: float = 0.5

    def __post_init__(self) -> None:
        if self.planner_cpus <= 0:
            raise ValueError(f"planner_cpus must be positive, got {self.planner_cpus}")


class PlanStage(pipelines_v1.Stage[Any, Any]):
    """Planning stage using an adapter to emit pipeline-specific shard work items."""

    def __init__(
        self,
        stage_config: PlanStageConfig,
        pipeline_context: PipelineContext,
        adapter: PlanAdapter,
    ) -> None:
        self._cfg = stage_config
        self._ctx = pipeline_context
        self._adapter = adapter
        self._fs: AbstractFileSystem | None = None

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

    def process_data(self, items: list[Any]) -> list[Any]:
        output: list[Any] = []
        for item in items:
            output.extend(self._plan_dataset(item))
        return output

    def _plan_dataset(self, item: Any) -> list[Any]:
        request = self._adapter.to_plan_request(item)
        plan = create_plan(request, self._fs)

        dataset_dir = f"{item.run_dir}/datasets/{item.dataset_name}/{plan.plan_hash}"
        receipts_dir = f"{dataset_dir}/receipts"
        ensure_dir(self._fs, dataset_dir)
        ensure_dir(self._fs, receipts_dir)
        write_json(self._fs, f"{dataset_dir}/plan.json", serialize_shard_plan(plan))

        verifier = self._adapter.get_output_verifier(self._fs)
        pending = get_pending_shards(plan, receipts_dir, self._fs, verify_output=verifier)
        sample_spec = getattr(item, "sample", None)
        if sample_spec is not None:
            pending = apply_shard_sampling(
                pending,
                plan,
                sample_spec,
                int(getattr(item, "sample_seed", 42)),
            )

        assignment_dicts = {
            assignment.shard_index: {
                "shard_index": assignment.shard_index,
                "files": [asdict(file_info) for file_info in assignment.files],
                "total_bytes": assignment.total_bytes,
            }
            for assignment in plan.file_assignments
        }

        return [
            self._adapter.to_shard_item(
                item,
                plan_hash=plan.plan_hash,
                shard_index=int(shard_index),
                assignment=assignment_dicts[int(shard_index)],
                output_dir=dataset_dir,
                receipts_dir=receipts_dir,
            )
            for shard_index in pending
        ]


__all__ = ["PlanAdapter", "PlanStage", "PlanStageConfig"]
