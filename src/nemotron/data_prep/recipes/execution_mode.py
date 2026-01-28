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

"""Auto execution mode resolution for data prep pipelines.

Resolves execution_mode="auto" to STREAMING or BATCH based on available
cluster resources. This avoids the cosmos-xenna ValueError when running
on small nodes while preserving STREAMING on properly-sized clusters.

Usage:
    decision = decide_execution_mode_for_stages(
        requested="auto",
        stage_specs=stage_specs,
        pipeline_name="pretrain",
        logger=logger,
    )
    # Use decision.resolved (a pipelines_v1.ExecutionMode) in PipelineConfig
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import cosmos_xenna.pipelines.v1 as pipelines_v1

ExecutionModeRequest: TypeAlias = pipelines_v1.ExecutionMode | Literal["auto", "streaming", "batch"]


@dataclass(frozen=True)
class ExecutionModeDecision:
    """Result of resolving an ExecutionModeRequest to a concrete mode."""

    requested: ExecutionModeRequest
    resolved: pipelines_v1.ExecutionMode
    required_cpus: float
    total_cpus: float | None
    num_nodes: int | None
    reason: str
    stage_breakdown: list[dict[str, Any]]


def _normalize_requested(
    requested: ExecutionModeRequest,
) -> pipelines_v1.ExecutionMode | Literal["auto"]:
    """Normalize string/enum input to a concrete mode or 'auto'."""
    if isinstance(requested, pipelines_v1.ExecutionMode):
        return requested
    if isinstance(requested, str):
        v = requested.strip().lower()
        if v == "auto":
            return "auto"
        if v == "streaming":
            return pipelines_v1.ExecutionMode.STREAMING
        if v == "batch":
            return pipelines_v1.ExecutionMode.BATCH
        raise ValueError(f"Invalid execution_mode={requested!r}. Allowed: 'auto', 'streaming', 'batch'.")
    raise TypeError(f"Invalid execution_mode type: {type(requested)!r}")


def _try_get_ray_cluster_summary(
    logger: logging.Logger,
) -> tuple[float, int] | None:
    """Query Ray for total CPUs and alive node count. Returns None on failure."""
    try:
        import ray
    except ImportError:
        logger.warning("execution_mode='auto': ray not importable; defaulting to STREAMING")
        return None

    try:
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True, logging_level=logging.ERROR)

        total_cpus = float((ray.cluster_resources() or {}).get("CPU", 0.0) or 0.0)
        nodes = ray.nodes() or []
        alive_nodes = sum(1 for n in nodes if n.get("Alive"))
        num_nodes = max(1, int(alive_nodes))
        return total_cpus, num_nodes
    except Exception as e:
        logger.warning(
            f"execution_mode='auto': could not query Ray cluster ({type(e).__name__}: {e}); "
            f"defaulting to STREAMING"
        )
        return None


def _estimate_required_cpus(
    *,
    stage_specs: list[pipelines_v1.StageSpec],
    num_nodes: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Estimate total CPUs required in STREAMING mode (all stages concurrent).

    Mirrors the logic in cosmos-xenna's _verify_enough_resources().
    """
    required = 0.0
    breakdown: list[dict[str, Any]] = []

    for spec in stage_specs:
        if spec.num_workers is not None:
            num_required = int(spec.num_workers)
            scale_reason = "num_workers"
        elif spec.num_workers_per_node is not None:
            num_required = int(math.ceil(float(spec.num_workers_per_node) * float(num_nodes)))
            scale_reason = "num_workers_per_node"
        else:
            num_required = 1
            scale_reason = "default_1"

        resources = spec.stage.required_resources
        cpus_per_worker = float(getattr(resources, "cpus", 0.0) or 0.0)
        stage_total = cpus_per_worker * float(num_required)
        required += stage_total

        breakdown.append({
            "stage": spec.stage.__class__.__name__,
            "cpus_per_worker": cpus_per_worker,
            "num_required": num_required,
            "scale_reason": scale_reason,
            "stage_total_cpus": stage_total,
        })

    return required, breakdown


def decide_execution_mode_for_stages(
    *,
    requested: ExecutionModeRequest,
    stage_specs: list[pipelines_v1.StageSpec],
    pipeline_name: str,
    logger: logging.Logger,
) -> ExecutionModeDecision:
    """Resolve an execution mode request to a concrete ExecutionMode.

    When requested="auto", queries Ray for cluster resources and chooses
    STREAMING if sufficient CPUs are available, BATCH otherwise.

    Args:
        requested: "auto", "streaming", "batch", or a pipelines_v1.ExecutionMode.
        stage_specs: The StageSpec list that will be passed to PipelineSpec.
        pipeline_name: Name for logging (e.g. "pretrain", "sft", "rl").
        logger: Logger instance.

    Returns:
        ExecutionModeDecision with the resolved mode and diagnostic info.
    """
    normalized = _normalize_requested(requested)

    # Query Ray cluster info
    summary = _try_get_ray_cluster_summary(logger)

    if summary is None:
        # Can't query Ray — best-effort: assume 1 node for breakdown, default to STREAMING
        required_cpus, breakdown = _estimate_required_cpus(stage_specs=stage_specs, num_nodes=1)
        resolved = pipelines_v1.ExecutionMode.STREAMING if normalized == "auto" else normalized
        reason = "ray_unavailable_default_streaming" if normalized == "auto" else "explicit"
        logger.info(
            f"[{pipeline_name}] execution_mode={requested!r} -> {resolved.name} "
            f"(reason={reason}, required_cpus~{required_cpus:.1f}, ray_cluster=unknown)"
        )
        return ExecutionModeDecision(
            requested=requested,
            resolved=resolved,
            required_cpus=required_cpus,
            total_cpus=None,
            num_nodes=None,
            reason=reason,
            stage_breakdown=breakdown,
        )

    total_cpus, num_nodes = summary
    required_cpus, breakdown = _estimate_required_cpus(stage_specs=stage_specs, num_nodes=num_nodes)

    # Explicit mode — just log and pass through
    if normalized != "auto":
        logger.info(
            f"[{pipeline_name}] execution_mode={requested!r} -> {normalized.name} "
            f"(explicit, required_cpus~{required_cpus:.1f}, total_cpus={total_cpus:.1f}, "
            f"num_nodes={num_nodes})"
        )
        return ExecutionModeDecision(
            requested=requested,
            resolved=normalized,
            required_cpus=required_cpus,
            total_cpus=total_cpus,
            num_nodes=num_nodes,
            reason="explicit",
            stage_breakdown=breakdown,
        )

    # Auto mode — choose based on resources
    if total_cpus + 1e-6 >= required_cpus:
        resolved = pipelines_v1.ExecutionMode.STREAMING
        reason = "cpu_sufficient"
        logger.info(
            f"[{pipeline_name}] execution_mode='auto' -> STREAMING "
            f"(required_cpus~{required_cpus:.1f}, total_cpus={total_cpus:.1f}, "
            f"num_nodes={num_nodes})"
        )
    else:
        resolved = pipelines_v1.ExecutionMode.BATCH
        reason = "cpu_insufficient_fallback_batch"
        logger.warning(
            f"[{pipeline_name}] execution_mode='auto' -> BATCH "
            f"(required_cpus~{required_cpus:.1f} > total_cpus={total_cpus:.1f}, "
            f"num_nodes={num_nodes})"
        )

    return ExecutionModeDecision(
        requested=requested,
        resolved=resolved,
        required_cpus=required_cpus,
        total_cpus=total_cpus,
        num_nodes=num_nodes,
        reason=reason,
        stage_breakdown=breakdown,
    )


__all__ = [
    "ExecutionModeRequest",
    "ExecutionModeDecision",
    "decide_execution_mode_for_stages",
]
