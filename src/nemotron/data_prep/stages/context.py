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

"""Pipeline context for stage construction.

PipelineContext provides shared runtime configuration that all stages need.
Each stage receives this context alongside its stage-specific config, enabling
a clean (stage_config, pipeline_context) constructor pattern.

This module is intentionally minimal to keep the context picklable for Ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cosmos_xenna.ray_utils.runtime_envs import RuntimeEnv

from nemotron.data_prep.config import ObservabilityConfig
from nemotron.data_prep.utils.hf_env import make_hf_runtime_env


@dataclass(frozen=True)
class PipelineContext:
    """Shared runtime context passed to all pipeline stages.

    This dataclass contains configuration that is shared across multiple stages
    in a pipeline. By consolidating shared config here, individual stages can
    have simpler constructors focused only on stage-specific settings.

    Attributes:
        output_root: Base output directory (local path or cloud URI)
        run_hash: Deterministic hash identifying this run configuration.
            Used for receipt metadata and idempotency. May be None for
            stages that don't write receipts.
        run_dir: Path to the runs/{run_hash} directory. May be None for
            stages that don't need run-specific paths.
        config_hash: Hash of the pipeline configuration. Used for shard
            planning and validation.
        resolved_tokenizer: Resolved tokenizer configuration dict (from
            resolve_tokenizer). Required for tokenization stages, may be
            None for download/planning-only stages.
        observability: Observability settings (logging intervals, wandb)
        hf_env: Pre-detected HuggingFace environment variables (HF_HOME, HF_TOKEN).
            Stages use this to propagate credentials to Ray workers.

    Example:
        >>> ctx = PipelineContext(
        ...     output_root="/data/output",
        ...     run_hash="abc123",
        ...     run_dir="/data/output/runs/abc123",
        ...     config_hash="def456",
        ...     resolved_tokenizer={"model": "gpt2", "type": "huggingface"},
        ...     observability=ObservabilityConfig(),
        ...     hf_env={"HF_HOME": "/cache/hf"},
        ... )
        >>> stage = MyStage(MyStageConfig(...), ctx)

    Note:
        This dataclass must remain picklable for Ray serialization. Only use
        primitive types, frozen dataclasses, and plain dicts. Do NOT store
        filesystem handles, tokenizers, loggers, or module objects.
    """

    output_root: str
    run_hash: str | None = None
    run_dir: str | None = None
    config_hash: str | None = None
    resolved_tokenizer: dict[str, Any] | None = None
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    hf_env: dict[str, str] = field(default_factory=dict)

    def hf_runtime_env(self, *, extra_env_vars: dict[str, str] | None = None) -> RuntimeEnv:
        """Create RuntimeEnv with HF credentials and optional extra env vars.

        This is a convenience method that stages can use in their env_info
        property to construct a RuntimeEnv with the pre-detected HF environment
        variables merged with any stage-specific overrides.

        Args:
            extra_env_vars: Additional environment variables to include. These
                take precedence over the base hf_env values.

        Returns:
            RuntimeEnv configured with HuggingFace environment variables.

        Example:
            >>> @property
            ... def env_info(self) -> RuntimeEnv:
            ...     # For hf_xet tuning, add extra env vars
            ...     return self._ctx.hf_runtime_env(extra_env_vars={
            ...         "HF_XET_HIGH_PERFORMANCE": "1",
            ...     })
        """
        return make_hf_runtime_env(base_env_vars=self.hf_env, extra_env_vars=extra_env_vars)


__all__ = ["PipelineContext"]
