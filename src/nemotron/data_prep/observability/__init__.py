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

"""Observability utilities for data preparation pipelines.

This package provides consolidated access to observability features:
- W&B integration for real-time pipeline metrics
- Stage naming conventions

W&B Integration:
    The WandbStatsHook is the primary observability mechanism used by recipes.
    It patches PipelineMonitor to intercept stats and log to W&B in real-time.

    >>> from nemotron.data_prep.observability import make_wandb_stats_hook
    >>> hook = make_wandb_stats_hook(observability=cfg, pipeline_kind="pretrain")
    >>> with hook:
    ...     pipelines_v1.run_pipeline(spec)
"""

# W&B integration (primary observability mechanism)
from nemotron.data_prep.observability.wandb_hook import (
    WandbStatsHook,
    log_plan_table_to_wandb,
    make_wandb_stats_hook,
)

# Stage naming utilities
from nemotron.data_prep.observability.stage_keys import (
    canonical_stage_id,
    get_stage_display_name,
)

__all__ = [
    # W&B integration
    "WandbStatsHook",
    "make_wandb_stats_hook",
    "log_plan_table_to_wandb",
    # Stage naming
    "canonical_stage_id",
    "get_stage_display_name",
]
