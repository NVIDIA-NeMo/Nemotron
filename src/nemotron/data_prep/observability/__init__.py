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
- Prometheus metrics scraping
- Pipeline stats callbacks
- Stage naming conventions

All implementations now live in this package for cleaner organization.

W&B Integration (Primary):
    The WandbStatsHook is the primary observability mechanism used by recipes.
    It patches PipelineMonitor to intercept stats and log to W&B in real-time.

    >>> from nemotron.data_prep.observability import make_wandb_stats_hook
    >>> hook = make_wandb_stats_hook(observability=cfg, pipeline_kind="pretrain")
    >>> with hook:
    ...     pipelines_v1.run_pipeline(spec)

Prometheus Metrics (Alternative):
    For environments where W&B is not available, Prometheus metrics can be
    scraped from Ray's metrics endpoint.

    >>> from nemotron.data_prep.observability import (
    ...     PrometheusMetricsLogger,
    ...     enable_ray_metrics_export,
    ... )
    >>> enable_ray_metrics_export(port=8080)
    >>> with PrometheusMetricsLogger(port=8080):
    ...     pipelines_v1.run_pipeline(spec)

Stats Callback (Legacy):
    For custom stats handling, use the callback factory:

    >>> from nemotron.data_prep.observability import make_pipeline_stats_callback
    >>> callback = make_pipeline_stats_callback(observability=cfg, pipeline_kind="pretrain")
"""

# W&B integration (primary observability mechanism)
from nemotron.data_prep.observability.wandb_hook import (
    WandbStatsHook,
    log_plan_table_to_wandb,
    make_wandb_stats_hook,
)

# Prometheus metrics (alternative)
from nemotron.data_prep.observability.prometheus_metrics import (
    PrometheusConfig,
    PrometheusMetricsLogger,
    enable_ray_metrics_export,
)

# Stats callback (for custom handling)
from nemotron.data_prep.observability.stats_callback import make_pipeline_stats_callback

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
    # Prometheus metrics
    "PrometheusConfig",
    "PrometheusMetricsLogger",
    "enable_ray_metrics_export",
    # Stats callback
    "make_pipeline_stats_callback",
    # Stage naming
    "canonical_stage_id",
    "get_stage_display_name",
]
