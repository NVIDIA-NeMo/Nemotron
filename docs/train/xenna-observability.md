# Xenna Pipeline Observability

Real-time observability for cosmos-xenna data preparation pipelines, including W&B metrics logging and pipeline statistics tracking.

> **Implementation Note**: This module uses a monkey-patching approach to intercept pipeline statistics, since cosmos-xenna does not currently expose a native stats callback API.

## Overview

When running data preparation pipelines (pretrain, SFT), you can enable real-time logging of pipeline statistics to Weights & Biases. This provides visibility into:

- **Pipeline progress** — Inputs processed, outputs generated, completion percentage
- **Cluster utilization** — CPU/GPU/memory usage across the Ray cluster
- **Per-stage metrics** — Actor counts, queue depths, processing speeds for each pipeline stage
- **Bottleneck detection** — Identify which stages are blocking throughput

## Configuration

Enable W&B logging via the `observability` section in your data prep config:

```yaml
# In your data_prep config (e.g., default.yaml)
observability:
  # Enable real-time W&B logging of pipeline stats
  wandb_log_pipeline_stats: true

  # How often to log (seconds) - matches cosmos-xenna's internal logging rate
  pipeline_logging_interval_s: 30

  # Optional: Also write stats to JSONL file for offline analysis
  pipeline_stats_jsonl_path: /path/to/stats.jsonl
```

## How It Works

### The Monkey-Patch Approach

cosmos-xenna's `PipelineMonitor` class builds a `PipelineStats` object every `logging_interval_s` via the internal `_make_stats()` method. Our hook intercepts this method:

```
┌─────────────────────────────────────────────────────────────────┐
│                     cosmos-xenna pipeline                        │
│                                                                  │
│  PipelineMonitor.update()                                       │
│       │                                                          │
│       ▼                                                          │
│  _make_stats() ◄──── Monkey-patched by WandbStatsHook          │
│       │                                                          │
│       ├──► Original _make_stats() returns PipelineStats         │
│       │                                                          │
│       └──► Hook intercepts stats ──► wandb.log() + JSONL        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key benefits of this approach:**
- **No cosmos-xenna changes required** — Works with current cosmos-xenna main
- **Same update frequency** — Matches cosmos-xenna's internal logging cadence
- **Structured data** — Gets full `PipelineStats` object, not just text output
- **Zero pipeline impact** — Original return value is preserved unchanged

### Thread Safety

The hook uses reference counting for safe nested contexts:
- Multiple hooks can be active simultaneously
- Patch is installed when first hook enters, restored when last hook exits
- Thread-safe with a reentrant lock

## Metrics Logged

### Pipeline-Level Metrics

| Metric | Description |
|--------|-------------|
| `{kind}/pipeline_duration_s` | Total elapsed time since pipeline start |
| `{kind}/main_loop_rate_hz` | Pipeline main loop frequency |
| `{kind}/progress` | Percentage of inputs processed (0-100) |
| `{kind}/num_input_remaining` | Inputs still waiting to be processed |
| `{kind}/num_outputs` | Total outputs generated |
| `{kind}/inputs_processed_per_s` | Input processing rate |
| `{kind}/outputs_per_s` | Output generation rate |

### Cluster Resource Metrics

| Metric | Description |
|--------|-------------|
| `{kind}/cluster/total_cpus` | Total CPUs in Ray cluster |
| `{kind}/cluster/avail_cpus` | Available (unused) CPUs |
| `{kind}/cluster/total_gpus` | Total GPUs in cluster |
| `{kind}/cluster/avail_gpus` | Available GPUs |
| `{kind}/cluster/total_mem_gb` | Total cluster memory (GB) |
| `{kind}/cluster/avail_mem_gb` | Available memory (GB) |

### Per-Stage Metrics (Consolidated Charts)

Stage metrics are logged as consolidated `line_series` charts (one chart per metric, one line per stage):

| Metric | Description |
|--------|-------------|
| `stages/actors_target` | Target number of actors per stage |
| `stages/actors_ready` | Actors ready to process per stage |
| `stages/actors_running` | Actors currently processing per stage |
| `stages/tasks_completed` | Total completed tasks per stage |
| `stages/queue_in` | Input queue depth per stage |
| `stages/queue_out` | Output queue depth per stage |
| `stages/speed_tasks_per_s` | Processing speed per stage |
| `stages/resource_cpu_util_pct` | CPU utilization per stage |
| `stages/resource_mem_gb` | Memory usage (GB) per stage |

## Usage in Recipes

The pretrain and SFT recipes automatically use the W&B hook when `wandb_log_pipeline_stats: true`:

```python
from nemotron.data_prep.observability import make_wandb_stats_hook

# Create hook if enabled
wandb_hook = make_wandb_stats_hook(
    observability=observability_cfg,
    pipeline_kind="pretrain",  # or "sft"
    run_hash=context.run_hash,
    run_dir=context.run_dir,
    dataset_names=context.dataset_names,
)

# Run pipeline with hook
if wandb_hook:
    with wandb_hook:
        pipelines_v1.run_pipeline(pipeline_spec)
else:
    pipelines_v1.run_pipeline(pipeline_spec)
```

## JSONL Output

For offline analysis or when W&B isn't available, enable JSONL output:

```yaml
observability:
  wandb_log_pipeline_stats: false
  pipeline_stats_jsonl_path: /output/pipeline_stats.jsonl
```

Each line contains a JSON record:

```json
{
  "timestamp": 1706123456.789,
  "pipeline_kind": "pretrain",
  "run_hash": "abc123",
  "metrics": {
    "pipeline_duration_s": 120.5,
    "progress": 50.0,
    "cluster/total_cpus": 64.0,
    "stages/tasks_completed/download": 100
  },
  "stages": ["PlanStage", "DownloadStage", "BinIdxTokenizationStage"]
}
```

## API Reference

### wandb_hook.py

| Export | Description |
|--------|-------------|
| `WandbStatsHook` | Context manager that patches `PipelineMonitor._make_stats` |
| `make_wandb_stats_hook()` | Factory function for recipes |
| `log_plan_table_to_wandb()` | Log plan table showing datasets and processing config |

### ObservabilityConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wandb_log_pipeline_stats` | `bool` | `True` | Enable W&B logging |
| `wandb_log_plan_table` | `bool` | `True` | Log plan table to W&B |
| `wandb_log_progress_table` | `bool` | `True` | Log per-dataset progress table |
| `wandb_log_stage_table` | `bool` | `True` | Log stage overview table |
| `pipeline_logging_interval_s` | `int` | `30` | Logging interval in seconds |
| `pipeline_stats_jsonl_path` | `str \| None` | `None` | Path for JSONL output |

## Troubleshooting

### Metrics not appearing in W&B

1. Verify W&B is initialized before the pipeline runs:
   ```python
   import wandb
   assert wandb.run is not None, "W&B not initialized"
   ```

2. Check that `wandb_log_pipeline_stats: true` in your config

3. Ensure the hook is active during pipeline execution (check for log message: "Installed PipelineMonitor._make_stats patch")

### Import errors for cosmos_xenna

The hook lazy-imports `cosmos_xenna` only when entering the context. If you see import errors:

1. Ensure cosmos-xenna is installed: `uv pip install cosmos-xenna`
2. For Ray workers, use `--extra xenna` in the run command (handled automatically by recipes)

### Missing stage metrics

Some stages may not report all metrics if:
- The stage hasn't processed any tasks yet
- The stage has `processing_speed_tasks_per_second = None` (no speed data available)

These are expected behaviors and the hook gracefully handles missing data.

## Further Reading

- [Weights & Biases Integration](./wandb.md) — W&B configuration and authentication
- [Data Preparation](./data-prep.md) — Data prep module overview
- [Artifact Lineage](./artifacts.md) — Tracking data lineage in W&B
