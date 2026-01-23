# Xenna Pipeline Observability

Real-time observability for cosmos-xenna data preparation pipelines, including W&B metrics logging and pipeline statistics tracking.

> **Temporary Implementation**: This module uses a monkey-patching approach to intercept pipeline statistics. This is a temporary solution until [cosmos-xenna](https://github.com/NVIDIA/cosmos-xenna) adds native support for stats callbacks. Once the upstream PR is merged, this implementation will be replaced with the native `stats_callback` API.

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
│  _make_stats() ◄──── Monkey-patched by XennaWandbStatsHook     │
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
| `xenna/pipeline_duration_s` | Total elapsed time since pipeline start |
| `xenna/main_loop_rate_hz` | Pipeline main loop frequency |
| `xenna/progress` | Percentage of inputs processed (0-100) |
| `xenna/num_input_remaining` | Inputs still waiting to be processed |
| `xenna/num_outputs` | Total outputs generated |
| `xenna/inputs_processed_per_s` | Input processing rate |
| `xenna/outputs_per_s` | Output generation rate |

### Cluster Resource Metrics

| Metric | Description |
|--------|-------------|
| `xenna/cluster/total_cpus` | Total CPUs in Ray cluster |
| `xenna/cluster/avail_cpus` | Available (unused) CPUs |
| `xenna/cluster/total_gpus` | Total GPUs in cluster |
| `xenna/cluster/avail_gpus` | Available GPUs |
| `xenna/cluster/total_mem_gb` | Total cluster memory (GB) |
| `xenna/cluster/avail_mem_gb` | Available memory (GB) |

### Per-Stage Metrics

For each pipeline stage (e.g., `plan_stage`, `download_stage`, `bin_idx_tokenization_stage`):

| Metric | Description |
|--------|-------------|
| `xenna/stage/{name}/actors_target` | Target number of actors |
| `xenna/stage/{name}/actors_ready` | Actors ready to process |
| `xenna/stage/{name}/actors_running` | Actors currently processing |
| `xenna/stage/{name}/actors_idle` | Idle actors |
| `xenna/stage/{name}/tasks_completed` | Total completed tasks |
| `xenna/stage/{name}/queue_in` | Input queue depth |
| `xenna/stage/{name}/queue_out` | Output queue depth |
| `xenna/stage/{name}/slots_used` | Used processing slots |
| `xenna/stage/{name}/speed_tasks_per_s` | Processing speed |

### Per-Stage Resource Usage

| Metric | Description |
|--------|-------------|
| `xenna/stage/{name}/resource/cpu_util_pct` | CPU utilization percentage |
| `xenna/stage/{name}/resource/mem_gb` | Memory usage (GB) |
| `xenna/stage/{name}/resource/actor_count` | Number of actors |

## Usage in Recipes

The pretrain and SFT recipes automatically use the W&B hook when `wandb_log_pipeline_stats: true`:

```python
from nemotron.data_prep.xenna_wandb_hook import make_xenna_wandb_stats_hook

# Create hook if enabled
wandb_hook = make_xenna_wandb_stats_hook(
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
    "stage/download_stage/tasks_completed": 100
  },
  "stages": ["PlanStage", "DownloadStage", "BinIdxTokenizationStage"]
}
```

## Viewing in W&B

Once enabled, metrics appear in your W&B run dashboard:

1. Navigate to your run in the W&B UI
2. Go to the **Charts** tab
3. Metrics are organized under the `xenna/` namespace:
   - `xenna/progress` — Overall pipeline progress
   - `xenna/cluster/*` — Cluster resource utilization
   - `xenna/stage/*` — Per-stage metrics

### Recommended Charts

Create custom charts for common monitoring scenarios:

**Pipeline Progress**
```
xenna/progress, xenna/inputs_processed_per_s, xenna/outputs_per_s
```

**Cluster Utilization**
```
xenna/cluster/avail_cpus, xenna/cluster/avail_mem_gb
```

**Stage Throughput**
```
xenna/stage/*/speed_tasks_per_s
```

**Queue Depths (Bottleneck Detection)**
```
xenna/stage/*/queue_in, xenna/stage/*/queue_out
```

## Future: Native cosmos-xenna Support

This monkey-patching implementation is temporary. The planned approach is:

1. **PR to cosmos-xenna**: Add a `stats_callback` parameter to `PipelineConfig`
2. **Native integration**: cosmos-xenna calls the callback with `PipelineStats` directly
3. **Migration**: Replace monkey-patch with native callback once merged

The callback API will look like:

```python
# Future native API (not yet available)
def my_stats_callback(stats: PipelineStats) -> None:
    metrics = flatten_stats(stats)
    wandb.log(metrics)

pipeline_spec = pipelines_v1.PipelineSpec(
    ...,
    config=pipelines_v1.PipelineConfig(
        stats_callback=my_stats_callback,  # Native callback
        logging_interval_s=30,
    ),
)
```

Until then, the `XennaWandbStatsHook` provides equivalent functionality without requiring cosmos-xenna changes.

## API Reference

### xenna_wandb_hook.py

| Export | Description |
|--------|-------------|
| `XennaWandbStatsHook` | Context manager that patches `PipelineMonitor._make_stats` |
| `make_xenna_wandb_stats_hook()` | Factory function for recipes |

### XennaObservabilityConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `wandb_log_pipeline_stats` | `bool` | `True` | Enable W&B logging |
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
