# Investigation: WandB Stage Charts Not Consolidated

## Summary
Multiple logging paths are emitting per-stage scalar metrics, creating separate charts instead of one chart per metric with lines for each stage. The `WandbStatsHook` is designed to produce consolidated `line_series` charts, but the `stats_callback` path also emits per-stage scalar keys that create extra charts.

## Symptoms
- WandB shows many separate charts (one per stage per metric) instead of consolidated charts
- Expected: One chart for "tasks_completed" with lines for Plan, Download, Tokenize stages
- Actual: Separate charts for each stage's metrics

## Investigation Log

### Phase 1 - Identify Logging Paths
**Hypothesis:** Multiple logging mechanisms may be conflicting

**Findings:** There are THREE independent W&B logging paths:

1. **`WandbStatsHook`** (`wandb_hook.py`)
   - Uses `wandb.plot.line_series` for consolidated charts
   - Logs to `{pipeline_kind}/charts/{metric_name}` (e.g., `pretrain/charts/tasks_completed`)
   - Designed to NOT emit per-stage scalar keys
   - **This is the correct approach** for "one chart per metric with lines per stage"

2. **`stats_callback`** (`stats_callback.py`)
   - Logs scalar keys: `pipeline/stage/{stage_id}/actors/target`, etc.
   - Each unique key creates a separate chart in WandB
   - **This creates chart proliferation**

3. **`PrometheusMetricsLogger`** (`prometheus_metrics.py`)
   - Logs scalar keys: `{namespace}/stages/{metric_name}_{state}/{stage_id}`
   - Same outcome: per-stage keys → many charts

**Evidence:**
- `wandb_hook.py:465-489` - Creates `line_series` charts under `{ns}/charts/{metric_name}`
- `stats_callback.py:127-159` - Logs `pipeline/stage/{stage_key}/...` scalar keys
- `prometheus_metrics.py:188-200` - Logs `{namespace}/stages/{metric_name}/{stage_id}` scalar keys

**Conclusion:** The `stats_callback` path is the likely culprit since it's the fallback mechanism and emits per-stage scalar keys.

### Phase 2 - Data Structure Mismatch
**Hypothesis:** `WandbStatsHook` and `stats_callback` expect different `PipelineStats` shapes

**Findings:**
- `wandb_hook.py:158-162` - Iterates `stats.actor_pools` as a **list of objects** with `.name` attribute:
  ```python
  for pool in stats.actor_pools:
      stage_id = canonical_stage_id(pool.name)
  ```

- `stats_callback.py:126-127` - Iterates `stats.actor_pools` as a **dict** with `.items()`:
  ```python
  for stage_name, pool_stats in stats.actor_pools.items():
  ```

**Evidence:** Lines 158-162 in `wandb_hook.py` vs lines 126-127 in `stats_callback.py`

**Conclusion:** If the runtime provides a dict-shaped `actor_pools`, `WandbStatsHook._extract_stage_metrics()` will fail silently (caught by try/except in `_on_stats`), falling back to only pipeline-level metrics while `stats_callback` still emits per-stage charts.

### Phase 3 - Configuration Analysis
**Hypothesis:** Both paths may be enabled simultaneously

**Findings:** Looking at `config.py` and recipe usage:
- `ObservabilityConfig.wandb_log_pipeline_stats` enables both the hook AND potentially the stats_callback
- The recipes (`pretrain.py`, `sft.py`) create `WandbStatsHook` but may also pass a `stats_callback` to the pipeline

**Conclusion:** If both are active, you get duplicate (and conflicting) logging.

## Root Cause

**Primary Issue:** The `stats_callback` path emits per-stage scalar metric keys (`pipeline/stage/{stage}/...`), which creates one WandB chart per unique key. Even if `WandbStatsHook` correctly produces consolidated `line_series` charts, the scalar keys from `stats_callback` still appear as separate charts.

**Secondary Issue:** Potential data structure mismatch - `wandb_hook.py` expects `actor_pools` as a list, `stats_callback.py` expects it as a dict. This could cause `WandbStatsHook` to silently fail stage metric extraction.

## Recommendations

### Fix 1: Disable stats_callback per-stage metrics when WandbStatsHook is active
Add a flag to `ObservabilityConfig` to control whether `stats_callback` should emit per-stage metrics:

```python
# config.py
@dataclass
class ObservabilityConfig:
    # ... existing fields ...
    wandb_consolidated_charts_only: bool = True  # When True, stats_callback skips per-stage scalars
```

Then in `stats_callback.py`:

```python
def _extract_wandb_metrics(stats: Any, start_time: float, *, skip_per_stage: bool = False) -> dict[str, float | int]:
    # ... pipeline-level metrics ...

    if not skip_per_stage:
        # Per-stage metrics from actor_pools
        if hasattr(stats, "actor_pools") and stats.actor_pools:
            # ... existing per-stage logging ...
```

### Fix 2: Normalize actor_pools iteration in WandbStatsHook
Make `_extract_stage_metrics` handle both list and dict shapes:

```python
def _iter_actor_pools(actor_pools: Any) -> list[tuple[str, Any]]:
    """Iterate actor_pools whether it's a list or dict."""
    if not actor_pools:
        return []
    if isinstance(actor_pools, dict):
        return list(actor_pools.items())
    # Assume iterable of objects with .name
    return [(pool.name, pool) for pool in actor_pools if hasattr(pool, "name")]
```

### Fix 3: Single source of truth for stage metrics
Ensure only ONE path logs stage metrics to WandB:
- If using `WandbStatsHook` (recommended): disable per-stage logging in `stats_callback`
- If using `stats_callback` only: convert it to use `line_series` charts

## Preventive Measures

1. **Add integration tests** that verify WandB receives the expected chart structure (one chart per metric, multiple lines per stage)

2. **Document the intended logging architecture** - make it clear that `WandbStatsHook` is the primary mechanism and `stats_callback` should only log pipeline-level metrics

3. **Add runtime warnings** if multiple stage-metric logging paths are detected active simultaneously
