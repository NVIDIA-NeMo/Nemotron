# Build function-calling benchmarks

Build BFCL-compatible function-calling datasets from deterministic, executable
Oracle Packs. The pipeline validates a local Python or HTTPS oracle, generates
and replays tasks in process isolation, checkpoints each stage, and publishes a
content-addressed parquet dataset with optional compatibility exports.

```{toctree}
:maxdepth: 2

getting-started
explanation/index
how-to/index
reference/index
```
