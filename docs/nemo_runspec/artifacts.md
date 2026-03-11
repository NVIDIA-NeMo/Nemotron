# Artifact Tracking

`nemo_runspec.artifacts` provides the tracking infrastructure that connects data prep, training, and evaluation stages through versioned artifact references. It supports two backends that can run simultaneously:

- **Manifest-based tracking** (`nemo_runspec.manifest_tracker`) — writes structured JSON manifests to any fsspec-compatible storage (local, Lustre, S3, GCS, HF Hub). Always reliable, zero external dependencies.
- **W&B artifact tracking** — logs artifacts to [Weights & Biases](../nemotron/wandb.md) for team collaboration, UI browsing, and lineage graphs.

Manifests are the always-on foundation; W&B is best-effort on top.

> For artifact *types* (PretrainBlendsArtifact, SFTDataArtifact, etc.) and the Pydantic base class, see [nemotron.kit artifacts](../nemotron/artifacts.md).

## Configuration

### env.toml

Configure artifact tracking in your `env.toml`. These are team-wide defaults merged into every job config by `build_job_config()`:

```toml
# Both backends: manifest always, wandb best-effort
[artifacts]
wandb = true

[artifacts.manifest]
root = "/lustre/team/artifacts"

[wandb]
project = "nemotron"
entity = "YOUR-TEAM"
```

Backend combinations:

| Setup | Use Case |
|-------|----------|
| `manifest` + `wandb = true` | **Recommended.** Manifest is always reliable; wandb adds UI + team features |
| `manifest` only | Offline / local development, no wandb dependency |
| `wandb = true` only | Legacy behavior, wandb-only tracking |
| Neither | No artifact tracking |

### Disabling per-run

Override via dotlist on any command:

```bash
# Run with manifest only (no wandb)
uv run nemotron super3 sft --run dlw artifacts.wandb=false

# Fully disable artifact tracking
uv run nemotron super3 sft --run dlw artifacts.manifest.root=null artifacts.wandb=false
```

### Manifest Storage Backends

The `manifest.root` value supports any fsspec-compatible URI:

| Backend | Example | Notes |
|---------|---------|-------|
| Local / Lustre / NFS | `/lustre/team/artifacts` | Best for training clusters |
| S3 | `s3://bucket/artifacts` | Requires AWS credentials |
| GCS | `gs://bucket/artifacts` | Requires GCP credentials |
| HF Hub | `hf://org/repo-name` | Great for sharing; batched commits |

### Config Flow

1. `build_job_config()` merges `[artifacts]` from env.toml into the job config (YAML overrides env.toml)
2. `extract_train_config()` preserves the top-level `artifacts:` section
3. Inside the container, `setup_artifact_tracking()` reads `config.artifacts` to initialize backends
4. `build_env_vars()` extracts `WANDB_PROJECT` / `WANDB_ENTITY` from `[wandb]` in env.toml

## The Two-Function API

Scripts use two functions from `nemo_runspec.artifacts`:

```python
from nemo_runspec.artifacts import setup_artifact_tracking, log_artifact
```

### `setup_artifact_tracking(config)` — Initialize backends and resolve references

Called early, **before** dataclass conversion. Reads `config.artifacts`, initializes the active backends, and resolves `${art:...}` references.

```python
config = load_omegaconf_yaml(config_path)
tracking = setup_artifact_tracking(config)  # BEFORE dataclass conversion
cfg = omegaconf_to_dataclass(config, MyConfig)
```

Returns an `ArtifactTrackingResult` with flags:
- `tracking.manifest` — whether manifest backend is active
- `tracking.wandb` — whether wandb backend is active
- `tracking.qualified_names` — wandb artifact qualified names for lineage registration

### `log_artifact(artifact, tracking)` — Save to all active backends

Saves the artifact to **all** active backends. Replaces `artifact.save()` in data-prep scripts:

```python
artifact = SFTDataArtifact(path=output_dir, pack_size=4096, ...)
log_artifact(artifact, tracking)  # logs to manifest + wandb
```

When both backends are active:
1. `artifact.save()` logs via the global lineage tracker (WandbTracker after `init_wandb_from_env()`)
2. `log_artifact` additionally writes to ManifestTracker (stored reference survives the global overwrite)

When only one backend is active, `artifact.save()` handles everything and `log_artifact` is effectively a pass-through.

### Resolution Priority

When resolving `${art:...}` references:

| Active Backends | Resolution Method |
|----------------|-------------------|
| wandb + manifest | Resolve via wandb API (downloads artifact, gets lineage) |
| manifest only | Resolve from local/fsspec manifest files |
| wandb only | Resolve via wandb API |
| neither | Fall through to local resolver |

## Usage Patterns

### Train Scripts (self-contained packager)

Train scripts are inlined into `main.py` by the self-contained packager. They use `setup_artifact_tracking` for initialization and stage-specific monkey-patches for checkpoint logging:

```python
tracking = setup_artifact_tracking(config, artifacts_key="run")

# Stage-specific monkey-patches based on active backends
if tracking.manifest and tracking.wandb:
    patch_checkpoint_logging_both()
elif tracking.wandb:
    patch_wandb_checkpoint_logging()
elif tracking.manifest:
    patch_manifest_checkpoint_logging()

# Wandb lineage: register input artifacts after wandb.init()
if tracking.wandb:
    patch_wandb_init_for_lineage(
        artifact_qualified_names=tracking.qualified_names,
        tags=["pretrain"],
    )
```

### Data Prep Scripts (code packager)

Data prep scripts use the code packager (full rsync), so the complete `nemo_runspec` API is available at runtime. They use `log_artifact` for explicit artifact saving:

```python
tracking = setup_artifact_tracking(config)

if tracking.wandb:
    init_wandb_from_env()          # creates wandb run for metrics

# ... run pipeline ...

log_artifact(artifact, tracking)   # logs to manifest + wandb
wandb_kit.finish_run(exit_code=0)
```

## Manifest Directory Structure

```
{root}/
├── nano3-pretrain-data/
│   ├── v1/
│   │   ├── manifest.json          # Full provenance record
│   │   └── metadata.json          # Resolver-compatible format
│   ├── v2/
│   │   ├── manifest.json
│   │   └── metadata.json
│   └── latest                     # Plain text file: "v2"
│
├── nano3-sft-model/
│   ├── v1/
│   │   ├── manifest.json
│   │   └── metadata.json
│   └── latest
```

- **Zero-copy**: Data stays at its original location. Only JSON metadata is written.
- **`latest` file**: Plain text containing the version directory name (e.g., `v2`). Works on all filesystems — no symlink issues.
- **Version discovery**: Scan `v*/` directories. `latest` provides O(1) latest lookup.

### manifest.json

Full provenance and lineage record:

```json
{
  "name": "nano3-sft-data",
  "version": 2,
  "type": "SFTDataArtifact",
  "path": "/lustre/data/nano3/sft/output/splits",
  "created_at": "2026-03-11T10:30:00-07:00",
  "producer": "nemo_abc123",
  "metadata": {
    "pack_size": 4096,
    "total_tokens": 15000000
  },
  "inputs": ["hf://nvidia/dataset-a"],
  "used_artifacts": ["nano3-pretrain-data:v2"]
}
```

### metadata.json

Resolver-compatible format — the full Pydantic `model_dump()` output. The `${art:data,field}` resolver reads fields from this file.

## CLI Display

### Job Submission Panel

Shows which artifact stores are active before the job runs:

```
Job Submission
├── configs
│   ├── job:   /path/to/job.yaml
│   └── train: /path/to/train.yaml
├── artifacts
│   ├── manifest: /lustre/.../artifacts
│   └── wandb:    ✓ enabled
└── mode: attached
```

### Completion Panel

After a data prep or training job completes:

```
╭── Step Complete (super3-pretrain-data-tiny) ──╮
│ ...                                           │
│ Manifest: /lustre/.../super3-pretrain-data/v1 │
│ W&B:      https://wandb.ai/romeyn/nemotron/.. │
╰───────────────────────────────────────────────╯
```

## Viewing Lineage

### From Manifest Files

Inspect manifests directly on the filesystem:

```bash
# See latest version
cat /lustre/artifacts/nano3-sft-data/latest
# v2

# Read manifest
cat /lustre/artifacts/nano3-sft-data/v2/manifest.json
# {"name": "nano3-sft-data", "version": 2, "path": "...", ...}
```

### In W&B

1. Navigate to your project's **Artifacts** tab
2. Select any artifact (e.g., `ModelArtifact-rl`)
3. Click the **Graph** view to see upstream dependencies

## Troubleshooting

### "Artifact not found"

- Check artifact name and version (inspect manifest directory or W&B UI)
- Verify `manifest.root` in env.toml points to the correct location
- For wandb: ensure `project` and `entity` match, and run `wandb login`

### Manifest path empty in completion panel

- Verify `[artifacts.manifest]` is configured in `env.toml`
- Check that the manifest root directory is writable

### Wandb digest mismatch

The pipeline automatically patches wandb's digest verification for local file references. If you still see errors, ensure you're using the latest version of the pipeline code.

### Running without wandb

Set `artifacts.wandb=false` on the command line or remove the `wandb = true` line from `[artifacts]` in env.toml. Manifest tracking works independently.

## Further Reading

- [Artifact Types](../nemotron/artifacts.md) – Pydantic artifact classes and custom types
- [OmegaConf Resolvers](./omegaconf.md) – `${art:...}` interpolations
- [W&B Integration](../nemotron/wandb.md) – credentials and configuration
- [Execution & env.toml](./nemo-run.md) – execution profiles and env.toml reference
