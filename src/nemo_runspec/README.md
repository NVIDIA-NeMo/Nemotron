# nemo_runspec

Bridge layer for PEP 723 `[tool.runspec]` metadata. Parses declarative metadata
from recipe scripts and provides the shared CLI toolkit that Nemotron commands
build on.

## Philosophy

Recipe scripts should be self-describing. Rather than scattering identity,
container images, launch methods, and resource defaults across CLI wrappers and
config files, each recipe script declares all of this as standard PEP 723 inline
metadata in a `[tool.runspec]` block at the top of the file. The CLI layer reads
this metadata and stays thin -- it doesn't encode policy about *how* to run a
script, it just asks the script what it needs. This keeps recipes portable
(any tool can read the same metadata), eliminates hidden coupling between CLI
commands and the scripts they wrap, and makes it trivial to add a new recipe:
write the script, add the `[tool.runspec]` block, and the CLI machinery picks
it up automatically.

## What it does

`nemo_runspec` solves two problems:

1. **Runspec parsing** -- Extracts `[tool.runspec]` TOML from PEP 723 inline
   script metadata blocks, returning a frozen `Runspec` dataclass describing a
   recipe's identity, container image, launch method, config directory, and
   resource requirements.

2. **CLI toolkit** -- Provides the reusable building blocks that every recipe
   command needs: config loading, env.toml profile resolution, display helpers,
   `RecipeTyper`, packaging, and nemo-run support.

## Quick start

```python
from nemo_runspec import parse

SPEC = parse("src/nemotron/recipes/nano3/stage0_pretrain/train.py")
print(SPEC.name)        # "nano3/pretrain"
print(SPEC.image)       # "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
print(SPEC.config_dir)  # Path("/abs/path/to/config")
```

## Runspec schema

See [docs/runspec/v1/spec.md](../../docs/runspec/v1/spec.md) for the
full `[tool.runspec]` specification -- field reference, format, and usage guide.

## Package modules

| Module | Purpose |
|--------|---------|
| `_parser` | PEP 723 TOML extraction and `[tool.runspec]` parsing |
| `_models` | Frozen `Runspec`, `RunspecRun`, `RunspecConfig`, `RunspecResources` dataclasses |
| `config/` | Config loading and OmegaConf resolver package |
| `config/loader` | Config pipeline: YAML loading, dotlist overrides, profile merging, job YAML |
| `config/resolvers` | OmegaConf resolvers: `${art:...}` artifact resolution, `${auto_mount:...}` git mounts |
| `env` | `env.toml` profile loading with inheritance (`extends`), plus wandb/cache/artifacts config |
| `cli_context` | `GlobalContext` for shared CLI state (config, run, batch, dry-run) |
| `recipe_config` | `RecipeConfig` -- normalizes CLI options into a typed object |
| `recipe_typer` | `RecipeTyper` -- Typer subclass standardizing recipe command registration |
| `help` | `RecipeCommand` with custom Rich help panels (configs, overrides, profiles) |
| `display` | Rich display utilities for dry-run output and job submission summaries |
| `step` | `Step` dataclass for pipeline step definition (module, torchrun, command builders) |
| `exceptions` | `ArtifactNotFoundError`, `ArtifactVersionNotFoundError` |
| `artifact_registry` | `ArtifactRegistry` with fsspec/wandb backends, global accessors, resolver mode |
| `filesystem` | `ArtifactFileSystem` -- fsspec filesystem for `art://` URIs |
| `run` | nemo-run patches (Ray CPU template, rsync host key handling) |
| `pipeline` | Pipeline orchestration: local subprocess piping, nemo-run, and sbatch launchers |
| `execution` | Execution helpers: startup commands, env vars, executor creation, local run |
| `packaging` | `SelfContainedPackager` and `CodePackager` for remote execution |
| `squash` | Container squash utilities (Docker to enroot sqsh, ensure squashed on cluster) |
| `templates/` | Custom Ray CPU Slurm template (`ray_cpu.sub.j2`) |
| `evaluator` | Evaluator helpers: task flag parsing, W&B injection, config save, image collection |
| `utils` | Shared utilities like `${run.*}` template interpolation |

## env.toml

Environment configuration uses TOML profiles with inheritance:

```toml
[base]
executor = "slurm"
account = "my-account"
remote_job_dir = "/lustre/jobs"

[dev]
extends = "base"
partition = "dev-gpu"
nodes = 1

[prod]
extends = "base"
partition = "prod-gpu"
nodes = 8

[wandb]
entity = "my-team"
project = "nemotron"

[artifacts]
backend = "file"
root = "/lustre/artifacts"

[cache]
git_dir = "/lustre/git-cache"
```

Profiles are selected via `--run <profile>` or `--batch <profile>`.
Special sections (`wandb`, `cli`, `cache`, `artifacts`) are not executor profiles.
