# Runspec: `[tool.runspec]` Specification

## What is runspec?

Runspec is a declarative metadata format for recipe scripts. Each recipe embeds
a `[tool.runspec]` block inside standard [PEP 723](https://peps.python.org/pep-0723/) inline script metadata,
describing **what the script is and what it needs** to run — identity, container
image, launch method, config layout, and resource defaults.

This keeps recipes self-describing and portable. Any tool (CLI, CI, notebook,
another cookbook) can read the same metadata without importing cookbook-specific
code.

## Motivation

Nemotron is a cookbook — a reference implementation you fork and customize. The
key architectural goal is that an agent (or human) can freely port recipes between
workflow engines (nemo-run, SkyPilot, plain bash) without reverse-engineering how
a script should be launched. Runspec makes this possible by separating **what a
script needs** (declarative metadata) from **how it gets run** (execution logic
in CLI commands).

Without runspec, recipe metadata is scattered across CLI wrappers, config files,
and docstrings. Adding a new recipe means editing multiple files. Porting to a
different workflow engine means tracing through code to rediscover identity,
image, launch method, and resource requirements. Runspec collapses all of this
into the script itself:

- **Identity**: what is this recipe called?
- **Environment**: what container image and setup does it need?
- **Launch**: how should it be started (torchrun, ray, direct)?
- **Config**: where do YAML configs live relative to the script?
- **Resources**: what are sensible defaults for nodes and GPUs?

Because this metadata is declarative and machine-readable, any tool can consume
it — a CLI, a CI pipeline, a notebook, or an agent building a completely
different execution backend. See the [Design Philosophy](../../docs/architecture/design-philosophy.md)
for the broader principles behind this approach.

## Format

Runspec lives inside [PEP 723](https://peps.python.org/pep-0723/) inline script metadata — the `# /// script` block
at the top of a Python file. Inside that block, runspec uses the `[tool.runspec]`
TOML table and its sub-tables.

```python
#!/usr/bin/env python3
# /// script
# [tool.runspec]
# schema = "1"
# docs = "https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md"
# name = "nano3/pretrain"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
# setup = "NeMo and all training dependencies are pre-installed in the image."
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# format = "omegaconf"
#
# [tool.runspec.resources]
# nodes = 1
# gpus_per_node = 8
# ///
```

Each line is prefixed with `# ` per PEP 723 convention. The block starts with
`# /// script` and ends with `# ///`.

## Schema (v1)

### `[tool.runspec]` — Top-level

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema` | `str` | `"1"` | Schema version. Always `"1"` for now. |
| `docs` | `str` | `""` | URL to this specification. Enables agents and tools to fetch the spec for context. Use a raw GitHub URL (e.g., `https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/docs/runspec/v1/spec.md`). |
| `name` | `str` | `""` | Recipe identity (e.g., `"nano3/pretrain"`, `"nano3/data/prep/sft"`). Used in job names, display, and directory layout. |
| `image` | `str?` | `null` | Default container image for remote execution. |
| `setup` | `str` | `""` | Human-readable description of what the image provides or what setup is needed. |

### `[tool.runspec.run]` — Launch configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `launch` | `str` | `"torchrun"` | How to start the script: `"torchrun"` (distributed), `"ray"` (Ray cluster), or `"direct"` (plain python). |
| `cmd` | `str` | `"python {script} --config {config}"` | Command template. Supports `{script}` and `{config}` placeholders. |
| `workdir` | `str?` | `null` | Working directory inside the container (e.g., `"/opt/nemo-rl"`). |

### `[tool.runspec.config]` — Config layout

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dir` | `str` | `"./config"` | Config directory, relative to the script file. |
| `default` | `str` | `"default"` | Name of the default config (without extension). |
| `format` | `str` | `"omegaconf"` | Config format: `"omegaconf"`, `"yaml"`, or `"json"`. |

### `[tool.runspec.resources]` — Resource defaults

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nodes` | `int` | `1` | Default number of nodes. |
| `gpus_per_node` | `int` | `8` | Default GPUs per node. |

### `[tool.runspec.env]` — Environment variables

An optional key-value table of environment variables to set at runtime:

```toml
[tool.runspec.env]
NCCL_NVLS_ENABLE = "0"
```

## How the CLI uses runspec

A typical CLI command reads runspec at import time and uses it throughout:

```python
from nemo_runspec import parse as parse_runspec

SCRIPT_PATH = "src/nemotron/recipes/nano3/stage0_pretrain/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

# SPEC.name        → "nano3/pretrain"
# SPEC.image       → container image for remote execution
# SPEC.config_dir  → absolute path to config directory
# SPEC.config.default → "default" config name
# SPEC.run.launch  → "torchrun"
```

The command file contains the visible execution logic — runspec tells it **what**
to run, the command decides **how** (local, slurm, ray, etc.).

## Adding a new recipe

1. Write the training/data-prep script
2. Add a `[tool.runspec]` block at the top
3. Create a CLI command that calls `parse_runspec()` on it
4. Register the command in the typer group

The runspec block is the single source of truth. No need to duplicate metadata
in the CLI layer.

## Reference implementation

The `nemo_runspec` Python package (`src/nemo_runspec/`) implements:

- **Parsing**: `_parser.py` extracts the PEP 723 block and parses the TOML
- **Models**: `_models.py` defines the frozen `Runspec` dataclass
- **CLI toolkit**: config loading, env.toml profiles, RecipeTyper, packaging, etc.

See `src/nemo_runspec/README.md` for package-level documentation.
