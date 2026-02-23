# CLI Architecture

This document explains how Nemotron's CLI layer works and how to modify it for your needs.

## Overview

The CLI layer (`src/nemotron/cli/`) handles **execution** -- how jobs are submitted and tracked. Each command file contains visible execution logic, making it easy to understand and modify.

The shared toolkit lives in the `nemo_runspec` package: config loading, env.toml profiles, execution helpers, packaging, and display utilities. CLI commands import from `nemo_runspec` and wire things together explicitly.

## Design Principle: Visible Execution

Nemotron makes execution **explicit** -- all nemo-run setup lives directly in each command function:

```python
# src/nemotron/cli/commands/nano3/pretrain.py

from nemo_runspec import parse as parse_runspec
from nemo_runspec.config import parse_config, build_job_config, save_configs
from nemo_runspec.execution import create_executor, execute_local, build_env_vars

# Metadata comes from PEP 723 [tool.runspec] in the script itself
SCRIPT_PATH = "src/nemotron/recipes/nano3/stage0_pretrain/train.py"
SPEC = parse_runspec(SCRIPT_PATH)

# Execution logic is VISIBLE in the function
def _execute_pretrain(cfg: RecipeConfig, *, experiment=None):
    # 1. Parse configuration
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    job_config = build_job_config(train_config, cfg.ctx, SPEC.name, ...)

    # 2. Build executor - THIS IS WHAT YOU'D CHANGE FOR SKYPILOT
    executor = create_executor(env=env, env_vars=env_vars, packager=packager, ...)

    # 3. Run experiment
    with run.Experiment(recipe_name) as exp:
        exp.add(script_task, executor=executor)
        exp.run(detach=not attached)
```

## Key Components

### Runspec (PEP 723 metadata)

Recipe scripts are self-describing. Each script declares its identity, container image, launch method, and config location via a `[tool.runspec]` TOML block:

```python
# /// script
# [tool.runspec]
# name = "nano3/pretrain"
# image = "nvcr.io/nvidia/nemo:25.11.nemotron_3_nano"
#
# [tool.runspec.run]
# launch = "torchrun"
#
# [tool.runspec.config]
# dir = "./config"
# default = "default"
# ///
```

The CLI reads this via `nemo_runspec.parse()`, returning a frozen `Runspec` dataclass. See [`src/nemo_runspec/README.md`](../../src/nemo_runspec/README.md) for the full schema.

### RecipeConfig

Parsed CLI options. Handles late globals (`--run` after subcommand) and dotlist overrides:

```python
from nemo_runspec.recipe_config import RecipeConfig, parse_recipe_config

cfg = parse_recipe_config(ctx)
# Now you have:
#   cfg.mode       - "run", "batch", or "local"
#   cfg.attached   - True if --run, False if --batch
#   cfg.profile    - The env profile name
#   cfg.passthrough - Args to pass through to script
#   cfg.dry_run    - True if --dry-run
```

### RecipeTyper

Standardizes command registration with proper context settings and rich help panels:

```python
from nemo_runspec.recipe_typer import RecipeTyper, RecipeMeta

app = RecipeTyper(name="nano3", help="Nano3 training recipes")

app.add_recipe_command(
    pretrain,
    meta=PRETRAIN_META,  # RecipeMeta with config_dir, artifacts, etc.
    rich_help_panel="Training Stages",
)
```

### Config Pipeline

Config loading uses `nemo_runspec.config`:

```python
from nemo_runspec.config import parse_config, build_job_config, extract_train_config, save_configs

# 1. Load YAML config with dotlist overrides
train_config = parse_config(ctx, config_dir, default_config)

# 2. Build full job config with provenance (env profile, CLI args, etc.)
job_config = build_job_config(train_config, ctx, recipe_name, script_path, argv, env_profile=env)

# 3. Extract clean train config for the script
train_config_for_script = extract_train_config(job_config, for_remote=True)

# 4. Save both configs to job directory
job_path, train_path = save_configs(job_config, train_config_for_script, job_dir)
```

## Directory Structure

```
src/nemotron/
├── cli/                              # EXECUTION LAYER
│   ├── bin/
│   │   └── nemotron.py               # Main entry point (typer app)
│   ├── commands/
│   │   └── nano3/
│   │       ├── _typer_group.py        # Command registration (RecipeTyper)
│   │       ├── pretrain.py            # Pretrain execution logic
│   │       ├── sft.py                 # SFT execution logic
│   │       ├── rl.py                  # RL execution logic (Ray)
│   │       ├── data/
│   │       │   └── prep/              # Data prep commands
│   │       │       ├── pretrain.py
│   │       │       ├── sft.py
│   │       │       └── rl.py
│   │       └── model/                 # Model import/eval commands
│
├── recipes/                           # RUNTIME LAYER
│   └── nano3/
│       ├── stage0_pretrain/
│       │   └── train.py               # -> Megatron-Bridge
│       ├── stage1_sft/
│       │   └── train.py               # -> Megatron-Bridge
│       └── stage2_rl/
│           └── train.py               # -> NeMo-RL

src/nemo_runspec/                      # SHARED TOOLKIT
├── _parser.py                         # PEP 723 [tool.runspec] parsing
├── _models.py                         # Runspec, RunspecRun, RunspecConfig, RunspecResources
├── config/                            # Config loading and OmegaConf resolvers
│   ├── loader.py                      # parse_config, build_job_config, save_configs
│   └── resolvers.py                   # ${art:...}, ${auto_mount:...}
├── env.py                             # env.toml profile loading with inheritance
├── cli_context.py                     # GlobalContext (shared CLI state)
├── recipe_config.py                   # RecipeConfig + parse_recipe_config
├── recipe_typer.py                    # RecipeTyper + RecipeMeta
├── help.py                            # Rich help panels
├── display.py                         # Dry-run and job submission display
├── execution.py                       # Startup commands, env vars, executor creation
├── run.py                             # RunConfig, nemo-run patches
├── packaging/                         # SelfContainedPackager, CodePackager
├── squash.py                          # Container squash utilities
├── pipeline.py                        # Pipeline orchestration
├── step.py                            # Step definition for pipelines
├── artifact_registry.py               # ArtifactRegistry (fsspec/wandb)
└── exceptions.py                      # ArtifactNotFoundError, etc.
```

## Execution Patterns

### Training (Slurm + torchrun)
Pretrain and SFT use Slurm with torchrun launcher:

```python
# cli/commands/nano3/pretrain.py, cli/commands/nano3/sft.py
executor = create_executor(env=env, env_vars=env_vars, packager=packager, ...)
script_task = run.Script(path="main.py", args=[...], entrypoint="python")

with run.Experiment(recipe_name) as exp:
    exp.add(script_task, executor=executor)
    exp.run()
```

### RL (Ray)
RL uses Ray for distributed execution:

```python
# cli/commands/nano3/rl.py
from nemo_run.run.ray.job import RayJob

executor = create_executor(env=env, ...)  # Still Slurm for infrastructure
ray_job = RayJob(name=job_name, executor=executor)

ray_job.start(
    command=cmd,
    workdir=str(Path.cwd()) + "/",
    pre_ray_start_commands=setup_commands,
)
```

### Data Prep (Ray + Code Packager)
Data prep uses Ray with full codebase rsync. The `[tool.runspec]` in the data prep script declares `launch = "ray"`:

```python
# cli/commands/nano3/data/prep/pretrain.py
SPEC = parse_runspec(SCRIPT_PATH)  # launch=ray, cmd template from [tool.runspec]
```

## How to Fork for Different Backends

### Example: Replace nemo-run with SkyPilot

1. **Read the current execution logic** in `cli/commands/nano3/pretrain.py`
2. **Replace `_execute_remote()`** with SkyPilot equivalents:

```python
# cli/commands/nano3/pretrain.py (forked for SkyPilot)

def _execute_skypilot(cfg: RecipeConfig):
    import sky

    # Config loading stays the same
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    job_config = build_job_config(train_config, ...)
    job_dir = generate_job_dir(SPEC.name)
    _, train_path = save_configs(job_config, ..., job_dir)

    task = sky.Task(
        run="python main.py --config config.yaml",
        workdir=str(job_dir),
        num_nodes=env_config.get("nodes", 1),
    )
    task.set_resources(sky.Resources(
        cloud=sky.AWS(),
        accelerators=f"A100:{env_config.get('gpus_per_node', 8)}",
    ))

    sky.launch(task, cluster_name="nano3-pretrain")
```

3. **Keep config loading** -- `nemo_runspec.config` works with any backend
4. **Keep the `[tool.runspec]` block** -- metadata stays with the script

## Shared Utilities

Low-level helpers in `nemo_runspec.execution` are shared across commands but keep orchestration visible:

| Function | Module | Description |
|----------|--------|-------------|
| `create_executor()` | `nemo_runspec.execution` | Build nemo-run executor from env config |
| `build_env_vars()` | `nemo_runspec.execution` | Build environment variables (HF, W&B tokens) |
| `execute_local()` | `nemo_runspec.execution` | Local subprocess execution via torchrun |
| `get_startup_commands()` | `nemo_runspec.execution` | Extract startup commands from env config |
| `ensure_squashed_image()` | `nemo_runspec.squash` | Container squashing for cluster |
| `clone_git_repos_via_tunnel()` | `nemo_runspec.execution` | Git repo cloning via SSH tunnel |

These are utilities, not abstractions. The calling code shows exactly how they're used.

## CLI Behavior Reference

| Feature | How It Works |
|---------|--------------|
| Late globals | `split_unknown_args()` in `parse_recipe_config()` |
| Dotlist overrides | Applied during `parse_config()` via OmegaConf |
| Packager selection | `SelfContainedPackager` or `CodePackager` from `nemo_runspec.packaging` |
| Ray execution | Visible in `_execute_remote()` functions in RL/data prep commands |
| Rich help panels | `RecipeTyper` + `RecipeMeta` from `nemo_runspec.recipe_typer` |
| env.toml profiles | Loaded via `nemo_runspec.env.parse_env()` with inheritance |
