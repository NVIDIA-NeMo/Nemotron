# Design Philosophy

Nemotron is a **cookbook**, not a framework. Each recipe is a complete, end-to-end example of a real training workflow — from data preparation through pretraining, fine-tuning, and reinforcement learning. Everything in this document follows from that choice.

A framework gives you building blocks and asks you to assemble them. A cookbook shows you the finished dish: here's how we actually trained this model, start to finish. The code is fully runnable, but it's not meant to be your team's production codebase. It's a reference implementation — you read it, understand the approach, and adapt the parts you need into your own setup.

This is a subtle but important distinction. A team's codebase evolves around their specific infrastructure, scale, and constraints. A cookbook is optimized for something different: **teaching by showing complete working examples**. Because it's a cookbook, we expect every team to customize it — different clusters, different data pipelines, different tracking tools. Customization isn't an edge case, it's the whole point.

That expectation shapes everything about how we write code. We favor clarity over abstraction, self-contained examples over reusable components, and making every step visible over hiding complexity behind convenience wrappers. When the code is explicit, a human can read a recipe and understand what to change. And just as importantly, an LLM or AI agent can too — which means customization can be assisted or fully driven by AI.

## What We Optimize For

**LLM-Native Development**: Structure the project so that AI agents and LLMs can easily understand, modify, and extend it.

We expect users to show up with prompts like:

> "Fork nemotron's CLI and replace nemo-run with SkyPilot"
> "Add a new data preprocessing stage to nano3"
> "Customize the SFT recipe to use my dataset format"

For an LLM to do this well, the codebase must be:
1. **Readable**: An LLM (or a human new to the project) can understand what the code does by reading it
2. **Self-contained**: Related logic lives together, not scattered across files
3. **Forkable**: Easy to copy a module and modify it for your needs

This is why we optimize for explicitness. When a user says "swap nemo-run for SkyPilot", an LLM should be able to read the relevant recipe and make the changes without getting lost in abstractions. This works because recipes are complete, end-to-end examples — not thin wrappers around hidden framework machinery.

## Design Principles

### 1. Explicit Over Implicit

Execution logic lives directly in each command function, not behind decorators or base classes:

```python
# Recipe metadata comes from PEP 723 [tool.runspec] block in the script itself
SPEC = parse_runspec(SCRIPT_PATH)

def _execute_pretrain(cfg: RecipeConfig):
    # ALL execution logic visible here
    train_config = parse_config(cfg.ctx, SPEC.config_dir, SPEC.config.default)
    job_config = build_job_config(train_config, ...)
    executor = create_executor(env=env, ...)  # <- Can see exactly what's happening
    with run.Experiment(name) as exp:
        exp.add(script_task, executor=executor)
        exp.run()
```

An LLM reads one file and sees the full picture: what config is loaded, how the executor is built, how the job is submitted.

### 2. Self-Describing Recipes

Recipe scripts declare their own identity and requirements via PEP 723 inline metadata:

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

The CLI layer reads this metadata via `nemo_runspec.parse()` instead of duplicating it. This keeps recipes portable -- any tool can read the same metadata.

### 3. Locality Over DRY

Keep related code together, even if it means some duplication.

**Why duplication is sometimes better:**
- Each command file is self-contained and readable
- Changes to one command don't accidentally affect others
- LLMs can understand one file without tracing through abstractions
- Forking is easy: copy the file, modify what you need

**What we don't share:**
- High-level orchestration (executor building, experiment running)
- Command-specific logic

**What we do share (via `nemo_runspec`):**
- Low-level utilities (env var building, packager construction)
- Config loading and merging
- Help formatting and display

### 4. Fork Over Extend

This is the cookbook principle in action: design for copying and modifying, not subclassing.

**Instead of inheritance hierarchies:**
```python
# cli/commands/nano3/pretrain.py - just copy and modify this file

def _execute_skypilot(cfg: RecipeConfig):
    import sky

    task = sky.Task(
        run="python main.py --config config.yaml",
        workdir=str(job_dir),
        num_nodes=env_config.get("nodes", 1),
    )
    sky.launch(task, cluster_name="nano3-pretrain")
```

This is simpler because:
- No inheritance hierarchy to understand
- No hook points to find
- No registration system to configure
- Just code you can read and change

## Two-Layer Architecture

We separate execution (how to run) from runtime (what to run):

| Layer | Purpose | Change Frequency |
|-------|---------|-----------------|
| **Execution** (`cli/commands/` + `nemo_runspec/`) | Job submission, tracking, orchestration | When changing backends |
| **Runtime** (`recipes/`) | Training algorithms, data processing | When changing algorithms |

The runtime layer is typically a **thin script** that delegates to NVIDIA AI stack libraries:

| Stage | Script | Library |
|-------|--------|---------|
| Pretrain | `train.py` | Megatron-Bridge |
| SFT | `train.py` | Megatron-Bridge |
| RL | `train.py` | NeMo-RL |
| Data prep | `data_prep.py` | Nemotron + Cosmos-Xenna |

**Key insight**: Execution is one forkable unit. Want SkyPilot + MLflow? Fork the CLI and rewrite it. Runtime scripts stay unchanged — they're just thin wrappers around library calls.

This separation means:
- Swapping nemo-run for SkyPilot only touches `cli/commands/`
- Changing training algorithms only touches `recipes/`
- Each layer can be forked independently

## Package Boundaries

| Package | Owns | Does NOT own |
|---------|------|-------------|
| **`nemo_runspec`** | Runspec parsing, config loading, env.toml, execution helpers, packaging, pipeline, artifact registry | Domain-specific artifact types, W&B integration |
| **`nemotron.kit`** | Artifact types, lineage trackers, W&B config, training script utilities | CLI, config loading, execution, packaging |
| **`nemotron.cli`** | Command implementations, typer app tree | Reusable infrastructure |

Dependency flows one way: `nemotron.cli` -> `nemo_runspec` + `nemotron.kit`. Never the reverse.

## What This Enables

### For LLMs

- **Clear boundaries**: "Read `cli/commands/nano3/pretrain.py`, replace the execution logic"
- **No dispatch tables**: No need to understand registration systems
- **No inheritance**: No base classes to trace
- **Self-describing scripts**: Metadata lives with the script, not in a separate config

### For Humans

- **Visible execution**: Read one file to understand how jobs run
- **Simple forking**: Copy file, modify, done
- **Testable runtime**: Training logic is pure and separate

### For Maintenance

- **Execution backends don't couple to algorithms**: Clean separation
- **Easy to add new models**: Just create new recipe directory
- **Easy to add new backends**: Just modify CLI layer

## Trade-offs We Accept

### More Duplication

Each command has similar executor setup code. We accept this because:
- Differences are visible (Slurm vs Ray, torchrun vs not)
- Changes are local to the file
- Forking is straightforward

### Bigger Diffs for Backend Changes

Changing from nemo-run to SkyPilot requires updating multiple command files. We accept this because:
- Each update is simple (no abstractions to navigate)
- The changes are mechanical (same pattern repeated)

### No Centralized Abstraction

There's no `run_recipe()` function that handles all execution. We accept this because:
- Each command shows its own execution path
- Differences between commands are visible
- LLMs can understand without tracing through layers

## References

- [The Grug Brained Developer](https://grugbrain.dev/) - Complexity bad, simple good
- [Write Plain Python](https://www.b-list.org/weblog/2024/feb/25/write-plain-python/) - Frameworks vs libraries
- [Locality of Behavior](https://htmx.org/essays/locality-of-behaviour/) - Keep related code together
