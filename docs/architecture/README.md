# Nemotron Architecture

This directory contains documentation about Nemotron's architecture and design principles.

## Overview

Nemotron is a **cookbook** - a reference implementation showing best practices for training LLMs at scale. It's not a framework you install; it's a codebase you fork and customize.

## Key Documents

- [Runspec Specification](../runspec/v1/spec.md) - The `[tool.runspec]` metadata format for recipe scripts
- [CLI Architecture](cli-architecture.md) - How the CLI layer works and how to fork it
- [Design Philosophy](design-philosophy.md) - What we optimize for and why

## Quick Start

```bash
# Run pretraining locally
nemotron nano3 pretrain -c tiny

# Submit to cluster
nemotron nano3 pretrain -c tiny --run dgx

# See what's happening (execution logic is visible in the code)
# Open: src/nemotron/cli/commands/nano3/pretrain.py
```

## Two-Layer Architecture

| Layer | What | Where | Fork When |
|-------|------|-------|-----------|
| **Execution** | How to run and track experiments | `cli/commands/` + `nemo_runspec/` | nemo-run/wandb -> SkyPilot/mlflow |
| **Runtime** | Training/data processing | `recipes/` | Algorithm changes |

```
Execution Layer                    Runtime Layer (recipes/)
┌──────────────────────────┐      ┌─────────────────────┐
│ cli/commands/nano3/      │      │ recipes/nano3/      │
│   pretrain.py            │      │   pretrain/train.py │
│                          │      │                     │
│ nemo_runspec (toolkit)   │─────►│ Megatron-Bridge     │
│   config, execution, env │      │                     │
└──────────────────────────┘      └─────────────────────┘
```

The runtime layer is typically a **thin script** that delegates to NVIDIA AI stack libraries. The execution layer contains all the job submission logic, which is what you'd change to swap nemo-run for SkyPilot or another backend.

## Package Responsibilities

| Package | Scope |
|---------|-------|
| **`nemo_runspec`** | Generic CLI toolkit: PEP 723 runspec parsing, config loading, env.toml profiles, execution helpers, packaging, pipeline orchestration |
| **`nemotron.kit`** | Domain-specific: artifact types (pretrain data, SFT data, checkpoints), lineage tracking (W&B, file-based), W&B integration |
| **`nemotron.cli`** | CLI commands: visible execution logic per command, typer-based command tree |
| **`nemotron.recipes`** | Runtime scripts: training, data prep, RL (thin scripts delegating to NVIDIA AI stack) |

Dependency direction: `nemotron.cli` -> `nemo_runspec` + `nemotron.kit` -> NVIDIA stack. Never the reverse.
