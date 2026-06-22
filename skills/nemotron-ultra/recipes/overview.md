---
paper: "NVIDIA Nemotron 3 Ultra v3 Tech Report (2026-06-03)"
model: "nemotron-ultra"
section: "recipes-overview"
title: "Released recipe status for Nemotron 3 Ultra"
currency: "tracking"
---

# Recipe status: partial runnable Ultra3 coverage

Nemotron now includes a public **Ultra3** recipe tree for the open pretraining and SFT surfaces:

```text
src/nemotron/recipes/ultra3/
docs/nemotron/ultra3/
src/nemotron/cli/commands/ultra3/
```

Treat these as runnable/public recipe entry points, but do **not** describe them as a full reproduction of the Ultra technical report. The report's long-context pretraining data and full two-iteration MOPD teacher/checkpoint chain are not fully open-sourced.

## What exists in the repo today

| Artifact | Path | Notes |
|---|---|---|
| Base-model usage cookbook | `usage-cookbook/Nemotron-3-Ultra-Base/README.md` | Identity, base benchmark table, availability framing. |
| Recipe overview docs | `docs/nemotron/ultra3/README.md` | Public docs for current Ultra3 pretrain/SFT plus MOPD/quantization references. |
| Pretrain recipe | `src/nemotron/recipes/ultra3/stage0_pretrain/` | Megatron-Bridge pretrain wrapper, paper-style schedule/configs, data-prep script and blends. |
| SFT recipe | `src/nemotron/recipes/ultra3/stage1_sft/` | Default packed-Parquet SFT config plus `openmath.yaml` fallback/demo config. |
| Ultra3 CLI group | `src/nemotron/cli/commands/ultra3/` | Registers pretrain and SFT commands; verify data/build command registration before documenting command examples. |
| Container build registry | `src/nemotron/cli/kit/slurm/build.py` | Shared `nemotron kit slurm build <profile> --recipe ultra3 --stage {pretrain,sft}` path. |

## Current public-recipe gaps

- **Long-context pretraining phase:** the paper's LC data is not bundled. The docs describe how to run a CPT analog with user-provided long-context data.
- **Full MOPD reproduction:** the docs include a representative NeMo RL wiring guide, but not the full two-iteration teacher panel and checkpoint chain from the report.
- **Intermediate checkpoints:** paper teacher/student intermediates required for exact MOPD reproduction are not public recipe artifacts.
- **Command registration:** before answering with exact CLI commands, check the current CLI tree; some recipe scripts may exist before their convenience subcommands are registered.

## How to answer reproduction questions

1. Say plainly that **pretrain and SFT recipe surfaces exist** under `src/nemotron/recipes/ultra3/`.
2. Point to the relevant recipe docs:
   - Pretrain: `docs/nemotron/ultra3/pretrain.md`
   - SFT: `docs/nemotron/ultra3/sft.md`
   - MOPD mechanics: `docs/nemotron/ultra3/mopd.md`
3. Distinguish paper-faithful facts from public-recipe approximations and gaps.
4. Hand procedural/build customization to `/nemotron-customize` when the user asks to modify or extend the pipeline.

## When this file should change

Update this tracker when new Ultra3 stages land, especially if:

- `ultra3 data` or `ultra3 build` convenience commands are added,
- long-context data-prep/training configs are released,
- a fuller MOPD recipe with public checkpoints/teachers is added,
- eval/export/deploy stages are added to the Ultra3 recipe tree.
