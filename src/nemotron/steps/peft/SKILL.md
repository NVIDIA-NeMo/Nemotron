---
name: nemotron-peft
description: Choose and configure Nemotron PEFT or LoRA backends for AutoModel and Megatron-Bridge. Use when adapter tuning is preferred over full fine-tuning because of memory, speed, checkpoint size, multi-domain adapters, or deployment constraints.
---

# Nemotron PEFT

Use this skill to choose a LoRA path and wire adapter outputs correctly.

## Route

| Backend | Best For | Input | Output |
| --- | --- | --- | --- |
| `peft/automodel` | HF models, smaller GPU counts, direct chat JSONL | `training_jsonl` | `checkpoint_lora` |
| `peft/megatron_bridge` | Distributed adapter tuning with Megatron checkpoints | `packed_parquet`, `checkpoint_megatron` | `checkpoint_lora` |

## Workflow

1. For Lepton, Slurm, Ray, or batch execution, verify the env profile file first. Default lookup uses repository-root `env.toml`; generated backend files such as `env.lepton.toml` or `env.slurm.toml` require `NEMOTRON_ENV_FILE`.
2. Prefer AutoModel PEFT for the simplest JSONL-to-adapter path.
3. Prefer Megatron-Bridge PEFT when TP/PP scaling or a Megatron base checkpoint is required.
4. Add `prep/sft_packing` before Megatron-Bridge PEFT.
5. Add `convert/merge_lora` when a standalone HF checkpoint is needed.
6. Check `src/nemotron/steps/patterns/small-dataset-lora.md` when deciding whether LoRA is appropriate.
7. Check `src/nemotron/steps/patterns/adapter-artifact-before-merge.md` before merging adapters for deployment.

## Guardrails

- Keep LoRA rank low for tight memory, then raise it for harder tasks.
- Treat adapters as separate artifacts until a merge step has been validated.
- Re-evaluate after merging adapters into a full checkpoint.
