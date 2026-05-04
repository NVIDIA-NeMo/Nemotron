---
name: nemotron-pretrain
description: Choose and configure Nemotron pretraining and continued-pretraining backends, including AutoModel and Megatron-Bridge. Use when planning pretraining from bin/idx data, choosing checkpoint format, selecting HF-native versus distributed Megatron execution, or validating pretrain configs.
---

# Nemotron Pretrain

Use this skill to choose between AutoModel and Megatron-Bridge pretraining.

## Route

| Backend | Best For | Input | Output |
| --- | --- | --- | --- |
| `pretrain/automodel` | HF-native CPT, smaller GPU counts, fast iteration | `binidx` | `checkpoint_hf` |
| `pretrain/megatron_bridge` | Large distributed training, TP/PP/CP/EP scaling | `binidx`, optional `checkpoint_megatron` | `checkpoint_megatron` |

## Workflow

1. For Lepton, Slurm, Ray, or batch execution, verify the env profile file first. Default lookup uses repository-root `env.toml`; generated backend files such as `env.lepton.toml` or `env.slurm.toml` require `NEMOTRON_ENV_FILE`.
2. Produce compatible bin/idx data with `prep/pretrain_prep`.
3. Choose AutoModel when staying HF-native matters more than Megatron parallelism.
4. Choose Megatron-Bridge when model size, sequence length, or throughput requires distributed parallelism.
5. Start from `config/tiny.yaml` to validate data access, launch, and checkpoint writes.
6. Check `src/nemotron/steps/patterns/pretrain-token-budget-before-scale.md` before moving beyond smoke tests.
7. Check `src/nemotron/steps/patterns/prepared-data-is-tokenizer-locked.md` before reusing bin/idx data.

## Local Files

- `pretrain/automodel/step.toml`, `pretrain/automodel/step.py`, `pretrain/automodel/config/default.yaml`, `pretrain/automodel/config/tiny.yaml`
- `pretrain/megatron_bridge/step.toml`, `pretrain/megatron_bridge/step.py`, `pretrain/megatron_bridge/config/default.yaml`, `pretrain/megatron_bridge/config/tiny.yaml`
- `src/nemotron/steps/_runners/automodel.py`
- `src/nemotron/steps/_runners/megatron_bridge.py`

## Guardrails

- Do not reuse bin/idx data across tokenizer changes.
- Set the token budget, LR schedule, checkpoint cadence, and validation corpus before scaling.
- Keep output checkpoint format aligned with the next pipeline stage.
