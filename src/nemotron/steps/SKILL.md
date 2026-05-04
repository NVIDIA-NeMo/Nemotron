---
name: nemotron-steps
description: Navigate the Nemotron step library across prep, pretrain, SFT, PEFT, RL, synthetic data generation, and optimization. Use when planning end-to-end pipelines, choosing a backend, checking artifact compatibility, or finding the correct step SKILL.md, step.toml, runner, config, and upstream reference repo.
---

# Nemotron Steps

Use this skill as the entry point for the Nemotron training and optimization step library under `src/nemotron/steps/`.

## Route

| Need | Start With | Primary Artifacts |
| --- | --- | --- |
| SFT packing, pretrain bin/idx, RL sharding | `prep/SKILL.md` | `training_jsonl`, `packed_parquet`, `binidx` |
| Pretraining or continued pretraining | `pretrain/SKILL.md` | `binidx`, `checkpoint_hf`, `checkpoint_megatron` |
| Supervised fine-tuning | `sft/SKILL.md` | `training_jsonl`, `packed_parquet`, checkpoints |
| LoRA or adapter tuning | `peft/SKILL.md` | `checkpoint_lora` |
| DPO, RLVR, or RLHF alignment | `rl/SKILL.md` | prompt or preference JSONL, Megatron checkpoints |
| SFT SDG or RL preference SDG | `synth/SKILL.md` | `synthetic_jsonl` |
| Quantization, distillation, pruning | `optimize/SKILL.md` | optimized HF or Megatron checkpoints |
| Execution profiles and Lepton/Ray env setup | `env/SKILL.md` | `env_toml` |

## Workflow

1. For any Lepton, Slurm, Ray, or other non-local run, create or verify the env profile file first with `env/SKILL.md`. The default lookup is repository-root `env.toml`; generated backend examples use `env.lepton.toml` or `env.slurm.toml` and must be selected with `NEMOTRON_ENV_FILE`.
2. Read the most specific `SKILL.md` for the requested stage.
3. Read that step's `step.toml` first to understand the flow: intent, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references. Treat it as the agent-facing contract before editing configs or step code.
4. Start from `config/tiny.yaml` for runner validation and `config/default.yaml` for production shape.
5. Keep artifact formats explicit when chaining steps. Convert only when the next consumer requires a different checkpoint layout.
6. Validate the smallest realistic path before scaling to cluster resources.

## Decision Patterns

- Use `patterns/prepared-data-is-tokenizer-locked.md` for packed Parquet, bin/idx, tokenizer, chat-template, and sequence-length compatibility.
- Use `patterns/pretrain-token-budget-before-scale.md` before scaling pretraining or continued pretraining.
- Use `patterns/small-dataset-lora.md` and `patterns/adapter-artifact-before-merge.md` for PEFT and LoRA decisions.
- Use `patterns/validate-rl-rewards-before-scale.md` before scaling DPO, RLVR, or RLHF jobs.
- Use `patterns/version-sdg-pipeline.md` for SFT SDG and RL preference SDG.
- Use `patterns/representative-calibration-before-optimization.md` and `patterns/distill-after-structural-compression.md` for ModelOpt decisions.
- Use each step's `step.toml [reference]` section for upstream repos and documentation.

## Guardrails

- Treat data, checkpoints, configs, and eval results as versioned artifacts.
- Keep tokenizer, chat template, sequence length, checkpoint format, and split names aligned across stages.
- Keep env profile files at the repository root for profile lookup, and never blindly regenerate over a user's existing `env.toml`, `env.lepton.toml`, or `env.slurm.toml`.
- Do not use tiny configs as quality evidence; they only prove that plumbing starts.
- Prefer existing runners and configs over inventing a new wrapper.
