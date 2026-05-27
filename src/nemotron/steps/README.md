---
name: nemotron-steps
description: Navigate the Nemotron step library across curation, data_prep, translation, pretrain, SFT, PEFT, RL, synthetic data generation, BYOB benchmarks, conversion, optimization, evaluation, and env setup. Use when planning end-to-end pipelines, choosing a backend, checking artifact compatibility, or finding the correct step README.md, step.toml, runner, config, and upstream reference repo.
---

# Nemotron Steps

Use this skill as the entry point for the Nemotron training and optimization step library under `src/nemotron/steps/`.

## Route

| Need | Start With | Primary Artifacts |
| --- | --- | --- |
| Lightweight text curation | `curate/nemo_curator/README.md` | `raw_jsonl`, `filtered_jsonl` |
| SFT packing, pretrain bin/idx, RL sharding | `data_prep/README.md` | `training_jsonl`, `packed_parquet`, `binidx` |
| Corpus translation and FAITH scoring | `translate/README.md` | `filtered_jsonl`, `translated_jsonl` |
| Pretraining or continued pretraining | `pretrain/README.md` | `binidx`, `checkpoint_hf`, `checkpoint_megatron` |
| Supervised fine-tuning | `sft/README.md` | `training_jsonl`, `packed_parquet`, checkpoints |
| LoRA or adapter tuning | `peft/README.md` | `checkpoint_lora` |
| DPO, RLVR, or RLHF alignment | `rl/README.md` | prompt or preference JSONL, Megatron checkpoints |
| SFT SDG or RL preference SDG | `sdg/README.md` | `synthetic_jsonl` |
| BYOB benchmark generation or translation | `byob/README.md` | benchmark parquet artifacts |
| Checkpoint format conversion or LoRA merge | `convert/README.md` | `checkpoint_hf`, `checkpoint_megatron`, `checkpoint_lora` |
| Quantization, distillation, pruning | `optimize/README.md` | optimized HF or Megatron checkpoints |
| Evaluation | `eval/model_eval/README.md` | `eval_results` |
| Execution profiles and Lepton/Ray env setup | `env/README.md` | `env_toml` |

## Workflow

1. For any Lepton, Slurm, DGX Cloud, Ray, or other non-local run, create or verify the env profile file first with `env/README.md`. The default lookup is repository-root `env.toml`; generated backend examples use `env.lepton.toml`, `env.slurm.toml`, or `env.dgxcloud.toml` and must be selected with `NEMOTRON_ENV_FILE`.
2. Read the most specific `README.md` for the requested stage.
3. Read that step's `step.toml` first to understand the flow: intent, consumed and produced artifacts, important parameters, strategies, failure modes, and upstream references. Treat it as the agent-facing contract before editing configs or step code.
4. Start from `config/tiny.yaml` for runner validation and `config/default.yaml` for production shape.
5. Keep artifact formats explicit when chaining steps. Convert only when the next consumer requires a different checkpoint layout.
6. Validate the smallest realistic path before scaling to cluster resources.

## Decision Patterns

- Use `patterns/prep-data-is-tokenizer-locked.md` for packed Parquet, bin/idx, tokenizer, chat-template, and sequence-length compatibility.
- Use `patterns/pretrain-token-budget-before-scale.md` before scaling pretraining or continued pretraining.
- Use `patterns/sft-small-dataset-prefer-lora.md` and `patterns/peft-adapter-merge-discipline.md` for PEFT and LoRA decisions.
- Use `patterns/rl-validate-rewards-before-scale.md` before scaling DPO, RLVR, or RLHF jobs.
- Use `patterns/sdg-pipeline-versioning.md` for SFT SDG and RL preference SDG.
- Use `patterns/eval-before-and-after-training.md` and `patterns/byob-benchmark-design.md` for ModelOpt quality decisions.
- Use each step's `step.toml [reference]` section for upstream repos and documentation.

## Guardrails

- Treat data, checkpoints, configs, and eval results as versioned artifacts.
- Keep tokenizer, chat template, sequence length, checkpoint format, and split names aligned across stages.
- Keep env profile files at the repository root for profile lookup, and never blindly regenerate over a user's existing `env.toml`, `env.lepton.toml`, or `env.slurm.toml`.
- Do not use tiny configs as quality evidence; they only prove that plumbing starts.
- Prefer existing runners and configs over inventing a new wrapper.
