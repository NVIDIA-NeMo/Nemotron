---
name: nemotron-sft
description: Choose between Nemotron SFT backends, AutoModel and Megatron-Bridge, and wire required input data and checkpoint formats. Use when planning, configuring, validating, or debugging supervised fine-tuning stages.
---

# Nemotron SFT

Use this skill to choose an SFT backend and keep data and checkpoint formats compatible.

## Route

| Backend | Best For | Input | Output |
| --- | --- | --- | --- |
| `sft/automodel` | HF-native outputs, direct JSONL, smaller GPU counts, quick LoRA-style experiments | `training_jsonl` | `checkpoint_hf` |
| `sft/megatron_bridge` | Large distributed runs with TP/PP/CP and packed-sequence throughput | `packed_parquet` | `checkpoint_megatron` |

## Workflow

1. For Lepton, Slurm, Ray, or batch execution, verify the env profile file first. Default lookup uses repository-root `env.toml`; generated backend files such as `env.lepton.toml` or `env.slurm.toml` require `NEMOTRON_ENV_FILE`.
2. Use AutoModel when the dataset is already chat JSONL and the target artifact is Hugging Face compatible.
3. Use Megatron-Bridge when packed data, distributed parallelism, or Nemotron recipe parity matters.
4. Add `prep/sft_packing` before Megatron-Bridge SFT.
5. Keep `pack_size`, `seq_length`, tokenizer, and chat template identical across prep, train, eval, and deployment.
6. Check `src/nemotron/steps/patterns/prepared-data-is-tokenizer-locked.md` before Megatron-Bridge SFT.
7. Check `src/nemotron/steps/patterns/eval-bookends.md` before comparing SFT results.

## Validation

- Smoke AutoModel with `nemotron step run sft/automodel -c tiny`.
- Smoke Megatron-Bridge with `nemotron step run sft/megatron_bridge -c tiny` after compatible packed data exists.
- Inspect formatted prompts and loss masks before treating a run as meaningful.
