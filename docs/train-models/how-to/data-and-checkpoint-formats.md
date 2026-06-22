# Data and Checkpoint Formats

Training steps declare compatible artifacts in each `step.toml` file. This page summarizes the types that most training steps use so you can chain steps without format surprises.

## Canonical Definitions

Artifact names and compatibility rules live in `src/nemotron/steps/types.toml` at the repository root of the step library. Treat that file as the source of truth for names such as `training_jsonl`, `packed_parquet`, and `checkpoint_megatron`.

## Frequently Used Types

- The `training_jsonl` type means JSON Lines (JSONL) with a `messages` field in OpenAI chat shape. AutoModel supervised fine tuning (SFT) and parameter-efficient fine tuning (PEFT) consume it, and several reinforcement learning (RL) data paths consume it.
- The `packed_parquet` type means packed shards with token columns and masks for Megatron Bridge style trainers. You get this type only after you run a packing prep step when you use the Parquet path.
- The `checkpoint_hf` type means Hugging Face layout checkpoints or full weights on disk.
- The `checkpoint_megatron` type means Megatron distributed checkpoints sharded across parallel ranks.
- The `checkpoint_lora` type means low-rank adaptation (LoRA) adapter weights. Many downstream tools still need merge or export before deployment.

## Chaining Guidance

1. AutoModel SFT never consumes `packed_parquet`. Megatron Bridge SFT does not consume raw JSON Lines (JSONL) for the packed pipeline.
2. RL steps in this repository expect a Megatron policy checkpoint for warm start. If your SFT used AutoModel, insert the appropriate conversion step before RL.
3. Optimization steps that start from `checkpoint_hf` require a merged base when the trainable artifact was LoRA.

## Where to Look in the Tree

Each step directory contains the following files:

- `step.toml` holds the identifier, human title, tags, `[[consumes]]`, `[[produces]]`, `[[parameters]]`, optional `[[strategies]]`, `[[errors]]`, and `[[models]]` blocks.
- `config/default.yaml` holds primary configuration tuned for real workloads.
- `config/tiny.yaml` holds reduced settings for short sample runs and end-to-end execution validation.
- Extra files such as `config/nemo_gym.yaml` appear only on steps that need alternate method profiles.

## Related Reading

- [Artifact Graph](../explanation/artifact-graph.md)
- [Step Catalog (Training)](../reference/step-catalog.md)
