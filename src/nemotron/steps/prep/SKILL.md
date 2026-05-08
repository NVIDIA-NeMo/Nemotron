---
name: nemotron-prep
description: Navigate Nemotron data preparation steps for SFT packing, pretraining bin/idx tokenization, and RL prompt or preference sharding. Use when choosing, configuring, validating, or chaining prep steps before pretrain, CPT, SFT, PEFT, DPO, RLVR, or RLHF training, including sovereign customizations where blend composition decides downstream behavior.
---

# Nemotron Prep

Pick a prep step, lock its outputs to a tokenizer, and keep the prepared
artifact compatible with the downstream trainer. Prepared data is a
**versioned data product** — name it after the (tokenizer, template, pack_size)
tuple, not after the date.

## Steps

| Need | Step | Produces |
|---|---|---|
| Pack chat JSONL for Megatron-Bridge SFT or PEFT | [`prep/sft_packing`](sft_packing/SKILL.md) | `packed_parquet` |
| Tokenize text into Megatron pretraining shards | [`prep/pretrain_prep`](pretrain_prep/SKILL.md) | `binidx` + `blend.json` |
| Resolve and shard RL prompt or preference data | [`prep/rl_prep`](rl_prep/SKILL.md) | `training_jsonl` (sharded) |

## When to use `prep/sft_packing`

| Downstream trainer | Packing required? | Why |
|---|---|---|
| `sft/megatron_bridge`, `peft/megatron_bridge` | **Yes** | Megatron-Bridge expects `packed_parquet`, not raw JSONL. |
| `sft/automodel`, `peft/automodel` | **No** | These read `training_jsonl` directly. |

Skip packing when:
- The trainer reads chat-format JSONL directly.
- You're still iterating on data shape and don't want to commit to a
  tokenizer / template / pack-size yet.

## Workflow

1. **Env profile first** — verify the env profile for Lepton/Slurm/Ray/batch
   runs (`env.toml` by default, or `NEMOTRON_ENV_FILE` for backend-specific
   files).
2. Read the target step's `step.toml` for artifacts, parameters, strategies,
   and references.
3. Start with `config/tiny.yaml` for smoke tests, `config/default.yaml` for
   production shape.
4. Keep tokenizer, chat template, sequence length, split names, and shard
   policy aligned with the downstream trainer.
5. Inspect sample outputs before launching expensive training.

## Smoke commands

```bash
nemotron step run prep/sft_packing   -c tiny
nemotron step run prep/pretrain_prep -c tiny
nemotron step run prep/rl_prep       -c tiny
```

## Patterns to cite

- [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md) — repack on tokenizer / template / seq_length changes.
- [../patterns/sft-sequence-packing.md](../patterns/sft-sequence-packing.md) — when packing helps vs hurts.
- [../patterns/data-quality-before-quantity.md](../patterns/data-quality-before-quantity.md) — quality gates per source before blending.
- [../patterns/multilingual-tokenizer-check.md](../patterns/multilingual-tokenizer-check.md) — non-English data needs a tokenization audit before prep.
- [../patterns/cpt-data-blend-scoping.md](../patterns/cpt-data-blend-scoping.md) — pretrain prep for CPT must be scoped to blend ratios and base-model size.
- [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md) — SFT packing inherits whatever blend went into the JSONL; reshuffle = repack.
- [../patterns/sdg-pipeline-versioning.md](../patterns/sdg-pipeline-versioning.md) — when synthetic data feeds prep.
- [../patterns/rl-validate-rewards-before-scale.md](../patterns/rl-validate-rewards-before-scale.md) — RL prep produces training_jsonl that the RL step trusts; validate before scaling.

## Guardrails

- **Repack** SFT data whenever tokenizer, chat template, or `pack_size` changes.
- **Rebuild** pretraining bin/idx whenever the tokenizer changes.
- **Materialize** remote data during RL prep when the training cluster
  cannot reach the source (`resolve_hf_placeholders=true`).
- Treat prepared artifacts as reproducible, versioned data products — name
  them after the (tokenizer, template, pack_size) tuple, not after the date.
- For sovereign / multilingual prep, run a tokenizer-coverage audit on a
  sample before committing to a tokenizer choice.
