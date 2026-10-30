# Nemotron Data Prep

Pick a data_prep step, lock its outputs to a tokenizer, and keep the prepared
artifact compatible with the downstream trainer. Prepared data is a
**versioned data product** — name it after the (tokenizer, template, pack_size)
tuple, not after the date.

## Developer Journey

Data prep is where raw or semi-structured training data becomes a trainer-ready
artifact. Start here when you need to decide whether data should stay as JSONL,
be packed into Parquet, become Megatron bin/idx shards, or be materialized into
RL prompt/preference splits.

1. Identify the downstream trainer first. The trainer decides the output format.
2. Inspect the source data schema on a small sample before running prep.
3. Choose the prep step that produces the artifact the trainer consumes.
4. Run the tiny config or a small project overlay, then inspect records and
   metadata before launching a full prep job.
5. Treat the output as immutable once training starts. Rebuild it after tokenizer,
   chat-template, sequence-length, split, or blend changes.

## Steps

| Need | Step | Produces |
|---|---|---|
| Pack chat JSONL for Megatron-Bridge SFT or PEFT | [`data_prep/sft_packing`](sft_packing/README.md) | `packed_parquet` |
| Tokenize text into Megatron pretraining shards | [`data_prep/pretrain_prep`](pretrain_prep/README.md) | `binidx` + `blend.json` |
| Resolve and shard RL prompt or preference data | [`data_prep/rl_prep`](rl_prep/README.md) | `training_jsonl` (sharded) |

## When to use `data_prep/sft_packing`

| Downstream trainer | Packing required? | Why |
|---|---|---|
| `sft/megatron_bridge`, `peft/megatron_bridge` | **Yes** | Megatron-Bridge expects `packed_parquet`, not raw JSONL. |
| `sft/automodel`, `peft/automodel` | **No** | These read `training_jsonl` directly. |

Skip packing when:
- The trainer reads chat-format JSONL directly.
- You're still iterating on data shape and don't want to commit to a
  tokenizer / template / pack-size yet.

## Data And Artifact Flow

```text
raw / curated text blend
  -> data_prep/pretrain_prep
  -> binidx + blend.json
  -> pretrain/*
```

```text
chat training_jsonl
  -> data_prep/sft_packing
  -> packed_parquet
  -> sft/megatron_bridge or peft/megatron_bridge
```

```text
prompt / preference blend
  -> data_prep/rl_prep
  -> sharded training_jsonl
  -> rl/nemo_rl/{dpo,rlvr,rlhf}
```

AutoModel SFT and PEFT skip `data_prep/sft_packing` and consume chat
`training_jsonl` directly. Megatron-Bridge paths usually require prepared data
because sequence packing and bin/idx layout are part of the training contract.

## Workflow

1. Read the target step's `step.toml` for artifacts, parameters, strategies,
   and references.
2. Start with `config/tiny.yaml` for smoke tests, `config/default.yaml` for
   production shape.
3. Keep tokenizer, chat template, sequence length, split names, and shard
   policy aligned with the downstream trainer.
4. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
5. Inspect sample outputs before launching expensive training.

## Smoke commands

```bash
uv run nemotron steps run data_prep/sft_packing   -c tiny --dry-run
uv run nemotron steps run data_prep/pretrain_prep -c tiny --dry-run
uv run nemotron steps run data_prep/rl_prep       -c tiny --dry-run
```

## Project layout for generated configs

Keep every generated overlay config and any supporting code under a single
self-contained project root that also holds the local input data, so the
whole directory is rsync/scp-portable to the remote machine that will run
the data_prep step.

- `<project>/config/` for generated YAML — never write into
  `src/nemotron/steps/data_prep/<step>/config/`; the shipped `default.yaml`
  and `tiny.yaml` stay as catalog references.
- `<project>/data/` for source blends (`blend.json`), local JSONL inputs,
  and the prepared artifact destination (packed Parquet shards, bin/idx +
  emitted `blend.json`, or RL JSONL splits).
- Tokenizer, chat-template, and `pack_size` / `seq_length` metadata should
  be captured under the same project root so downstream training can be
  shipped together as one portable bundle.
- Project-root scripts only when catalog code cannot serve the request.
- Do not split generated files into home dirs, scratch dirs, or paths
  outside the project root that will not ship with the bundle.

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
