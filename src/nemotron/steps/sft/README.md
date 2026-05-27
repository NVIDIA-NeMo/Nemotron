---
name: nemotron-sft
description: Choose between Nemotron SFT backends, AutoModel and Megatron-Bridge, and wire required input data and checkpoint formats. Use when planning, configuring, validating, or debugging supervised fine-tuning stages.
---

# Nemotron SFT

Pick an SFT backend and keep data and checkpoint formats compatible.

## Backends

| Backend | Best for | Min GPUs | Input | Output |
|---|---|---|---|---|
| [`sft/automodel`](automodel/README.md) | HF-native outputs, direct JSONL, smaller GPU counts, quick LoRA experiments | 4 | `training_jsonl` (no packing) | `checkpoint_hf` |
| [`sft/megatron_bridge`](megatron_bridge/README.md) | Large distributed runs with TP/PP/CP, packed-sequence throughput, Nano3/Super3 recipe parity | 8 (Nano3), 32 (Super3) | `packed_parquet` (needs `data_prep/sft_packing`) | `checkpoint_megatron` |

## Decision tree

- Need TP/PP/CP parallelism or official Nano3/Super3 recipe patterns? → **Megatron-Bridge**.
- Fewer than 8 GPUs? → **AutoModel**.
- Want LoRA with minimal setup? → **AutoModel** (or [`peft/automodel`](../peft/automodel/README.md)).
- Need the highest-throughput multi-node path? → **Megatron-Bridge**.
- Just want SFT running fast on existing JSONL? → **AutoModel**.

## Pipeline impact

**If Megatron-Bridge:**
- Add [`data_prep/sft_packing`](../data_prep/sft_packing/README.md) upstream.
- Output is `checkpoint_megatron`. For HF-format consumers downstream, add
  [`convert/megatron_to_hf`](../convert/megatron_to_hf/step.toml).

**If AutoModel:**
- No packing step. Reads `training_jsonl` directly.
- Output is `checkpoint_hf`.
- LoRA / PEFT is the default starting point for small GPU counts.

## Workflow

1. Read the chosen step's `step.toml` for parameters/strategies/errors.
2. Smoke-test with `config/tiny.yaml` before scaling.
3. Keep `pack_size`, `seq_length`, tokenizer, and chat template identical
   across prep, train, eval, and deployment — see
   [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md).
4. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
5. Inspect formatted prompts and loss masks before treating a run as meaningful.
6. Bookend with eval — see [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).

## Smoke commands

```bash
uv run nemotron steps run sft/automodel -c tiny --dry-run
uv run nemotron steps run sft/megatron_bridge -c tiny --dry-run   # requires compatible packed_parquet
```

## Project layout for generated configs

Keep every generated overlay config and any supporting code under a single
self-contained project root that also holds the local input data, so the
whole directory is rsync/scp-portable to the remote machine that will run
the SFT step.

- `<project>/config/` for generated YAML — never write into
  `src/nemotron/steps/sft/<backend>/config/`; the shipped `default.yaml`
  and `tiny.yaml` stay as catalog references.
- `<project>/data/` for local datasets, chat-format JSONL, and packed
  Parquet splits referenced by the overlay.
- Project-root scripts only when catalog code cannot serve the request.
- Do not split generated files into home dirs, scratch dirs, or paths
  outside the project root that will not ship with the bundle.

## Patterns to cite

- [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md) — tokenizer/template/seq_length must match prep.
- [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md) — capability-aware blending (sovereign / multilingual / mixed-source SFT corpora).
- [../patterns/sft-small-dataset-prefer-lora.md](../patterns/sft-small-dataset-prefer-lora.md) — when full SFT vs PEFT/LoRA.
- [../patterns/sft-sequence-packing.md](../patterns/sft-sequence-packing.md) — packing efficiency for variable-length data.
- [../patterns/multilingual-tokenizer-check.md](../patterns/multilingual-tokenizer-check.md) — non-English data needs tokenization audit.
- [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md) — bookend every SFT run.
- [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md) — sovereign deployments need a held-out target-domain benchmark.

## Guardrails

- **Backend choice is sticky.** Switching mid-pipeline is a conversion
  problem (`convert/megatron_to_hf` / `convert/hf_to_megatron`).
- For Megatron-Bridge, `pack_size` (in prep) must equal `seq_length` (in
  training). Mismatch surfaces as shape errors mid-train.
- For AutoModel, never add `data_prep/sft_packing` — the trainer reads JSONL
  directly.
- Inspect formatted prompts and loss masks before trusting loss curves —
  template bugs look like quality bugs.
- Re-eval after any data-blend change, not just after hyperparameter changes.
