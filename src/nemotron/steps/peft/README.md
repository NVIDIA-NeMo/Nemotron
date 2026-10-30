# Nemotron PEFT

PEFT is the adapter-tuning path for cases where the base model is already close
and the target change is narrow: domain vocabulary, target-language style,
instruction formatting, tool-call behavior, or another capability that should
not require rewriting the full checkpoint.

Start here when you need to decide whether LoRA is the right tool, which backend
to run, what data artifact to prepare, and what happens to the adapter after
training.

## Developer Journey

1. Confirm PEFT is appropriate. If the data is tiny, the GPU budget is tight, or
   you need swappable domain adapters, PEFT is usually the first pass. If the
   model needs broad behavior change and you can afford it, compare against
   full SFT in [`../sft/README.md`](../sft/README.md).
2. Choose the backend from the table below.
3. Prepare the artifact the backend expects:
   - AutoModel reads chat `training_jsonl` directly.
   - Megatron-Bridge needs `packed_parquet` from
     [`../data_prep/sft_packing/README.md`](../data_prep/sft_packing/README.md)
     plus a base `checkpoint_megatron`.
4. Run the backend smoke command before writing a production overlay.
5. Keep adapter provenance with the output: base checkpoint, tokenizer, chat
   template, LoRA rank, alpha, target modules, data blend, and training config.
6. Decide the downstream branch:
   - Evaluate with the adapter loaded when the serving path supports adapters.
   - Merge with [`../convert/merge_lora/README.md`](../convert/merge_lora/README.md)
     when deployment or downstream evaluation expects a standalone HF checkpoint.
   - Convert only when the next consumer requires a different checkpoint layout.

## Backends

| Backend | Best for | Min GPUs | Input | Output |
|---|---|---|---|---|
| [`peft/automodel`](automodel/README.md) | HF-native, single-node, direct chat JSONL, fast iteration | 1–4 | `training_jsonl` | `checkpoint_lora` (HF adapter) |
| [`peft/megatron_bridge`](megatron_bridge/README.md) | Distributed adapter tuning over a Megatron base, packed-Parquet throughput | 8+ | `packed_parquet` + `checkpoint_megatron` | `checkpoint_lora` (Megatron-format adapter) |

## Decision Guide

- 1–4 GPUs, HF base, JSONL data → **AutoModel**.
- 8+ GPUs, Megatron base or packed Parquet → **Megatron-Bridge**.
- Need a deployable HF checkpoint after training → plan a merge from the start.
  The adapter must be merged into the exact base it was trained against.
- Adapter output must be HF-native for downstream tools → prefer AutoModel, or
  budget the Megatron-Bridge export/merge path before training.
- Need full checkpoint weights, not an adapter → use [`../sft/README.md`](../sft/README.md)
  instead of PEFT.

## Data And Artifact Flow

**If AutoModel:**

```text
training_jsonl + checkpoint_hf/base model
  -> peft/automodel
  -> checkpoint_lora (HF adapter)
  -> convert/merge_lora when a standalone checkpoint_hf is required
```

No packing step is needed. This is the fastest path for JSONL data and
single-node iteration.

**If Megatron-Bridge:**

```text
training_jsonl
  -> data_prep/sft_packing
  -> packed_parquet + checkpoint_megatron
  -> peft/megatron_bridge
  -> checkpoint_lora (Megatron adapter)
  -> convert/merge_lora / export path when HF deployment is required
```

Use this path when the base model and training topology are already in
Megatron-Bridge or when packed-sequence throughput matters.

## Key Knobs To Decide Early

- `peft.dim`: start low (`8` or `16`) when memory is tight; raise only when the
  task needs more capacity.
- `peft.alpha`: usually keep near `2 * peft.dim` unless you are deliberately
  tuning adapter scale.
- Base checkpoint: record the exact base path or model ID because merge needs
  the same base.
- Tokenizer and chat template: keep them identical across data prep, adapter
  training, merge, and eval.
- Output paths: keep base, adapter, merged checkpoint, and eval outputs in
  separate directories.

## Workflow

1. Pick backend per the decision guide above.
2. Read the chosen backend README and `step.toml`.
3. Run the smoke command for the backend.
4. Create a project-local overlay config; do not edit checked-in `default.yaml`
   or `tiny.yaml`.
5. Keep base model + tokenizer + chat template identical to any later
   `sft/*` or `eval/*` consumer — see
   [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md).
6. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
7. Treat the adapter as a **separate artifact** until merge — see
   [../patterns/peft-adapter-merge-discipline.md](../patterns/peft-adapter-merge-discipline.md).
8. Decide whether LoRA is still the right tool after the first eval — see
   [../patterns/sft-small-dataset-prefer-lora.md](../patterns/sft-small-dataset-prefer-lora.md).
9. For sovereign / domain SFT blends, also consult
   [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md).
10. Bookend with eval — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).

## Smoke commands

```bash
uv run nemotron steps run peft/automodel -c tiny --dry-run
uv run nemotron steps run peft/megatron_bridge -c tiny --dry-run   # requires compatible packed_parquet + base checkpoint
```

## Project layout for generated configs

Keep every generated overlay config and any supporting code under a single
self-contained project root that also holds the local input data, so the
whole directory is rsync/scp-portable to the remote machine that will run
the PEFT step.

- `<project>/config/` for generated YAML — never write into
  `src/nemotron/steps/peft/<backend>/config/`; the shipped `default.yaml`
  and `tiny.yaml` stay as catalog references.
- `<project>/data/` for local datasets, chat-format JSONL, and packed
  Parquet splits referenced by the overlay.
- Adapter output paths (`checkpoint_lora`) should resolve under the same
  project root so the trained adapter ships with its provenance.
- Project-root scripts only when catalog code cannot serve the request.
- Do not split generated files into home dirs, scratch dirs, or paths
  outside the project root that will not ship with the bundle.

## Guardrails

- Keep LoRA rank low for tight memory; raise it only for harder tasks.
- Do not treat a successful adapter smoke run as evidence of quality.
- Never compare adapter-loaded scores against merged scores assuming
  identity — they can drift.
- For multi-domain adapters trained off the same base, version each adapter
  with the (base, blend, rank, alpha, target_modules) tuple.
- Re-eval after merge; don't trust adapter eval as a proxy for merged quality.
