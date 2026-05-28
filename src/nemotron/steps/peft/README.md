---
name: nemotron-peft
description: Choose and configure Nemotron PEFT/LoRA backends for AutoModel and Megatron-Bridge. Use when adapter tuning is preferred over full fine-tuning because of memory, speed, checkpoint size, multi-domain adapters, or sovereign deployment constraints requiring small swappable artifacts.
---

# Nemotron PEFT

Pick a LoRA path and wire adapter outputs correctly. PEFT is the default for
sovereign customizations where the base model is strong enough and the goal is
a narrow capability delta (target-language style, domain terminology,
instruction-format adherence).

## Backends

| Backend | Best for | Min GPUs | Input | Output |
|---|---|---|---|---|
| [`peft/automodel`](automodel/README.md) | HF-native, single-node, direct chat JSONL, fast iteration | 1–4 | `training_jsonl` | `checkpoint_lora` (HF adapter) |
| [`peft/megatron_bridge`](megatron_bridge/README.md) | Distributed adapter tuning over a Megatron base, packed-Parquet throughput | 8+ | `packed_parquet` + `checkpoint_megatron` | `checkpoint_lora` (Megatron-format adapter) |

## Decision tree

- 1–4 GPUs, HF base, JSONL data → **AutoModel**.
- 8+ GPUs, Megatron base or packed Parquet → **Megatron-Bridge**.
- Need a deployable HF checkpoint after training → either path; chain
  [`convert/merge_lora`](../convert/merge_lora/step.toml) (Megatron-format
  adapters need merging into the same base they were trained against).
- Adapter output format must be HF for downstream tools → AutoModel directly,
  or Megatron-Bridge then `convert/megatron_to_hf` then `convert/merge_lora`.

## Pipeline impact

**If AutoModel:**
- No data_prep step. Reads `training_jsonl` directly.
- LoRA defaults: `peft.dim=8` or `16`, `peft.alpha ≈ 2 * peft.dim`.
- Output is an HF-format adapter merged via `convert/merge_lora`.

**If Megatron-Bridge:**
- Add [`data_prep/sft_packing`](../data_prep/sft_packing/README.md) upstream.
- Requires a base `checkpoint_megatron` at `checkpoint.pretrained_checkpoint`.
- Output is a Megatron-format adapter.

## Workflow

1. Pick backend per the decision tree above.
2. Read the chosen step's `step.toml` for parameters/strategies/errors.
3. Smoke-test with `config/tiny.yaml` before scaling.
4. Keep base model + tokenizer + chat template identical to any later
   `sft/*` or `eval/*` consumer — see
   [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md).
5. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
6. Treat the adapter as a **separate artifact** until merge — see
   [../patterns/peft-adapter-merge-discipline.md](../patterns/peft-adapter-merge-discipline.md).
7. Decide whether LoRA is even the right tool — see
   [../patterns/sft-small-dataset-prefer-lora.md](../patterns/sft-small-dataset-prefer-lora.md).
8. For sovereign / domain SFT blends, also consult
   [../patterns/sft-data-blending.md](../patterns/sft-data-blending.md).
9. Bookend with eval — see
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
- Never compare adapter-loaded scores against merged scores assuming
  identity — they can drift.
- For multi-domain adapters trained off the same base, version each adapter
  with the (base, blend, rank, alpha, target_modules) tuple.
- Re-eval after merge; don't trust adapter eval as a proxy for merged quality.
