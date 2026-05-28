# Nemotron Pretrain

Pick a pretraining backend and lock the token budget + data blend before
requesting cluster time. Pretraining is the highest-cost stage in the
catalog — the budget contract decides everything downstream.

## Developer Journey

Pretraining and continued pretraining are data-budget decisions before they are
training decisions. The corpus blend, tokenizer, held-out validation slices, and
target token budget should be written down before a backend is chosen.

1. Curate or assemble the source text blend.
2. Tokenize it with the exact model tokenizer through
   [`../data_prep/pretrain_prep/README.md`](../data_prep/pretrain_prep/README.md).
3. Preserve the emitted `blend.json`; both pretraining backends use it as the
   data entrypoint.
4. Choose AutoModel for HF-native iteration or Megatron-Bridge for large
   distributed runs.
5. Track validation loss and downstream evals across checkpoints, especially for
   CPT where forgetting is the main risk.

## Backends

| Backend | Best for | Default model / recipe | Input | Output |
|---|---|---|---|---|
| [`pretrain/automodel`](automodel/README.md) | HF-native CPT, single-node, fast iteration | Qwen3-30B-A3B (MoE example) | `binidx` | `checkpoint_hf` |
| [`pretrain/megatron_bridge`](megatron_bridge/README.md) | Large distributed pretraining/CPT, TP/PP/CP/EP scaling, Nemotron recipe parity | `nemotron_3_nano_pretrain_config` recipe (Nano3) | `binidx` (+ optional `checkpoint_megatron`) | `checkpoint_megatron` |

The "default model" column shows what the shipped `config/default.yaml`
selects. Override at CLI:

```bash
uv run nemotron steps run pretrain/automodel -c default --dry-run \
  model.pretrained_model_name_or_path=<your-hf-id>
```

## Decision Guide

- HF-format output, ≤1 node, fast iteration → **AutoModel**.
- Multi-node, TP/PP/CP/EP parallelism, or Nano3/Super3 recipe parity → **Megatron-Bridge**.
- Switching backends mid-run is a conversion problem in itself. Pick once,
  stay there. See [../patterns/convert-checkpoint-safety.md](../patterns/convert-checkpoint-safety.md).

## Pretrain vs continued pretraining (CPT)

Both backends support both. The differences are in **budget, learning rate,
and data blend** — not in the runner.

- **From scratch** (`load_weights=false`): warmup + cosine schedule sized to
  the full token budget, conventional pretrain learning rate.
- **CPT** (`load_weights=true`): learning rate **5–10× lower** than
  from-scratch (1e-5 to 5e-5 — see step.toml CPT strategy), and **the data
  blend matters more than the budget**.

For sovereign CPT — adapting a base model to a target language, jurisdiction,
or domain corpus — the blend ratios are the customization decision. See
[../patterns/cpt-data-blend-scoping.md](../patterns/cpt-data-blend-scoping.md)
for token-budget tiers (1–5B / 5–20B / 20–50B), forgetting checks, and
mandatory blend-with-general-data discipline.

## Pre-conditions

1. **Compatible bin/idx data** from [`data_prep/pretrain_prep`](../data_prep/pretrain_prep/README.md).
   `blend.json` is the trainer's entry — its tokenizer must match the model's.
2. **A documented token budget** (target_tokens, seq_length, gbs, train_iters,
   lr schedule, ckpt cadence). See [../patterns/pretrain-token-budget-before-scale.md](../patterns/pretrain-token-budget-before-scale.md).
3. **A held-out validation corpus** that never appears in training. For CPT,
   this includes both target-domain prompts (to measure shift) and general
   benchmarks the base model already passes (to detect forgetting).

## Data And Artifact Flow

```
curate/nemo_curator → data_prep/pretrain_prep → pretrain/automodel        → checkpoint_hf
                                              → pretrain/megatron_bridge → checkpoint_megatron
                                                                       (then convert/megatron_to_hf if HF needed downstream)
```

```text
raw / filtered text corpus
  -> blend file with train/validation/test intent
  -> data_prep/pretrain_prep
  -> binidx shards + blend.json
  -> pretrain/*
```

The `binidx` data is tokenizer-locked. If the tokenizer, model family, or
sequence-length assumptions change, rebuild the prep output instead of reusing
old shards. For CPT, keep target-domain validation and general-capability
validation separate so forgetting is visible.

## Workflow

1. Run [`data_prep/pretrain_prep`](../data_prep/pretrain_prep/README.md) on a tokenizer
   that matches the trainer.
2. Write the budget down (target_tokens / seq_length / gbs / train_iters /
   lr schedule / ckpt cadence) **before code changes**.
3. Pick backend per the decision tree.
4. Smoke with `config/tiny.yaml` to verify launch + data access + checkpoint
   write/restore.
5. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
6. Run a short *representative* job at production sequence length and
   parallelism to validate throughput and val-loss movement.
7. For CPT, evaluate at every checkpoint to catch forgetting early.
8. Bookend with eval — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).
9. For sovereign deployments, judge against a Build-Your-Own-Benchmark — see
   [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).

## Smoke commands

```bash
uv run nemotron steps run pretrain/automodel       -c tiny --dry-run
uv run nemotron steps run pretrain/megatron_bridge -c tiny --dry-run
```

## Project layout for generated configs

Keep every generated overlay config and any supporting code under a single
self-contained project root that also holds the local input data, so the
whole directory is rsync/scp-portable to the remote machine that will run
the pretrain step.

- `<project>/config/` for generated YAML — never write into
  `src/nemotron/steps/pretrain/<backend>/config/`; the shipped
  `default.yaml` and `tiny.yaml` stay as catalog references.
- `<project>/data/` for the bin/idx shards and the `blend.json` emitted by
  `data_prep/pretrain_prep`, plus any held-out validation slices.
- Keep checkpoint save dirs and budget/lr-schedule notes under the same
  project root so the run is reproducible after a remote transfer.
- Project-root scripts only when catalog code cannot serve the request.
- Do not split generated files into home dirs, scratch dirs, or paths
  outside the project root that will not ship with the bundle.

## Patterns to cite

- [../patterns/pretrain-token-budget-before-scale.md](../patterns/pretrain-token-budget-before-scale.md) — budget contract before scaling.
- [../patterns/cpt-data-blend-scoping.md](../patterns/cpt-data-blend-scoping.md) — CPT-specific blend ratios + forgetting discipline.
- [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md) — bin/idx is tokenizer-locked.
- [../patterns/multilingual-tokenizer-check.md](../patterns/multilingual-tokenizer-check.md) — target-language tokenization affects token budget.
- [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md) — measure pretrain/CPT effects.
- [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md) — sovereign deployment evaluation.

## Repository Layout

- `pretrain/automodel/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`
- `pretrain/megatron_bridge/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`
- Shared runners: [../_runners/automodel.py](../_runners/automodel.py), [../_runners/megatron_bridge.py](../_runners/megatron_bridge.py)

## Guardrails

- Don't reuse bin/idx data across tokenizer changes — rebuild prep.
- For CPT, never train without a forgetting baseline (general-capability
  validation slice).
- Don't switch backends mid-run; the checkpoints aren't interchangeable
  without explicit conversion.
- Plan the restart policy before launching a multi-day cluster job.
- Don't ship a sovereign deployment without a held-out target-language /
  target-domain benchmark.
