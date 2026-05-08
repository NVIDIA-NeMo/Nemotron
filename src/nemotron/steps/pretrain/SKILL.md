---
name: nemotron-pretrain
description: Choose and configure Nemotron pretraining and continued-pretraining (CPT) backends — AutoModel and Megatron-Bridge. Use when planning pretraining from bin/idx data, choosing checkpoint format, selecting HF-native vs distributed Megatron execution, sizing the token budget, or scoping the data blend for sovereign CPT.
---

# Nemotron Pretrain

Pick a pretraining backend and lock the token budget + data blend before
requesting cluster time. Pretraining is the highest-cost stage in the
catalog — the budget contract decides everything downstream.

## Backends

| Backend | Best for | Default model / recipe | Input | Output |
|---|---|---|---|---|
| [`pretrain/automodel`](automodel/SKILL.md) | HF-native CPT, single-node, fast iteration | Qwen3-30B-A3B (MoE example) | `binidx` | `checkpoint_hf` |
| [`pretrain/megatron_bridge`](megatron_bridge/SKILL.md) | Large distributed pretraining/CPT, TP/PP/CP/EP scaling, Nemotron recipe parity | `nemotron_3_nano_pretrain_config` recipe (Nano3) | `binidx` (+ optional `checkpoint_megatron`) | `checkpoint_megatron` |

The "default model" column shows what the shipped `config/default.yaml`
selects. Override at CLI:

```bash
nemotron step run pretrain/automodel -c default \
  model.pretrained_model_name_or_path=<your-hf-id>
```

## Decision tree

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

1. **Compatible bin/idx data** from [`prep/pretrain_prep`](../prep/pretrain_prep/SKILL.md).
   `blend.json` is the trainer's entry — its tokenizer must match the model's.
2. **A documented token budget** (target_tokens, seq_length, gbs, train_iters,
   lr schedule, ckpt cadence). See [../patterns/pretrain-token-budget-before-scale.md](../patterns/pretrain-token-budget-before-scale.md).
3. **A held-out validation corpus** that never appears in training. For CPT,
   this includes both target-domain prompts (to measure shift) and general
   benchmarks the base model already passes (to detect forgetting).

## Pipeline placement

```
curate/nemo_curator → prep/pretrain_prep → pretrain/automodel        → checkpoint_hf
                                          → pretrain/megatron_bridge → checkpoint_megatron
                                                                       (then convert/megatron_to_hf if HF needed downstream)
```

## Workflow

1. **Env profile first** — verify the env profile for Lepton/Slurm/Ray runs
   (`env.toml` by default, or `NEMOTRON_ENV_FILE` for backend-specific files).
2. Run [`prep/pretrain_prep`](../prep/pretrain_prep/SKILL.md) on a tokenizer
   that matches the trainer.
3. Write the budget down (target_tokens / seq_length / gbs / train_iters /
   lr schedule / ckpt cadence) **before code changes**.
4. Pick backend per the decision tree.
5. Smoke with `config/tiny.yaml` to verify launch + data access + checkpoint
   write/restore.
6. Run a short *representative* job at production sequence length and
   parallelism to validate throughput and val-loss movement.
7. For CPT, evaluate at every checkpoint to catch forgetting early.
8. Bookend with eval — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md).
9. For sovereign deployments, judge against a Build-Your-Own-Benchmark — see
   [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).

## Smoke commands

```bash
nemotron step run pretrain/automodel       -c tiny
nemotron step run pretrain/megatron_bridge -c tiny
```

## Patterns to cite

- [../patterns/pretrain-token-budget-before-scale.md](../patterns/pretrain-token-budget-before-scale.md) — budget contract before scaling.
- [../patterns/cpt-data-blend-scoping.md](../patterns/cpt-data-blend-scoping.md) — CPT-specific blend ratios + forgetting discipline.
- [../patterns/prep-data-is-tokenizer-locked.md](../patterns/prep-data-is-tokenizer-locked.md) — bin/idx is tokenizer-locked.
- [../patterns/multilingual-tokenizer-check.md](../patterns/multilingual-tokenizer-check.md) — target-language tokenization affects token budget.
- [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md) — measure pretrain/CPT effects.
- [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md) — sovereign deployment evaluation.

## Local files

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
