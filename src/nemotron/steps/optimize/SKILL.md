---
name: nemotron-optimizer
description: "Choose and configure Nemotron optimization steps using NVIDIA ModelOpt and Megatron-Bridge: quantization, distillation, and pruning. Use when reducing inference cost, compressing checkpoints, recovering quality, targeting FP8 or NVFP4 hardware, or preparing optimized Megatron or HuggingFace outputs."
---

# Nemotron Optimizer

Pick a ModelOpt path and chain converters where checkpoint formats cross.
Optimization is a **post-customization** stage — apply it after SFT/RL is
proven, not before.

## Steps

| Need | Step | Input | Output |
|---|---|---|---|
| FP8 (Hopper/H100) or NVFP4 (Blackwell/B200) post-training quantization | [`optimize/modelopt/quantize`](modelopt/quantize/SKILL.md) | `checkpoint_hf` | `checkpoint_megatron` |
| Structured architecture pruning (Minitron-style search or fixed export) | [`optimize/modelopt/prune`](modelopt/prune/SKILL.md) | `checkpoint_hf` | `checkpoint_hf` |
| Teacher-student quality transfer (often after pruning or quantization) | [`optimize/modelopt/distill`](modelopt/distill/SKILL.md) | `checkpoint_hf` (teacher + student) + optional `binidx` | `checkpoint_megatron` |

The umbrella subcategory [`optimize/modelopt/`](modelopt/SKILL.md) ties the
three together.

## Decision tree

- Want a smaller numeric format only (no architecture change) → **quantize**.
- Want a smaller architecture (fewer layers / heads / FFN width) → **prune**.
- Quality dropped after compression and you want to recover it → **distill**
  (teacher = original BF16, student = compressed checkpoint).
- Hardware target is Hopper/H100 → quantize FP8.
- Hardware target is Blackwell/B200 → quantize NVFP4.
- Hardware target unknown → quantize FP8 first; NVFP4 has narrower
  serving-stack support.

## Pipeline placement

```
sft/automodel  → optimize/modelopt/quantize → eval/model_eval        # smaller serving footprint
sft/automodel  → optimize/modelopt/prune    → optimize/modelopt/distill → eval/model_eval   # smaller architecture + quality recovery
data_prep/pretrain_prep → optimize/modelopt/distill → eval/model_eval     # standalone distillation
```

## Pre-conditions

1. **A clean source checkpoint.** Optimization on a half-trained or untested
   checkpoint just propagates the problem — see
   [../patterns/convert-checkpoint-safety.md](../patterns/convert-checkpoint-safety.md).
2. **A held-out benchmark.** Quantization and pruning both move quality;
   without a baseline you can't measure the move — see
   [../patterns/eval-before-and-after-training.md](../patterns/eval-before-and-after-training.md)
   and (for sovereign deployments) [../patterns/byob-benchmark-design.md](../patterns/byob-benchmark-design.md).
3. **A merged base** if the input is a LoRA — quantize/prune/distill don't
   accept adapters directly. See
   [../patterns/peft-adapter-merge-discipline.md](../patterns/peft-adapter-merge-discipline.md).

## Workflow

1. Decide deployment hardware, serving stack, checkpoint format, and quality
   budget **before** picking the step.
2. Read the target step's `step.toml` and `config/default.yaml`.
3. Smoke with `config/tiny.yaml` (quantize/prune) or
   `args.use_mock_data=true` (distill) — these prove the wrapper, not quality.
4. For remote submission, select the profile from
   `env/env_toml/config/{lepton,slurm,dgxcloud}.yaml` or the generated env file;
   do not hardcode profile names here.
5. Run the full job on representative calibration / distillation data.
6. Convert the output if the next stage expects a different checkpoint format
   (`convert/megatron_to_hf` after quantize/distill if HF is needed).
7. Re-eval against the same benchmark used pre-optimization.

## Smoke commands

```bash
uv run nemotron steps run optimize/modelopt/quantize -c tiny --dry-run
uv run nemotron steps run optimize/modelopt/prune -c tiny --dry-run
uv run nemotron steps run optimize/modelopt/distill -c tiny --dry-run    # uses use_mock_data=true
```

## Guardrails

- Don't judge optimization quality from tiny / mock-data runs. They're
  plumbing, not evidence.
- Preserve the full-precision (BF16) source checkpoint and its eval results.
  You'll need them as the teacher if recovery is required.
- For Mamba/MoE models, check tensor-parallel divisibility before launching
  (per-step SKILL covers the specific knobs).
- Distill after pruning, not the other way around — pruning before
  distillation lets the student inherit the smaller architecture; distillation
  before pruning wastes the teacher signal on a soon-to-shrink student.
