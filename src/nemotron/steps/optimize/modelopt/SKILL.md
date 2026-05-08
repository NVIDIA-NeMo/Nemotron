---
name: nemotron-optimizer-modelopt
description: Navigate Nemotron optimize/modelopt steps for ModelOpt quantization, distillation, and pruning through Megatron-Bridge. Use when working under the ModelOpt optimization family, choosing compression order, setting calibration or distillation data, or validating optimized checkpoint outputs.
---

# ModelOpt Optimization

Use the `optimize/modelopt` family when NVIDIA Model Optimizer drives
checkpoint compression or quality recovery. The three steps share a wrapper
pattern (`script:` + `args:` + `torchrun:` + `extra_args`) — see the shared
runner at [../../_runners/modelopt.py](../../_runners/modelopt.py).

## Steps

| Step | Verb | Output |
|---|---|---|
| [`quantize`](quantize/SKILL.md) | Change numeric format (FP8 / NVFP4 / INT-AWQ) | `checkpoint_megatron` |
| [`prune`](prune/SKILL.md) | Change architecture (Minitron search or fixed export) | `checkpoint_hf` |
| [`distill`](distill/SKILL.md) | Transfer behavior from teacher to student | `checkpoint_megatron` |

## Compose order

```
prune  →  distill   # quality recovery after architecture cut
quantize → (optional) distill   # numeric format cut, recover only if quality drops
distill alone        # standalone teacher → student transfer
```

Quantize and prune are independent — pick by what you need to change (numeric
format vs architecture). Distill recovers quality after either, or runs
standalone for capability transfer.

## Workflow

1. Decide deployment hardware, checkpoint format, and quality budget first.
2. Pick the step by intent (numeric format / architecture / quality recovery).
3. Use `config/fp8.yaml` or `config/nvfp4.yaml` (quantize) when the hardware
   target is set; otherwise start from `config/default.yaml`.
4. Pass new upstream flags through `extra_args` instead of editing wrappers.
5. Keep teacher, student, calibration data, and output paths **explicit** in
   YAML — never inferred from previous outputs.

## Config nuances (across the family)

- **flag style** differs: quantize uses **hyphen** flags (`--hf-model-id`),
  prune and distill use **underscore** flags (`--hf_model_name_or_path`,
  `--teacher_hf_path`).
- **W&B logging**: distill exposes native upstream W&B flags. Quantize and
  prune don't — wire W&B through the wrapper config instead.
- **Container drift**: keep the ModelOpt checkout and installed package in
  sync. Errors like missing `warn_rank_0` or mismatched `megatron_mmlu`
  signatures are version drift, not config bugs. Re-pip-install the checkout
  before applying patches.
- **Compatibility patches** (e.g. `moe_grouped_gemm`) should be narrow and
  named after the upstream mismatch they address.
- **Calibration / distillation data is the quality lever**, not the wrapper.
  Tiny configs and mock data prove the runner; representative data proves the
  model.

## Patterns to cite

- [../../patterns/convert-checkpoint-safety.md](../../patterns/convert-checkpoint-safety.md) — quantize/prune/distill from a clean checkpoint, never from training-state files.
- [../../patterns/eval-before-and-after-training.md](../../patterns/eval-before-and-after-training.md) — bookend every optimization run with a fixed eval.
- [../../patterns/byob-benchmark-design.md](../../patterns/byob-benchmark-design.md) — for sovereign deployments, calibrate and judge against a representative held-out benchmark, not on calibration loss alone.
- [../../patterns/peft-adapter-merge-discipline.md](../../patterns/peft-adapter-merge-discipline.md) — when the input is a LoRA-trained model, merge first.

## Local files

- `optimize/modelopt/quantize/`: `step.toml`, `step.py`, `config/{default,fp8,nvfp4,tiny}.yaml`
- `optimize/modelopt/prune/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`
- `optimize/modelopt/distill/`: `step.toml`, `step.py`, `config/{default,tiny}.yaml`

## Guardrails

- Don't treat launch-validation runs as quality signals.
- Use representative calibration / distillation data before judging quality.
- Preserve the full-precision baseline checkpoint and its eval record.
- Distill after pruning (not before) when recovery is the goal.
